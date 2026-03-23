"""
Compute trajectory-based FVMD from saved trajectory files.

Expected files per run directory
--------------------------------
The official FVMD implementation tracks keypoints from videos using PIPs++, then
computes motion histograms on velocity & acceleration and finally a Fréchet
distance between generated vs GT distributions.

In our benchmark we already have aligned per-point tracks (and visibility), so
running a tracker again is unnecessary and can introduce extra noise. This
script replicates the FVMD motion histogram + Fréchet distance computation, but
takes as input:

  - pred_tracks.npy: [T, N, 2]  (pixel coords; x,y)
  - gt_tracks.npy:   [T, N, 2]
  - visibility file: [T, N] (0/1 or prob). When enabled, we mask
    motion vectors when the chosen visibility file says the point is not visible.

Example
-------
python -m benchmark.compute_fvmd_from_trajectory \
  --runs_root "/path/to/fvmd_runs/" \
  --clip_len 24 --clip_stride 1 \
  --use_gt_visibility \
  --out_json "/path/to/fvmd_tracks.json"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import linalg


# ------------------------- FVMD core -------------------------
def _cut_subcube(
        vectors_bshw2: np.ndarray,
        *,
        cell_size: int = 5,
        cube_frames: int = 4,
) -> Tuple[np.ndarray, int, int, int]:
    """
    vectors_bshw2: [B, S, H, W, 2]

    Returns:
      subcubes: [B*MS*MH*MW, cube_frames, cell_size, cell_size, 2]
      MS, MH, MW: number of subcubes along (time, height, width)
    """
    if vectors_bshw2.ndim != 5 or vectors_bshw2.shape[-1] != 2:
        raise ValueError(f"Expected vectors [B,S,H,W,2], got {vectors_bshw2.shape}")

    B, S, H, W, _ = vectors_bshw2.shape
    MH = H // int(cell_size)
    MW = W // int(cell_size)
    MS = S // int(cube_frames)

    if MH <= 0 or MW <= 0 or MS <= 0:
        raise ValueError(
            f"Too small for cube split: S={S},H={H},W={W}, cell_size={cell_size}, cube_frames={cube_frames}"
        )

    # Trim to whole subcubes.
    vectors = vectors_bshw2[:, : MS * cube_frames, : MH * cell_size, : MW * cell_size, :]

    # [B, MS, cube_frames, MH, cell_size, MW, cell_size, 2]
    vectors = vectors.reshape(B, MS, cube_frames, MH, cell_size, MW, cell_size, 2)
    # [B, MS, MH, MW, cube_frames, cell_size, cell_size, 2]
    vectors = vectors.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    # Flatten subcubes into one batch.
    subcubes = vectors.reshape(-1, cube_frames, cell_size, cell_size, 2)
    return subcubes, MS, MH, MW


def _count_subcube_hist(
        vector_cell: np.ndarray,
        *,
        angle_bins: int = 8,
        magnitude_bins: int = 256,
) -> np.ndarray:
    """Compute the angle histogram for one spatiotemporal cell."""
    v = vector_cell.reshape(-1, 2).astype(np.float32)  # [M,2]
    # Keep the same angle convention as the reference FVMD implementation.
    ang = np.arctan2(v[:, 0], v[:, 1])  # [-pi,pi]
    bin_f = (ang + np.pi) / (2.0 * np.pi / float(angle_bins))
    bins = np.floor(bin_f).astype(np.int32)
    bins = np.clip(bins, 0, int(angle_bins) - 1)

    mag = np.linalg.norm(v, axis=1)
    mag = np.clip(mag, 0.0, float(magnitude_bins - 1))
    mag = mag + 1.0
    mag = np.log2(mag)
    mag = np.clip(mag, 0.0, float(np.log2(magnitude_bins)))
    mag = np.ceil(mag)
    mag = mag / float(np.log2(magnitude_bins))

    hist = np.bincount(bins, weights=mag, minlength=int(angle_bins)).astype(np.float32)
    return hist


def calc_hist_grid(
        vectors_bsn2: np.ndarray,
        *,
        grid_hw: Tuple[int, int],
        cell_size: int = 5,
        angle_bins: int = 8,
        cube_frames: int = 4,
        magnitude_bins: int = 256,
) -> np.ndarray:
    """
    FVMD histogram over a *rectangular* point grid.

    vectors_bsn2: [B, S, N, 2] where N = H_grid * W_grid (row-major)
    returns: [B, MS, MH, MW, angle_bins]
    """
    if vectors_bsn2.ndim != 4 or vectors_bsn2.shape[-1] != 2:
        raise ValueError(f"Expected vectors [B,S,N,2], got {vectors_bsn2.shape}")

    B, S, N, _ = vectors_bsn2.shape
    Hg, Wg = int(grid_hw[0]), int(grid_hw[1])
    if Hg * Wg != N:
        raise ValueError(f"grid_hw {grid_hw} implies N={Hg * Wg} but vectors have N={N}")

    vectors = vectors_bsn2.reshape(B, S, Hg, Wg, 2)
    subcubes, MS, MH, MW = _cut_subcube(vectors, cell_size=cell_size, cube_frames=cube_frames)

    # subcubes: [B*MS*MH*MW, cube_frames, cell_size, cell_size, 2]
    hists = np.stack(
        [
            _count_subcube_hist(sc, angle_bins=angle_bins, magnitude_bins=magnitude_bins)
            for sc in subcubes
        ],
        axis=0,
    )  # [B*MS*MH*MW, angle_bins]

    return hists.reshape(B, MS, MH, MW, int(angle_bins))


def _activation_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian mean and covariance for a batch of features."""
    x = feats.reshape(feats.shape[0], -1).astype(np.float64)
    if x.shape[0] < 2:
        raise ValueError(f"Need >=2 samples to compute covariance, got {x.shape[0]}")
    mu = np.mean(x, axis=0)
    xc = x - mu
    sigma = (xc.T @ xc) / float(x.shape[0] - 1)
    return mu, sigma


def _trace_sqrt_product(sigma1: np.ndarray, sigma2: np.ndarray, eps: float) -> float:
    """Compute Tr(sqrt(sigma1 * sigma2)) in a numerically stable way for PSD covariances.

    We use the identity:
      Tr(sqrt(sigma1 sigma2)) = Tr(sqrt(sqrt(sigma1) sigma2 sqrt(sigma1)))
    and compute eigenvalues of the symmetric PSD matrix:
      A = sqrt(sigma1) sigma2 sqrt(sigma1)
    """
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    d = sigma1.shape[0]
    eye = np.eye(d, dtype=np.float64)

    s1 = sigma1 + float(eps) * eye
    s2 = sigma2 + float(eps) * eye

    # sqrt(s1) via eigen decomposition (s1 is symmetric PSD)
    w1, v1 = np.linalg.eigh(s1)
    w1 = np.clip(w1, 0.0, None)
    sqrt_w1 = np.sqrt(w1)

    # A = sqrt(s1) @ s2 @ sqrt(s1), but avoid forming sqrt(s1) explicitly:
    # Let B = v1^T @ s2 @ v1
    B = v1.T @ (s2 @ v1)
    # Then A = diag(sqrt_w1) @ B @ diag(sqrt_w1)
    A = (sqrt_w1[:, None] * B) * sqrt_w1[None, :]

    wA = np.linalg.eigvalsh(A)  # symmetric
    wA = np.clip(wA, 0.0, None)
    return float(np.sum(np.sqrt(wA)))


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray,
                     eps: float = 1e-5) -> float:
    """Fréchet distance between two Gaussians (FVMD/FID core).

    Uses an eigen-based trace(sqrt(.)) computation for speed/stability.
    """
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)

    if mu1.shape != mu2.shape:
        raise ValueError(f"mu shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"sigma shape mismatch: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2
    tr_covmean = _trace_sqrt_product(sigma1, sigma2, eps=float(eps))
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def calculate_fd_given_vectors(feat_gt: np.ndarray, feat_pred: np.ndarray) -> float:
    m1, s1 = _activation_stats(feat_gt)
    m2, s2 = _activation_stats(feat_pred)
    return frechet_distance(m1, s1, m2, s2)


# ------------------------- Tracks -> motion vectors -------------------------
def _compute_velocity_and_acceleration(
        tracks_tn2: np.ndarray,
        *,
        vis_tn: Optional[np.ndarray] = None,
        use_vis: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    tracks_tn2: [T,N,2] (float)
    vis_tn:     [T,N] (0/1) optional

    Returns:
      vel_tn2: [T,N,2] with vel[0]=0
      acc_tn2: [T,N,2] with acc[0]=acc[1]=0
    """
    tracks = tracks_tn2.astype(np.float32)

    T, N, _ = tracks.shape
    vel = np.zeros_like(tracks, dtype=np.float32)
    if T >= 2:
        vel[1:] = tracks[1:] - tracks[:-1]

    acc = np.zeros_like(tracks, dtype=np.float32)
    if T >= 3:
        acc[2:] = vel[2:] - vel[1:-1]

    if use_vis and vis_tn is not None:
        vis = (vis_tn.astype(np.float32) > 0.5)
        # velocity valid when visible at t and t-1
        vmask = np.zeros((T, N), dtype=np.float32)
        if T >= 2:
            vmask[1:] = (vis[1:] & vis[:-1]).astype(np.float32)
        vel *= vmask[..., None]

        # acceleration valid when visible at t, t-1, t-2
        amask = np.zeros((T, N), dtype=np.float32)
        if T >= 3:
            amask[2:] = (vis[2:] & vis[1:-1] & vis[:-2]).astype(np.float32)
        acc *= amask[..., None]

    return vel, acc


# ------------------------- Grid inference & ordering -------------------------
def _factor_grid(N: int) -> Tuple[int, int]:
    """Pick a plausible (H, W) grid from N, preferring wider layouts."""
    best = None
    for h in range(1, int(math.sqrt(N)) + 1):
        if N % h != 0:
            continue
        w = N // h
        cand = (h, w)
        # choose by min aspect diff
        score = abs(w - h)
        if best is None or score < best[0]:
            best = (score, cand)
    if best is None:
        raise ValueError(f"Cannot factor N={N} into an integer grid.")
    h, w = best[1]
    if w < h:
        h, w = w, h
    return int(h), int(w)


def _infer_grid_and_reorder(query_xy: Optional[np.ndarray], N: int) -> Tuple[Tuple[int, int], Optional[np.ndarray]]:
    """
    Returns:
      grid_hw = (H,W)
      reorder_idx: array of shape [N] such that tracks[:, reorder_idx] is row-major HxW,
                   or None if we assume input is already row-major.
    """
    if query_xy is None:
        H, W = _factor_grid(N)
        return (H, W), None

    q = np.asarray(query_xy)
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError(f"query_xy must be [N,2], got {q.shape}")
    if q.shape[0] != N:
        raise ValueError(f"query_xy N={q.shape[0]} != tracks N={N}")

    xs = np.unique(q[:, 0])
    ys = np.unique(q[:, 1])
    xs = np.sort(xs)
    ys = np.sort(ys)
    H, W = int(len(ys)), int(len(xs))
    if H * W != N:
        # Fallback: could not infer a full grid from query_xy.
        H2, W2 = _factor_grid(N)
        return (H2, W2), None

    x_to_ix = {int(x): i for i, x in enumerate(xs.astype(np.int64))}
    y_to_iy = {int(y): i for i, y in enumerate(ys.astype(np.int64))}

    grid_pos = np.zeros((N,), dtype=np.int64)
    for n in range(N):
        x = int(q[n, 0])
        y = int(q[n, 1])
        if x not in x_to_ix or y not in y_to_iy:
            # fallback
            return (H, W), None
        grid_pos[n] = int(y_to_iy[y]) * W + int(x_to_ix[x])

    # Invert the mapping so tracks[:, reorder] is row-major.
    reorder = np.empty((N,), dtype=np.int64)
    reorder[grid_pos] = np.arange(N, dtype=np.int64)
    return (H, W), reorder


# ------------------------- IO helpers -------------------------
@dataclass
class RunSample:
    run_dir: Path
    gt_tracks: np.ndarray  # [T,N,2]
    pred_tracks: np.ndarray  # [T,N,2]
    gt_vis: Optional[np.ndarray]  # [T,N] or None (loaded from --visibility_name; default gt_visibility.npy)
    query_xy: Optional[np.ndarray]  # [N,2] or None


def _load_optional(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        return np.load(str(path))
    return None


def _load_run_dir(run_dir: Path, *, visibility_name: str = "gt_visibility.npy") -> RunSample:
    gt_tracks_p = run_dir / "gt_tracks.npy"
    pred_tracks_p = run_dir / "pred_tracks.npy"

    if not (gt_tracks_p.exists() and pred_tracks_p.exists()):
        raise FileNotFoundError(f"Missing gt_tracks.npy/pred_tracks.npy in {run_dir}")

    gt_tracks = np.load(str(gt_tracks_p))
    pred_tracks = np.load(str(pred_tracks_p))

    if gt_tracks.ndim == 4 and gt_tracks.shape[0] == 1:
        gt_tracks = gt_tracks[0]
    if pred_tracks.ndim == 4 and pred_tracks.shape[0] == 1:
        pred_tracks = pred_tracks[0]

    if gt_tracks.ndim != 3 or gt_tracks.shape[-1] != 2:
        raise ValueError(f"{gt_tracks_p} must be [T,N,2], got {gt_tracks.shape}")
    if pred_tracks.ndim != 3 or pred_tracks.shape[-1] != 2:
        raise ValueError(f"{pred_tracks_p} must be [T,N,2], got {pred_tracks.shape}")

    gt_vis = _load_optional(run_dir / str(visibility_name))
    if gt_vis is not None:
        if gt_vis.ndim == 3 and gt_vis.shape[0] == 1:
            gt_vis = gt_vis[0]
        if gt_vis.ndim != 2:
            raise ValueError(f"{visibility_name} must be [T,N], got {gt_vis.shape}")

    query_xy = _load_optional(run_dir / "query_xy.npy")
    if query_xy is not None:
        if query_xy.ndim != 2 or query_xy.shape[1] != 2:
            raise ValueError(f"query_xy.npy must be [N,2], got {query_xy.shape}")

    return RunSample(
        run_dir=run_dir,
        gt_tracks=gt_tracks,
        pred_tracks=pred_tracks,
        gt_vis=gt_vis,
        query_xy=query_xy,
    )


def _discover_run_dirs(root: Path) -> List[Path]:
    """Find directories containing both gt_tracks.npy and pred_tracks.npy."""
    if (root / "gt_tracks.npy").exists() and (root / "pred_tracks.npy").exists():
        return [root]
    run_dirs: List[Path] = []
    for p in root.rglob("gt_tracks.npy"):
        d = p.parent
        if (d / "pred_tracks.npy").exists():
            run_dirs.append(d)
    run_dirs = sorted(set(run_dirs))
    return run_dirs


# ------------------------- Feature extraction -------------------------
def _iter_windows(T: int, clip_len: int, clip_stride: int) -> Iterable[Tuple[int, int]]:
    """Yield (start, end) window indices, with end exclusive."""
    if clip_len <= 0 or clip_len >= T:
        yield (0, T)
        return
    if clip_stride <= 0:
        clip_stride = 1
    for s in range(0, T - clip_len + 1, clip_stride):
        yield (s, s + clip_len)


def _extract_features_from_samples(
        samples: List[RunSample],
        *,
        clip_len: int,
        clip_stride: int,
        use_gt_visibility: bool,
        cell_size: int,
        cube_frames: int,
        angle_bins: int,
        magnitude_bins: int,
        batch_size: int,
        eval_start: int = 0,
        eval_len: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Returns:
      feats_v_gt:   [B, Dv]
      feats_v_pred: [B, Dv]
      feats_a_gt:   [B, Da]
      feats_a_pred: [B, Da]
      stats: dict with counts
    """
    feats_v_gt: List[np.ndarray] = []
    feats_v_pr: List[np.ndarray] = []
    feats_a_gt: List[np.ndarray] = []
    feats_a_pr: List[np.ndarray] = []

    # Batch windows to reduce overhead.
    batch_v_gt: List[np.ndarray] = []
    batch_a_gt: List[np.ndarray] = []
    batch_v_pr: List[np.ndarray] = []
    batch_a_pr: List[np.ndarray] = []
    grid_hw: Optional[Tuple[int, int]] = None

    def flush_batch() -> None:
        nonlocal batch_v_gt, batch_a_gt, batch_v_pr, batch_a_pr, feats_v_gt, feats_v_pr, feats_a_gt, feats_a_pr, grid_hw
        if not batch_v_gt:
            return
        v_gt = np.stack(batch_v_gt, axis=0)  # [b,S,N,2]
        a_gt = np.stack(batch_a_gt, axis=0)
        v_pr = np.stack(batch_v_pr, axis=0)
        a_pr = np.stack(batch_a_pr, axis=0)
        assert grid_hw is not None

        hv_gt = calc_hist_grid(
            v_gt,
            grid_hw=grid_hw,
            cell_size=cell_size,
            angle_bins=angle_bins,
            cube_frames=cube_frames,
            magnitude_bins=magnitude_bins,
        ).reshape(v_gt.shape[0], -1)
        ha_gt = calc_hist_grid(
            a_gt,
            grid_hw=grid_hw,
            cell_size=cell_size,
            angle_bins=angle_bins,
            cube_frames=cube_frames,
            magnitude_bins=magnitude_bins,
        ).reshape(v_gt.shape[0], -1)

        hv_pr = calc_hist_grid(
            v_pr,
            grid_hw=grid_hw,
            cell_size=cell_size,
            angle_bins=angle_bins,
            cube_frames=cube_frames,
            magnitude_bins=magnitude_bins,
        ).reshape(v_pr.shape[0], -1)
        ha_pr = calc_hist_grid(
            a_pr,
            grid_hw=grid_hw,
            cell_size=cell_size,
            angle_bins=angle_bins,
            cube_frames=cube_frames,
            magnitude_bins=magnitude_bins,
        ).reshape(v_pr.shape[0], -1)

        feats_v_gt.append(hv_gt)
        feats_v_pr.append(hv_pr)
        feats_a_gt.append(ha_gt)
        feats_a_pr.append(ha_pr)

        batch_v_gt, batch_a_gt, batch_v_pr, batch_a_pr = [], [], [], []

    num_videos = 0
    num_windows = 0

    eval_start_i = int(eval_start)
    eval_len_i = int(eval_len)

    for s in samples:
        num_videos += 1
        gt = s.gt_tracks
        pr = s.pred_tracks
        if gt.shape[:2] != pr.shape[:2]:
            Tm = min(gt.shape[0], pr.shape[0])
            Nm = min(gt.shape[1], pr.shape[1])
            gt = gt[:Tm, :Nm]
            pr = pr[:Tm, :Nm]
            if s.gt_vis is not None:
                s_vis = s.gt_vis[:Tm, :Nm]
            else:
                s_vis = None
        else:
            s_vis = s.gt_vis

        T, N, _ = gt.shape

        # Optional time slicing within each saved run.
        if eval_start_i != 0 or eval_len_i > 0:
            a0 = max(0, eval_start_i)
            b0 = T if eval_len_i <= 0 else min(T, a0 + eval_len_i)
            if b0 <= a0:
                # Nothing to evaluate for this sample.
                continue
            gt = gt[a0:b0]
            pr = pr[a0:b0]
            if s_vis is not None:
                s_vis = s_vis[a0:b0]
            T, N, _ = gt.shape

        # Infer grid shape and row-major ordering.
        g_hw, reorder = _infer_grid_and_reorder(s.query_xy, N)
        if grid_hw is None:
            grid_hw = g_hw
        else:
            if grid_hw != g_hw:
                raise ValueError(
                    f"Grid mismatch across samples: first={grid_hw} but {s.run_dir} has {g_hw}. "
                    "Ensure all runs use the same query point set."
                )
        if reorder is not None:
            gt = gt[:, reorder]
            pr = pr[:, reorder]
            if s_vis is not None:
                s_vis = s_vis[:, reorder]

        for (a, b) in _iter_windows(T, clip_len=int(clip_len), clip_stride=int(clip_stride)):
            gt_w = gt[a:b]
            pr_w = pr[a:b]
            vis_w = s_vis[a:b] if (use_gt_visibility and s_vis is not None) else None

            v_gt, a_gt = _compute_velocity_and_acceleration(gt_w, vis_tn=vis_w, use_vis=use_gt_visibility)
            v_pr, a_pr = _compute_velocity_and_acceleration(pr_w, vis_tn=vis_w, use_vis=use_gt_visibility)

            batch_v_gt.append(v_gt)
            batch_a_gt.append(a_gt)
            batch_v_pr.append(v_pr)
            batch_a_pr.append(a_pr)
            num_windows += 1

            if len(batch_v_gt) >= int(batch_size):
                flush_batch()

    flush_batch()

    feats_v_gt_arr = np.concatenate(feats_v_gt, axis=0) if feats_v_gt else np.zeros((0, 1), dtype=np.float32)
    feats_v_pr_arr = np.concatenate(feats_v_pr, axis=0) if feats_v_pr else np.zeros((0, 1), dtype=np.float32)
    feats_a_gt_arr = np.concatenate(feats_a_gt, axis=0) if feats_a_gt else np.zeros((0, 1), dtype=np.float32)
    feats_a_pr_arr = np.concatenate(feats_a_pr, axis=0) if feats_a_pr else np.zeros((0, 1), dtype=np.float32)

    stats = {
        "num_videos": int(num_videos),
        "num_windows": int(num_windows),
        "grid_H": int(grid_hw[0]) if grid_hw is not None else -1,
        "grid_W": int(grid_hw[1]) if grid_hw is not None else -1,
        "feature_dim_v": int(feats_v_gt_arr.shape[1]) if feats_v_gt else -1,
        "feature_dim_a": int(feats_a_gt_arr.shape[1]) if feats_a_gt else -1,
        "feature_dim_va": int(feats_v_gt_arr.shape[1] + feats_a_gt_arr.shape[1]) if feats_v_gt and feats_a_gt else -1,
        "eval_start": int(eval_start_i),
        "eval_len": int(eval_len_i),
    }
    return feats_v_gt_arr, feats_v_pr_arr, feats_a_gt_arr, feats_a_pr_arr, stats


# ------------------------- CLI -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute track-based FVMD between pred_tracks.npy and gt_tracks.npy across a benchmark directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runs_root", type=str, required=True,
                   help="Run directory or benchmark root containing per-video subdirs.")

    # Time slicing within each saved run.
    p.add_argument(
        "--eval_start",
        type=int,
        default=0,
        help="Start frame index (inclusive) to evaluate within each run's tracks.",
    )
    p.add_argument(
        "--eval_len",
        type=int,
        default=-1,
        help="How many frames to evaluate from --eval_start. <=0 means 'use all remaining frames'.",
    )

    # Windowing for histogram extraction.
    p.add_argument("--clip_len", type=int, default=16,
                   help="Window length in frames. <=0 uses full (sliced) sequence.")
    p.add_argument("--clip_stride", type=int, default=15, help="Stride between windows.")
    p.add_argument("--use_gt_visibility", action="store_true",
                   help="Mask motion vectors using --visibility_name (default: gt_visibility.npy).")
    p.add_argument(
        "--visibility_name",
        type=str,
        default="gt_visibility.npy",
        help="Visibility filename inside each run_dir to use when --use_gt_visibility is set. "
             "Default: gt_visibility.npy. Example for predicted: pred_visibility.npy.",
    )
    p.add_argument("--cell_size", type=int, default=5)
    p.add_argument("--cube_frames", type=int, default=4)
    p.add_argument("--angle_bins", type=int, default=8)
    p.add_argument("--magnitude_bins", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64,
                   help="How many windows to process per histogram batch.")
    p.add_argument("--out_json", type=str, default="", help="Optional path to write a JSON summary.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.runs_root)

    run_dirs = _discover_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs found under {root} (looking for gt_tracks.npy + pred_tracks.npy).")

    # Warn when visibility masking is requested but the file is missing.
    if args.use_gt_visibility:
        missing = [str(d / args.visibility_name) for d in run_dirs if not (d / args.visibility_name).exists()]
        if missing:
            print(
                f"[warning] --use_gt_visibility enabled but {len(missing)}/{len(run_dirs)} "
                f"run dirs are missing '{args.visibility_name}'. "
                "Those samples will be evaluated WITHOUT visibility masking.",
                file=sys.stderr,
            )
            for p in missing[:10]:
                print(f"  missing: {p}", file=sys.stderr)
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more", file=sys.stderr)

    samples = [_load_run_dir(d, visibility_name=args.visibility_name) for d in run_dirs]
    feats_v_gt, feats_v_pr, feats_a_gt, feats_a_pr, stats = _extract_features_from_samples(
        samples,
        clip_len=int(args.clip_len),
        clip_stride=int(args.clip_stride),
        use_gt_visibility=bool(args.use_gt_visibility),
        cell_size=int(args.cell_size),
        cube_frames=int(args.cube_frames),
        angle_bins=int(args.angle_bins),
        magnitude_bins=int(args.magnitude_bins),
        batch_size=int(args.batch_size),
        eval_start=int(args.eval_start),
        eval_len=int(args.eval_len),
    )

    if feats_v_gt.shape[0] < 2 or feats_v_pr.shape[0] < 2:
        raise RuntimeError(
            f"Not enough samples to compute covariance (need >=2). Got gt={feats_v_gt.shape[0]} "
            f"pred={feats_v_pr.shape[0]}"
        )

    fvmd_vel = calculate_fd_given_vectors(feats_v_gt, feats_v_pr)
    fvmd_acc = calculate_fd_given_vectors(feats_a_gt, feats_a_pr)
    fvmd_comb = calculate_fd_given_vectors(
        np.concatenate([feats_v_gt, feats_a_gt], axis=1),
        np.concatenate([feats_v_pr, feats_a_pr], axis=1),
    )

    result = {
        "fvmd_tracks_velocity": float(fvmd_vel),
        "fvmd_tracks_acceleration": float(fvmd_acc),
        "fvmd_tracks_combine": float(fvmd_comb),
        **stats,
        "clip_len": int(args.clip_len),
        "clip_stride": int(args.clip_stride),
        "eval_start": int(args.eval_start),
        "eval_len": int(args.eval_len),
        "use_gt_visibility": bool(args.use_gt_visibility),
        "visibility_name": str(args.visibility_name),
        "cell_size": int(args.cell_size),
        "cube_frames": int(args.cube_frames),
        "angle_bins": int(args.angle_bins),
        "magnitude_bins": int(args.magnitude_bins),
        "runs_root": str(root),
    }

    print(json.dumps(result, indent=2))

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
