"""
Compute trajectory quality metrics from saved trajectory files.

Expected files per run directory
--------------------------------
- pred_tracks.npy      [T,N,2] float32
- query_xy.npy         [N,2]   int64   (optional; used to infer grid order)
- gt_visibility.npy    [T,N]   uint8/bool (optional; default visibility file)
- pred_visibility.npy  [T,N]   uint8/bool (optional; alternate visibility file)

Reported summary metrics
------------------------
- flowsmooth_tv_video_mean
- divcurl_energy_video_mean

Example
-------
python -m benchmark.compute_trajectory_quality_metrics \
  --runs_root "/path/to/trajectory_runs/" \
  --use_visibility \
  --out_json "/path/to/trajectory_quality_metrics.json"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ------------------------- Discovery / I/O -------------------------


def _discover_video_dirs(root: Path) -> List[Path]:
    """Find per-video directories containing pred_tracks.npy.

    - If root itself contains pred_tracks.npy, treat it as a single video dir.
    - Otherwise recurse and return all parents of pred_tracks.npy.
    """
    if (root / "pred_tracks.npy").exists():
        return [root]
    out: List[Path] = []
    for p in root.rglob("pred_tracks.npy"):
        out.append(p.parent)
    return sorted(set(out))


def _load_optional(path: Path) -> Optional[np.ndarray]:
    return np.load(str(path)) if path.exists() else None


def _load_tracks(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    # Some pipelines save [1,T,N,2]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"{path} must be [T,N,2], got {arr.shape}")
    return arr.astype(np.float32)


def _load_visibility(path: Path, *, T: int, N: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    vis = np.load(str(path))
    if vis.ndim == 3 and vis.shape[0] == 1:
        vis = vis[0]
    if vis.ndim != 2:
        raise ValueError(f"{path} must be [T,N], got {vis.shape}")
    Tm = min(int(T), int(vis.shape[0]))
    Nm = min(int(N), int(vis.shape[1]))
    return (vis[:Tm, :Nm] > 0.5)


# ------------------------- Grid inference -------------------------


def _factor_grid(N: int) -> Tuple[int, int]:
    """Pick (H,W) integer factors close to each other, preferring W >= H."""
    best = None
    for h in range(1, int(math.sqrt(N)) + 1):
        if N % h != 0:
            continue
        w = N // h
        score = abs(w - h)
        if best is None or score < best[0]:
            best = (score, (h, w))
    if best is None:
        raise ValueError(f"Cannot factor N={N} into an integer grid")
    h, w = best[1]
    if w < h:
        h, w = w, h
    return int(h), int(w)


def _infer_grid_and_reorder(query_xy: Optional[np.ndarray], N: int) -> Tuple[Tuple[int, int], Optional[np.ndarray]]:
    """Infer (H,W) and optional reorder index to row-major [H,W]."""
    if query_xy is None:
        return _factor_grid(N), None

    q = np.asarray(query_xy)
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError(f"query_xy must be [N,2], got {q.shape}")
    if q.shape[0] != N:
        raise ValueError(f"query_xy has N={q.shape[0]} but trajectories have N={N}")

    xs = np.unique(q[:, 0])
    ys = np.unique(q[:, 1])
    xs = np.sort(xs)
    ys = np.sort(ys)
    H, W = int(len(ys)), int(len(xs))
    if H * W != N:
        return _factor_grid(N), None

    x_to_ix = {int(x): i for i, x in enumerate(xs.astype(np.int64))}
    y_to_iy = {int(y): i for i, y in enumerate(ys.astype(np.int64))}

    grid_pos = np.zeros((N,), dtype=np.int64)
    for n in range(N):
        x = int(q[n, 0])
        y = int(q[n, 1])
        if x not in x_to_ix or y not in y_to_iy:
            return (H, W), None
        grid_pos[n] = int(y_to_iy[y]) * W + int(x_to_ix[x])

    reorder = np.empty((N,), dtype=np.int64)
    reorder[grid_pos] = np.arange(N, dtype=np.int64)
    return (H, W), reorder


# ------------------------- Metric implementations -------------------------


@dataclass
class FlowTVVideo:
    flowsmooth_tv: float


def _compute_flowsmooth_tv_video(
        tracks_thw2: np.ndarray,
        valid_pos_thw: np.ndarray,
        *,
        grid_stride: float,
) -> FlowTVVideo:
    """Per-video FlowSmoothTV: mean over time of spatial TV of flow."""
    tracks = tracks_thw2.astype(np.float32)
    valid_pos = valid_pos_thw.astype(bool)

    T, H, W, _ = tracks.shape
    if T < 2 or H < 1 or W < 1:
        return FlowTVVideo(flowsmooth_tv=float("nan"))

    flow = tracks[1:] - tracks[:-1]  # [T-1,H,W,2]
    valid_flow = valid_pos[1:] & valid_pos[:-1]  # [T-1,H,W]

    u = flow[..., 0]
    v = flow[..., 1]

    # x-derivatives (along W)
    if W >= 2:
        du_dx = (u[:, :, 1:] - u[:, :, :-1]) / float(grid_stride)
        dv_dx = (v[:, :, 1:] - v[:, :, :-1]) / float(grid_stride)
        m_dx = valid_flow[:, :, 1:] & valid_flow[:, :, :-1]
        term_x_sum = np.where(m_dx, np.abs(du_dx) + np.abs(dv_dx), 0.0).sum(axis=(1, 2)).astype(np.float64)
        term_x_cnt = m_dx.sum(axis=(1, 2)).astype(np.int64)
        term_x = np.full((T - 1,), np.nan, dtype=np.float64)
        okx = term_x_cnt > 0
        term_x[okx] = term_x_sum[okx] / term_x_cnt[okx]
    else:
        term_x = np.full((T - 1,), np.nan, dtype=np.float64)

    # y-derivatives (along H)
    if H >= 2:
        du_dy = (u[:, 1:, :] - u[:, :-1, :]) / float(grid_stride)
        dv_dy = (v[:, 1:, :] - v[:, :-1, :]) / float(grid_stride)
        m_dy = valid_flow[:, 1:, :] & valid_flow[:, :-1, :]
        term_y_sum = np.where(m_dy, np.abs(du_dy) + np.abs(dv_dy), 0.0).sum(axis=(1, 2)).astype(np.float64)
        term_y_cnt = m_dy.sum(axis=(1, 2)).astype(np.int64)
        term_y = np.full((T - 1,), np.nan, dtype=np.float64)
        oky = term_y_cnt > 0
        term_y[oky] = term_y_sum[oky] / term_y_cnt[oky]
    else:
        term_y = np.full((T - 1,), np.nan, dtype=np.float64)

    # TV per time
    tv_t = np.zeros((T - 1,), dtype=np.float64)
    has_any = np.zeros((T - 1,), dtype=bool)
    if np.any(np.isfinite(term_x)):
        okx = np.isfinite(term_x)
        tv_t[okx] += term_x[okx]
        has_any |= okx
    if np.any(np.isfinite(term_y)):
        oky = np.isfinite(term_y)
        tv_t[oky] += term_y[oky]
        has_any |= oky

    tv_t = np.where(has_any, tv_t, np.nan)
    if not np.any(np.isfinite(tv_t)):
        return FlowTVVideo(flowsmooth_tv=float("nan"))

    return FlowTVVideo(flowsmooth_tv=float(np.nanmean(tv_t)))


@dataclass
class DivCurlVideo:
    divcurl_energy: float


def _compute_divcurl_energy_video(
        tracks_thw2: np.ndarray,
        valid_pos_thw: np.ndarray,
        *,
        grid_stride: float,
) -> DivCurlVideo:
    """Per-video divergence+curl energy."""
    tracks = tracks_thw2.astype(np.float32)
    valid_pos = valid_pos_thw.astype(bool)

    T, H, W, _ = tracks.shape
    if T < 2 or H < 2 or W < 2:
        return DivCurlVideo(divcurl_energy=float("nan"))

    flow = tracks[1:] - tracks[:-1]  # [T-1,H,W,2]
    valid_flow = valid_pos[1:] & valid_pos[:-1]

    u = flow[..., 0]
    v = flow[..., 1]

    du_dx = (u[:, :, 1:] - u[:, :, :-1]) / float(grid_stride)  # [T-1,H,W-1]
    dv_dx = (v[:, :, 1:] - v[:, :, :-1]) / float(grid_stride)
    m_dx = valid_flow[:, :, 1:] & valid_flow[:, :, :-1]

    du_dy = (u[:, 1:, :] - u[:, :-1, :]) / float(grid_stride)  # [T-1,H-1,W]
    dv_dy = (v[:, 1:, :] - v[:, :-1, :]) / float(grid_stride)
    m_dy = valid_flow[:, 1:, :] & valid_flow[:, :-1, :]

    # interior cells (H-1, W-1) using forward differences
    du_dx_c = du_dx[:, :-1, :]
    dv_dx_c = dv_dx[:, :-1, :]
    du_dy_c = du_dy[:, :, :-1]
    dv_dy_c = dv_dy[:, :, :-1]

    m_cell = m_dx[:, :-1, :] & m_dy[:, :, :-1]

    div = du_dx_c + dv_dy_c
    curl = dv_dx_c - du_dy_c
    energy = (div.astype(np.float64) ** 2 + curl.astype(np.float64) ** 2)

    sum_e = np.where(m_cell, energy, 0.0).sum(axis=(1, 2))
    cnt = m_cell.sum(axis=(1, 2)).astype(np.int64)
    e_t = np.full((T - 1,), np.nan, dtype=np.float64)
    ok = cnt > 0
    e_t[ok] = sum_e[ok] / cnt[ok]

    if not np.any(np.isfinite(e_t)):
        return DivCurlVideo(divcurl_energy=float("nan"))

    return DivCurlVideo(divcurl_energy=float(np.nanmean(e_t)))


# ------------------------- Run-level computation -------------------------


def _safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def compute_metrics_for_runs_root(
        runs_root: Path,
        *,
        eval_start: int,
        eval_len: int,
        use_visibility: bool,
        visibility_name: str,
        grid_stride: float,
        quiet: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute metrics for a single run root.

    Returns:
      summary: compact dataset-level summary (no per-video list)
      per_video: list of per-video metrics dicts
    """
    video_dirs = _discover_video_dirs(runs_root)
    if not video_dirs:
        raise FileNotFoundError(f"No per-video dirs found under {runs_root} (looking for pred_tracks.npy)")

    num_missing_vis = 0
    num_processed = 0

    # Dataset aggregates (video means)
    tv_vals: List[float] = []
    dc_vals: List[float] = []

    per_video: List[Dict[str, Any]] = []

    for d in video_dirs:
        pred_p = d / "pred_tracks.npy"
        try:
            pred = _load_tracks(pred_p)
        except Exception as e:
            if not quiet:
                print(f"[warning] skipping {d}: failed to load pred_tracks.npy ({e})", file=sys.stderr)
            continue

        T, N, _ = pred.shape

        # Optional visibility
        vis = None
        vis_used = False
        if use_visibility:
            vis_path = d / visibility_name
            vis = _load_visibility(vis_path, T=T, N=N)
            if vis is None:
                num_missing_vis += 1
                if not quiet:
                    print(
                        f"[warning] {d}: --use_visibility enabled but missing '{visibility_name}'. "
                        f"Evaluating unmasked.",
                        file=sys.stderr,
                    )
            else:
                vis_used = True

        # Defensive crop
        if vis is not None:
            Tm = min(T, int(vis.shape[0]))
            Nm = min(N, int(vis.shape[1]))
        else:
            Tm, Nm = T, N
        pred = pred[:Tm, :Nm]
        if vis is not None:
            vis = vis[:Tm, :Nm]
        T, N, _ = pred.shape

        # Eval slicing
        a0 = max(0, int(eval_start))
        b0 = T if int(eval_len) <= 0 else min(T, a0 + int(eval_len))
        if b0 <= a0:
            per_video.append(
                {
                    "video": _safe_relpath(d, runs_root),
                    "run_dir": str(d),
                    "error": "empty_eval_slice",
                    "T": int(T),
                    "N": int(N),
                }
            )
            continue

        pred = pred[a0:b0]
        if vis is not None:
            vis = vis[a0:b0]
        T, N, _ = pred.shape

        # Grid inference / reorder
        qxy = _load_optional(d / "query_xy.npy")
        grid_hw, reorder = _infer_grid_and_reorder(qxy, N)
        if reorder is not None:
            pred = pred[:, reorder]
            if vis is not None:
                vis = vis[:, reorder]
        Hg, Wg = int(grid_hw[0]), int(grid_hw[1])

        # Valid positions: finite AND (optional) visible
        finite = np.isfinite(pred).all(axis=-1)
        if use_visibility and vis is not None:
            valid_pos = finite & vis
        else:
            valid_pos = finite

        # --- FlowSmoothTV / DivCurlEnergy per video ---
        can_grid = (Hg * Wg == N)
        tv_val = float("nan")
        dc_val = float("nan")
        if can_grid:
            pred_thw2 = pred.reshape(T, Hg, Wg, 2)
            valid_thw = valid_pos.reshape(T, Hg, Wg)
            tv_val = _compute_flowsmooth_tv_video(pred_thw2, valid_thw, grid_stride=float(grid_stride)).flowsmooth_tv
            dc_val = _compute_divcurl_energy_video(pred_thw2, valid_thw, grid_stride=float(grid_stride)).divcurl_energy

        if np.isfinite(tv_val):
            tv_vals.append(float(tv_val))
        if np.isfinite(dc_val):
            dc_vals.append(float(dc_val))

        per_video.append(
            {
                "video": _safe_relpath(d, runs_root),
                "run_dir": str(d),
                "T": int(T),
                "N": int(N),
                "grid_H": int(Hg),
                "grid_W": int(Wg),
                "visibility_used": bool(vis_used),
                "flowsmooth_tv": float(tv_val),
                "divcurl_energy": float(dc_val),
            }
        )
        num_processed += 1

    summary: Dict[str, Any] = {
        "runs_root": str(runs_root),
        "num_videos": int(len(video_dirs)),
        "num_videos_processed": int(num_processed),
        "eval_start": int(eval_start),
        "eval_len": int(eval_len),
        "use_visibility": bool(use_visibility),
        "visibility_name": str(visibility_name),
        "grid_stride": float(grid_stride),
        "num_missing_visibility_files": int(num_missing_vis),
        # Dataset metrics (video means)
        "flowsmooth_tv_video_mean": float(np.mean(tv_vals)) if tv_vals else float("nan"),
        "divcurl_energy_video_mean": float(np.mean(dc_vals)) if dc_vals else float("nan"),
    }

    return summary, per_video


# ------------------------- CLI -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute trajectory quality metrics from saved trajectory files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--runs_root", type=str, required=True,
                   help="Run directory or benchmark root containing per-video subdirectories.")

    # Batch mode
    p.add_argument("--batch", action="store_true",
                   help="Treat --runs_root as a benchmark root and process matching subdirectories as runs.")
    p.add_argument("--batch_glob", type=str, default="*",
                   help="Glob pattern (relative to --runs_root) selecting run directories in batch mode.")
    p.add_argument(
        "--batch_out_dir",
        type=str,
        default="",
        help="If set in batch mode, write per-run summary JSONs into this directory as <run_name>.json. "
             "If empty, write them into each run directory.",
    )
    p.add_argument(
        "--batch_out_name",
        type=str,
        default="trajectory_quality_summary.json",
        help="In batch mode when --batch_out_dir is empty, write per-run summaries as <run_dir>/<batch_out_name>.",
    )

    # Time slicing within the saved trajectories.
    p.add_argument("--eval_start", type=int, default=0,
                   help="Start frame index (inclusive) within the pred_tracks.npy trajectory array.")
    p.add_argument(
        "--eval_len",
        type=int,
        default=-1,
        help="How many frames to evaluate from --eval_start. <=0 means 'use all remaining frames'.",
    )

    # Visibility
    p.add_argument("--use_visibility", action="store_true",
                   help="Mask computations using a visibility file (default: gt_visibility.npy).")
    p.add_argument(
        "--visibility_name",
        type=str,
        default="gt_visibility.npy",
        help="Visibility filename inside each per-video dir (e.g., gt_visibility.npy or pred_visibility.npy).",
    )

    # Grid / spatial derivative settings
    p.add_argument("--grid_stride", type=float, default=32.0,
                   help="Patch-grid spacing in pixels (used to normalize spatial derivatives).")

    # Output
    p.add_argument("--out_json", type=str, default="",
                   help="Optional path to write a JSON summary. Single-run: summary plus per-video metrics. "
                        "Batch: combined batch JSON.")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce warnings (still prints summary JSON to stdout).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    runs_root = Path(args.runs_root)

    def compute_one(run_root: Path, quiet: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        return compute_metrics_for_runs_root(
            run_root,
            eval_start=int(args.eval_start),
            eval_len=int(args.eval_len),
            use_visibility=bool(args.use_visibility),
            visibility_name=str(args.visibility_name),
            grid_stride=float(args.grid_stride),
            quiet=quiet,
        )

    if not args.batch:
        # Single-run mode
        summary, per_video = compute_one(runs_root, quiet=bool(args.quiet))

        # stdout: summary only
        print(json.dumps(summary, indent=2))

        # optional full JSON to file
        if args.out_json:
            outp = Path(args.out_json)
            outp.parent.mkdir(parents=True, exist_ok=True)
            full = dict(summary)
            full["per_video"] = per_video
            with open(outp, "w") as f:
                json.dump(full, f, indent=2)
            print(f"[saved] {outp}", file=sys.stderr)
        return

    # Batch mode
    if not runs_root.exists() or not runs_root.is_dir():
        raise FileNotFoundError(f"--runs_root must be an existing directory in batch mode, got {runs_root}")

    # Candidate run roots are immediate children matching the glob
    cand = [p for p in sorted(runs_root.glob(str(args.batch_glob))) if p.is_dir()]

    run_summaries: List[Dict[str, Any]] = []
    per_run_written: List[str] = []

    batch_out_dir = Path(args.batch_out_dir) if args.batch_out_dir else None
    if batch_out_dir is not None:
        batch_out_dir.mkdir(parents=True, exist_ok=True)

    for run_root in cand:
        # Skip directories without saved trajectory files.
        if not _discover_video_dirs(run_root):
            continue
        try:
            summary, _per_video = compute_one(run_root, quiet=bool(args.quiet))
        except Exception as e:
            if not args.quiet:
                print(f"[warning] failed on run {run_root}: {e}", file=sys.stderr)
            continue

        # Write per-run summary JSON
        if batch_out_dir is None:
            outp = run_root / str(args.batch_out_name)
        else:
            outp = batch_out_dir / f"{run_root.name}.json"

        try:
            with open(outp, "w") as f:
                json.dump(summary, f, indent=2)
            per_run_written.append(str(outp))
        except Exception as e:
            if not args.quiet:
                print(f"[warning] could not write {outp}: {e}", file=sys.stderr)

        run_summaries.append({"run_name": run_root.name, **summary})

    batch_result: Dict[str, Any] = {
        "batch_root": str(runs_root),
        "num_candidate_dirs": int(len(cand)),
        "num_runs_processed": int(len(run_summaries)),
        "eval_start": int(args.eval_start),
        "eval_len": int(args.eval_len),
        "use_visibility": bool(args.use_visibility),
        "visibility_name": str(args.visibility_name),
        "grid_stride": float(args.grid_stride),
        "runs": run_summaries,
        "per_run_summary_paths": per_run_written,
    }

    # stdout: combined batch JSON
    print(json.dumps(batch_result, indent=2))

    # optional combined batch JSON to file
    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(batch_result, f, indent=2)
        print(f"[saved] {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
