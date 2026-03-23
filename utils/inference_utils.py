"""
Utility helpers for video trajectory preprocessing, alignment, and visualization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Video IO
# -----------------------------------------------------------------------------


def read_video_rgb(path: Path, *, max_frames: int = -1) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames



def parse_hw(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in str(s).replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected 'H,W' but got '{s}'")
    return int(parts[0]), int(parts[1])



def maybe_resize_frames(frames: List[np.ndarray], out_hw: Optional[Tuple[int, int]]) -> List[np.ndarray]:
    if out_hw is None:
        return frames
    Ht, Wt = out_hw
    return [cv2.resize(fr, (Wt, Ht), interpolation=cv2.INTER_LINEAR) for fr in frames]



def frames_to_cthw(frames_rgb: List[np.ndarray]) -> torch.Tensor:
    arr = np.stack(frames_rgb, axis=0)  # [T,H,W,3] uint8
    ten = torch.from_numpy(arr).float() / 127.5 - 1.0
    return ten.permute(3, 0, 1, 2).contiguous()  # [3,T,H,W]


# -----------------------------------------------------------------------------
# Query grid
# -----------------------------------------------------------------------------


def make_query_grid_xy(H: int, W: int, *, stride: int, border: int, origin: str) -> np.ndarray:
    stride = int(stride)
    border = int(border)
    if origin not in ("topleft", "center"):
        raise ValueError("grid origin must be 'topleft' or 'center'")

    if origin == "topleft":
        ys = np.arange(border, H - border, stride, dtype=np.int64)
        xs = np.arange(border, W - border, stride, dtype=np.int64)
    else:
        off = stride // 2
        ys = np.arange(border + off, H - border, stride, dtype=np.int64)
        xs = np.arange(border + off, W - border, stride, dtype=np.int64)

    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx, yy], axis=-1).astype(np.int64)



def inbounds_visibility(tracks_tn2: np.ndarray, H: int, W: int) -> np.ndarray:
    x = tracks_tn2[..., 0]
    y = tracks_tn2[..., 1]
    vis = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= float(W - 1))
        & (y >= 0.0)
        & (y <= float(H - 1))
    )
    return vis.astype(np.uint8)


def _apply_oob_mask_to_vis(tracks_tn2: np.ndarray, vis_tn: np.ndarray, W: int, H: int) -> np.ndarray:
    """Force invisible if coords are non-finite or out of bounds."""
    x = tracks_tn2[..., 0]
    y = tracks_tn2[..., 1]
    oob = (
        ~np.isfinite(x)
        | ~np.isfinite(y)
        | (x < 0.0)
        | (x > float(W - 1))
        | (y < 0.0)
        | (y > float(H - 1))
    )
    vb = vis_tn.astype(bool)
    vb[oob] = False
    return vb.astype(vis_tn.dtype) if vis_tn.dtype != np.bool_ else vb


# -----------------------------------------------------------------------------
# GT loading + robust coordinate conversion
# -----------------------------------------------------------------------------


def _as_numpy(a: Any) -> np.ndarray:
    return np.asarray(a)



def _maybe_transpose_tracks_to_tn2(arr: np.ndarray, expected_n: int) -> Optional[np.ndarray]:
    if arr.ndim != 3 or arr.shape[-1] != 2:
        return None
    T0, N0 = arr.shape[0], arr.shape[1]
    if N0 == expected_n:
        return arr.astype(np.float32)
    if T0 == expected_n:
        return np.transpose(arr, (1, 0, 2)).astype(np.float32)
    return arr.astype(np.float32)



def _maybe_transpose_vis_to_tn(arr: np.ndarray, T: int, N: int) -> Optional[np.ndarray]:
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    if arr.shape == (T, N):
        return (arr > 0.5).astype(np.uint8)
    if arr.shape == (N, T):
        return (arr.T > 0.5).astype(np.uint8)
    return None



def _extract_query_xy(arr: np.ndarray, expected_n: int) -> Optional[np.ndarray]:
    a = arr
    if a.ndim == 3 and a.shape[1] == 1 and a.shape[2] in (2, 3):
        a = a[:, 0, :]
    if a.ndim != 2 or a.shape[0] != expected_n:
        return None
    if a.shape[1] == 2:
        return a.astype(np.float32)
    if a.shape[1] == 3:
        t = a[:, 0]
        y = a[:, 1]
        x = a[:, 2]
        if np.nanmax(t) > 5 and np.nanmax(x) <= 5:
            return a[:, :2].astype(np.float32)
        return np.stack([x, y], axis=-1).astype(np.float32)
    return None



def load_gt_tracks_vis_from_file(
    path: Path,
    *,
    expected_n: int,
    tracks_key: str = "",
    vis_key: str = "",
    query_key: str = "",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    meta: Dict[str, Any] = {"path": str(path)}

    if path.suffix.lower() == ".npy":
        arr = _as_numpy(np.load(str(path)))
        tracks = _maybe_transpose_tracks_to_tn2(arr, expected_n=expected_n)
        if tracks is None:
            raise ValueError(f"Could not interpret GT tracks from {path} with shape {arr.shape}")
        meta["keys"] = None
        return tracks, None, None, meta

    if path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported GT extension: {path.suffix} (expected .npz/.npy)")

    z = np.load(str(path))
    keys = list(z.files)
    meta["keys"] = keys

    tracks_arr: Optional[np.ndarray] = None
    if tracks_key:
        tracks_arr = _as_numpy(z[tracks_key])
        meta["tracks_key"] = tracks_key
    else:
        candidates: List[Tuple[int, str, np.ndarray]] = []
        for k in keys:
            a = _as_numpy(z[k])
            if a.ndim == 3 and a.shape[-1] == 2:
                score = 0
                if a.shape[1] == expected_n:
                    score += 10
                if a.shape[0] == expected_n:
                    score += 6
                if np.issubdtype(a.dtype, np.floating):
                    score += 1
                candidates.append((score, k, a))
        if not candidates:
            raise ValueError(f"{path}: no [*,*,2] tracks candidate found. keys={keys}")
        candidates.sort(key=lambda x: x[0], reverse=True)
        tracks_arr = candidates[0][2]
        meta["tracks_key"] = candidates[0][1]

    tracks_tn2 = _maybe_transpose_tracks_to_tn2(tracks_arr, expected_n=expected_n)
    if tracks_tn2 is None:
        raise ValueError(f"{path}: selected tracks array has unexpected shape {tracks_arr.shape}")

    T, N, _ = tracks_tn2.shape
    if N != expected_n:
        raise ValueError(f"{path}: GT tracks N={N} != expected_n={expected_n}")

    vis_tn: Optional[np.ndarray] = None
    if vis_key:
        v = _as_numpy(z[vis_key])
        vis_tn = _maybe_transpose_vis_to_tn(v, T=T, N=N)
        if vis_tn is None:
            raise ValueError(f"{path}: could not interpret vis_key '{vis_key}' with shape {v.shape}")
        meta["vis_key"] = vis_key
        if "occ" in vis_key.lower():
            vis_tn = (1 - (vis_tn > 0).astype(np.uint8)).astype(np.uint8)
    else:
        best: Tuple[int, str, np.ndarray] = (-1, "", np.zeros((1, 1), dtype=np.uint8))
        for k in keys:
            a = _as_numpy(z[k])
            vv = _maybe_transpose_vis_to_tn(a, T=T, N=N)
            if vv is None:
                continue
            score = 0
            name = k.lower()
            if "vis" in name:
                score += 5
            if "occ" in name or "occl" in name:
                score += 3
            if "mask" in name:
                score += 2
            if score > best[0]:
                best = (score, k, vv)
        if best[0] >= 0:
            vis_tn = best[2]
            meta["vis_key"] = best[1]
            if "occ" in best[1].lower():
                vis_tn = (1 - (vis_tn > 0).astype(np.uint8)).astype(np.uint8)

    query_xy: Optional[np.ndarray] = None
    if query_key:
        query_xy = _extract_query_xy(_as_numpy(z[query_key]), expected_n=expected_n)
        meta["query_key"] = query_key
    else:
        preferred = ["query_xy", "query_points", "query", "queries", "grid_xy"]
        for k in preferred:
            if k in keys:
                query_xy = _extract_query_xy(_as_numpy(z[k]), expected_n=expected_n)
                if query_xy is not None:
                    meta["query_key"] = k
                    break
        if query_xy is None:
            for k in keys:
                query_xy = _extract_query_xy(_as_numpy(z[k]), expected_n=expected_n)
                if query_xy is not None:
                    meta["query_key"] = k
                    break

    return tracks_tn2.astype(np.float32), (vis_tn.astype(np.uint8) if vis_tn is not None else None), query_xy, meta


@dataclass
class Letterbox256:
    orig_h: int
    orig_w: int
    side: int
    pad_top: int
    pad_left: int

    def base_to_orig_np(self, xy: np.ndarray) -> np.ndarray:
        xy = xy.astype(np.float32)
        s = float(self.side) / 256.0
        out = xy.copy()
        out[..., 0] *= s
        out[..., 1] *= s
        out[..., 0] -= float(self.pad_left)
        out[..., 1] -= float(self.pad_top)
        return out



def _build_letterbox256(H: int, W: int) -> Letterbox256:
    side = int(max(H, W))
    pad_top = (side - H) // 2
    pad_left = (side - W) // 2
    return Letterbox256(orig_h=H, orig_w=W, side=side, pad_top=pad_top, pad_left=pad_left)



def _score_coords_px(coords_px: np.ndarray, H: int, W: int) -> float:
    x = coords_px[..., 0].reshape(-1)
    y = coords_px[..., 1].reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return -1e9
    x = x[finite]
    y = y[finite]

    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    inside_frac = float(np.mean(inside)) if inside.size else 0.0
    if inside_frac <= 0.0:
        return inside_frac

    x_in = x[inside]
    y_in = y[inside]
    if x_in.size == 0 or y_in.size == 0:
        return inside_frac

    x_span = float(x_in.max() - x_in.min()) / max(1.0, float(W - 1))
    y_span = float(y_in.max() - y_in.min()) / max(1.0, float(H - 1))
    return inside_frac + 0.25 * (x_span + y_span)



def _apply_gt_transform(
    coords: np.ndarray,
    *,
    H: int,
    W: int,
    mode: str,
    swap_xy: bool,
    grid_stride: int,
    grid_origin: str,
    lb256: Letterbox256,
) -> np.ndarray:
    c = coords.astype(np.float32)
    if swap_xy:
        c = c[..., [1, 0]]

    mode = str(mode).lower().strip()

    if mode == "pixel":
        return c
    if mode == "unit":
        out = c.copy()
        out[..., 0] = out[..., 0] * float(W - 1)
        out[..., 1] = out[..., 1] * float(H - 1)
        return out
    if mode == "norm":
        out = c.copy()
        out[..., 0] = (out[..., 0] + 1.0) * 0.5 * float(W - 1)
        out[..., 1] = (out[..., 1] + 1.0) * 0.5 * float(H - 1)
        return out
    if mode == "grid":
        off = 0.0 if str(grid_origin) == "topleft" else float(grid_stride) * 0.5
        out = c.copy()
        out[..., 0] = out[..., 0] * float(grid_stride) + off
        out[..., 1] = out[..., 1] * float(grid_stride) + off
        return out
    if mode == "base256_resize":
        out = c.copy()
        out[..., 0] = out[..., 0] * (float(W - 1) / 255.0)
        out[..., 1] = out[..., 1] * (float(H - 1) / 255.0)
        return out
    if mode == "base256_letterbox":
        return lb256.base_to_orig_np(c)
    raise ValueError(f"Unknown gt coord mode: {mode}")



def auto_convert_gt_tracks_to_video_px(
    tracks_tn2: np.ndarray,
    query_xy_n2: Optional[np.ndarray],
    *,
    H: int,
    W: int,
    grid_stride: int,
    grid_origin: str,
    prefer_mode: str,
    prefer_swap: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    lb256 = _build_letterbox256(H, W)

    T, N, _ = tracks_tn2.shape
    t_idx = np.linspace(0, T - 1, min(T, 8)).astype(np.int64)
    n_idx = np.linspace(0, N - 1, min(N, 1024)).astype(np.int64)
    sample = tracks_tn2[np.ix_(t_idx, n_idx)].reshape(-1, 2)

    candidates: List[Tuple[float, str, bool, str]] = []

    mode_list = ["pixel", "unit", "norm", "grid", "base256_resize", "base256_letterbox"]
    if prefer_mode != "auto":
        mode_list = [prefer_mode]

    swap_list = [False, True]
    if prefer_swap == "0":
        swap_list = [False]
    elif prefer_swap == "1":
        swap_list = [True]

    origin_list = [grid_origin]
    if prefer_mode == "auto":
        origin_list = ["topleft", "center"]

    for m in mode_list:
        for s in swap_list:
            if m == "grid":
                for org in origin_list:
                    px = _apply_gt_transform(
                        sample,
                        H=H,
                        W=W,
                        mode=m,
                        swap_xy=s,
                        grid_stride=grid_stride,
                        grid_origin=org,
                        lb256=lb256,
                    )
                    sc = _score_coords_px(px, H=H, W=W)
                    candidates.append((sc, m, s, org))
            else:
                px = _apply_gt_transform(
                    sample,
                    H=H,
                    W=W,
                    mode=m,
                    swap_xy=s,
                    grid_stride=grid_stride,
                    grid_origin=grid_origin,
                    lb256=lb256,
                )
                sc = _score_coords_px(px, H=H, W=W)
                candidates.append((sc, m, s, grid_origin))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_sc, best_mode, best_swap, best_org = candidates[0]

    tracks_px = _apply_gt_transform(
        tracks_tn2,
        H=H,
        W=W,
        mode=best_mode,
        swap_xy=best_swap,
        grid_stride=grid_stride,
        grid_origin=best_org,
        lb256=lb256,
    ).astype(np.float32)

    query_px = None
    if query_xy_n2 is not None:
        query_px = _apply_gt_transform(
            query_xy_n2,
            H=H,
            W=W,
            mode=best_mode,
            swap_xy=best_swap,
            grid_stride=grid_stride,
            grid_origin=best_org,
            lb256=lb256,
        ).astype(np.float32)

    meta = {
        "gt_coord_score": float(best_sc),
        "gt_coord_mode": str(best_mode),
        "gt_swap_xy": bool(best_swap),
        "gt_grid_origin_used": str(best_org),
    }
    return tracks_px, query_px, meta



def reorder_gt_to_match_query(
    gt_tracks_tn2: np.ndarray,
    gt_vis_tn: Optional[np.ndarray],
    gt_query_xy: Optional[np.ndarray],
    query_xy_ref: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if gt_query_xy is None:
        return gt_tracks_tn2, gt_vis_tn

    ref = np.round(query_xy_ref).astype(np.int64)
    src = np.round(gt_query_xy).astype(np.int64)

    mapping: Dict[Tuple[int, int], int] = {}
    for i in range(src.shape[0]):
        mapping[(int(src[i, 0]), int(src[i, 1]))] = int(i)

    perm = np.empty((ref.shape[0],), dtype=np.int64)
    missing = 0
    for i in range(ref.shape[0]):
        key = (int(ref[i, 0]), int(ref[i, 1]))
        if key not in mapping:
            missing += 1
            perm[i] = 0
        else:
            perm[i] = mapping[key]

    if missing > 0:
        raise ValueError(
            f"GT query points do not match expected grid: missing {missing}/{ref.shape[0]}. "
            f"Check gt_coord_mode/swap_xy or ensure GT is for the same patch grid."
        )

    gt_tracks_tn2 = gt_tracks_tn2[:, perm]
    if gt_vis_tn is not None:
        gt_vis_tn = gt_vis_tn[:, perm]
    return gt_tracks_tn2, gt_vis_tn



def infer_gt_time0(gt_T: int, vid_T: int, cond_index: int) -> int:
    if gt_T == vid_T:
        return 0
    if gt_T == (vid_T - cond_index):
        return int(cond_index)
    if cond_index > 0 and gt_T == (vid_T - (cond_index - 1)):
        return int(cond_index - 1)
    return 0


# -----------------------------------------------------------------------------
# Track field conversion
# -----------------------------------------------------------------------------


def normalize_points_torch(x: torch.Tensor, *, max_width: int, max_height: int) -> torch.Tensor:
    scale = x.new_tensor([float(max_width), float(max_height)])
    return (x / scale) * 2.0 - 1.0



def denormalize_points_torch(x: torch.Tensor, *, max_width: int, max_height: int) -> torch.Tensor:
    scale = x.new_tensor([float(max_width), float(max_height)])
    return ((x + 1.0) * 0.5) * scale



def _cell_edges_from_query_1d(coords: np.ndarray, full_size: int, origin: str) -> List[Tuple[int, int]]:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 1:
        raise ValueError(f"coords must be 1D, got {coords.shape}")
    if coords.size == 0:
        return []

    cells: List[Tuple[int, int]] = []
    if origin == "topleft":
        if coords.size == 1:
            starts = np.array([0.0], dtype=np.float32)
            ends = np.array([float(full_size)], dtype=np.float32)
        else:
            step = float(np.median(np.diff(coords)))
            starts = coords
            ends = np.concatenate([coords[1:], np.array([coords[-1] + step], dtype=np.float32)])
        for s, e in zip(starts, ends):
            y0 = int(np.clip(np.floor(s + 1e-6), 0, full_size))
            y1 = int(np.clip(np.floor(e + 1e-6), 0, full_size))
            if y1 <= y0:
                y1 = min(full_size, y0 + 1)
            cells.append((y0, y1))
        return cells

    # center origin
    if coords.size == 1:
        start = 0
        end = int(full_size)
        return [(start, end)]

    mids = 0.5 * (coords[:-1] + coords[1:])
    edges = np.concatenate([
        np.array([0.0], dtype=np.float32),
        mids.astype(np.float32),
        np.array([float(full_size)], dtype=np.float32),
    ])
    for s, e in zip(edges[:-1], edges[1:]):
        y0 = int(np.clip(np.floor(s + 1e-6), 0, full_size))
        y1 = int(np.clip(np.floor(e + 1e-6), 0, full_size))
        if y1 <= y0:
            y1 = min(full_size, y0 + 1)
        cells.append((y0, y1))
    return cells



def patchgrid_to_dense_raw(
    tracks_tn2: np.ndarray,
    vis_tn: Optional[np.ndarray],
    query_xy_hw2: np.ndarray,
    *,
    raw_height: int,
    raw_width: int,
    grid_origin: str = "topleft",
) -> Tuple[np.ndarray, np.ndarray]:
    if query_xy_hw2.ndim != 3 or query_xy_hw2.shape[-1] != 2:
        raise ValueError(f"query_xy_hw2 must be [Hg,Wg,2], got {query_xy_hw2.shape}")

    T, N, _ = tracks_tn2.shape
    Hg, Wg = query_xy_hw2.shape[:2]
    if Hg * Wg != N:
        raise ValueError(f"query grid {Hg}x{Wg} does not match tracks N={N}")

    tracks_hw = tracks_tn2.reshape(T, Hg, Wg, 2).astype(np.float32)
    if vis_tn is None:
        vis_hw = np.ones((T, Hg, Wg), dtype=np.uint8)
    else:
        vis_hw = vis_tn.reshape(T, Hg, Wg).astype(np.uint8)

    x_coords = query_xy_hw2[0, :, 0].astype(np.float32)
    y_coords = query_xy_hw2[:, 0, 1].astype(np.float32)
    x_cells = _cell_edges_from_query_1d(x_coords, raw_width, grid_origin)
    y_cells = _cell_edges_from_query_1d(y_coords, raw_height, grid_origin)

    dense_tracks = np.zeros((T, raw_height, raw_width, 2), dtype=np.float32)
    dense_vis = np.zeros((T, raw_height, raw_width), dtype=np.uint8)

    for iy, (y0, y1) in enumerate(y_cells):
        for ix, (x0, x1) in enumerate(x_cells):
            dense_tracks[:, y0:y1, x0:x1, :] = tracks_hw[:, iy:iy + 1, ix:ix + 1, :]
            dense_vis[:, y0:y1, x0:x1] = vis_hw[:, iy:iy + 1, ix:ix + 1]

    return dense_tracks, dense_vis



def sample_dense_at_query_xy(dense_thwc: np.ndarray, query_xy_n2: np.ndarray) -> np.ndarray:
    T, H, W, C = dense_thwc.shape
    if C != 2:
        raise ValueError(f"dense_thwc must have last dim=2, got {dense_thwc.shape}")
    q = np.round(query_xy_n2).astype(np.int64)
    x = np.clip(q[:, 0], 0, W - 1)
    y = np.clip(q[:, 1], 0, H - 1)
    out = dense_thwc[:, y, x, :]
    return out.astype(np.float32)



def make_latent_vis_mask(vis: torch.Tensor, *, patch_size: int = 8, temp_stride: int = 4) -> torch.Tensor:
    """Match the training-time visibility pooling.

    Accepts either:
      - [B,T,H,W] bool/uint8 raw visibility
      - [B,T,N] bool/uint8 patch-grid visibility
      - [T,N] bool/uint8 patch-grid visibility

    Returns:
      - [B,T_lat,N,1] float32
    """
    if vis.dim() == 2:
        vis = vis.unsqueeze(0)

    if vis.dim() == 3:
        # already patch-grid visibility [B,T,N]
        vis = vis.bool()
        B, T, N = vis.shape
        starts = torch.arange(0, T, step=temp_stride, device=vis.device)
        blocks = []
        for s in starts.tolist():
            e = min(s + temp_stride, T)
            blocks.append(vis[:, s:e].any(dim=1))
        v = torch.stack(blocks, dim=1)  # [B,T_lat,N]
        return v.unsqueeze(-1).float()

    if vis.dim() != 4:
        raise ValueError(f"expected vis [B,T,H,W] or [B,T,N] or [T,N], got {tuple(vis.shape)}")

    vis = vis.bool()
    B, T, H, W = vis.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"H,W must be divisible by patch_size={patch_size}, got {(H, W)}")

    starts = torch.arange(0, T, step=temp_stride, device=vis.device)
    blocks = []
    for s in starts.tolist():
        e = min(s + temp_stride, T)
        blocks.append(vis[:, s:e].any(dim=1))
    vis_t = torch.stack(blocks, dim=1)  # [B,T_lat,H,W]

    Hp = H // patch_size
    Wp = W // patch_size
    v = vis_t.reshape(B, vis_t.shape[1], Hp, patch_size, Wp, patch_size)
    v = v.any(dim=3).any(dim=-1)
    v = v.reshape(B, vis_t.shape[1], Hp * Wp, 1).float()
    return v


# -----------------------------------------------------------------------------
# Captions
# -----------------------------------------------------------------------------


def load_caption_map_from_csv(
    csv_path: str,
    *,
    caption_text_col: str = "text",
    caption_video_col: str = "video_path",
    caption_vid_col: str = "videoid",
) -> Dict[str, str]:
    if not csv_path:
        return {}
    path = Path(csv_path)
    if not path.exists():
        return {}

    import csv

    out: Dict[str, str] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_vid = (row.get(caption_vid_col) or "").strip()
            if not row_vid:
                vp = (row.get(caption_video_col) or "").strip()
                if vp:
                    row_vid = Path(vp).stem
            if not row_vid:
                continue
            out[row_vid] = (row.get(caption_text_col) or "")
    return out


# -----------------------------------------------------------------------------
# Comparison video
# -----------------------------------------------------------------------------


def _colors_from_query_xy(query_xy: np.ndarray, W: int, H: int) -> np.ndarray:
    x = query_xy[:, 0].astype(np.float32)
    y = query_xy[:, 1].astype(np.float32)
    x01 = np.clip(x / max(float(W - 1), 1.0), 0.0, 1.0)
    y01 = np.clip(y / max(float(H - 1), 1.0), 0.0, 1.0)

    hsv = np.zeros((query_xy.shape[0], 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (x01 * 179.0).astype(np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = (128.0 + 127.0 * y01).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]



def _draw_tracks_one_frame_bgr(
    frame_bgr: np.ndarray,
    xy_n2: np.ndarray,
    vis_n: np.ndarray,
    colors_bgr: np.ndarray,
    radius: int,
) -> np.ndarray:
    out = frame_bgr.copy()
    vis_idx = np.flatnonzero(vis_n.astype(bool))
    for n in vis_idx:
        x, y = float(xy_n2[n, 0]), float(xy_n2[n, 1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), int(radius),
                   tuple(int(c) for c in colors_bgr[n]), thickness=-1)
    return out



def _put_label_bgr(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    out = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (pad, pad + th), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out



def write_mp4_rgb(frames_rgb: List[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=int(fps), codec="libx264", quality=8)
    try:
        for fr in frames_rgb:
            writer.append_data(fr)
    finally:
        writer.close()



def make_3col_video(
    *,
    out_mp4: Path,
    cond_rgb: np.ndarray,
    fut_frames_rgb: List[np.ndarray],
    gt_tracks_tn2: Optional[np.ndarray],
    gt_vis_tn: Optional[np.ndarray],
    pr_tracks_tn2: np.ndarray,
    pr_vis_tn: np.ndarray,
    query_xy: np.ndarray,
    fps: int,
    radius: int,
    viz_max_points: int,
    pred_label: str = "Pred",
) -> None:
    T = len(fut_frames_rgb)
    H, W = fut_frames_rgb[0].shape[:2]

    N = pr_tracks_tn2.shape[1]
    sel = np.arange(N, dtype=np.int64)
    if viz_max_points > 0 and N > viz_max_points:
        sel = np.linspace(0, N - 1, viz_max_points).astype(np.int64)

    colors_bgr = _colors_from_query_xy(query_xy[sel], W=W, H=H)

    have_gt = gt_tracks_tn2 is not None and gt_vis_tn is not None
    cond_bgr = _put_label_bgr(cv2.cvtColor(cond_rgb, cv2.COLOR_RGB2BGR), "COND")

    frames_out: List[np.ndarray] = []
    for t in range(T):
        base_bgr = cv2.cvtColor(fut_frames_rgb[t], cv2.COLOR_RGB2BGR)

        col1 = cond_bgr
        if have_gt:
            col2 = _draw_tracks_one_frame_bgr(base_bgr, gt_tracks_tn2[t, sel], gt_vis_tn[t, sel], colors_bgr, radius)
            col2 = _put_label_bgr(col2, "GT")
        else:
            col2 = _put_label_bgr(base_bgr, "GT (missing)")

        if have_gt:
            vis_draw = (gt_vis_tn[t, sel].astype(bool) & pr_vis_tn[t, sel].astype(bool)).astype(np.uint8)
            label3 = f"{pred_label} (GT vis)"
        else:
            vis_draw = pr_vis_tn[t, sel]
            label3 = pred_label

        col3 = _draw_tracks_one_frame_bgr(base_bgr, pr_tracks_tn2[t, sel], vis_draw, colors_bgr, radius)
        col3 = _put_label_bgr(col3, label3)

        concat_bgr = np.concatenate([col1, col2, col3], axis=1)
        frames_out.append(cv2.cvtColor(concat_bgr, cv2.COLOR_BGR2RGB))

    write_mp4_rgb(frames_out, out_mp4, fps=fps)


# -----------------------------------------------------------------------------
# Meta helpers
# -----------------------------------------------------------------------------


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
