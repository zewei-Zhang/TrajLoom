"""
Render qualitative future-trajectory visualizations from per-example folders.

Expected files per example folder
---------------------------------
- meta.json             image size / cond_index / optional GT metadata
- pred_tracks.npy       [T,N,2] float32
- pred_visibility.npy   [T,N] uint8/bool (optional)
- query_xy.npy          [N,2] float32 (optional; used for coloring)

For GT rendering, the script also looks for a full GT file from
meta.json["gt_meta"]["path"] or local files such as gt_full.npz/.npy.

Outputs are saved to <example_dir>/viz_traj by default.

Example
-------
python -m benchmark.render_trajectory \
  --root_dir "/path/to/benchmark_outputs/" \
  --video_id <video_id> \
  --out_dir "/path/to/render_outputs/<video_id>"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from utils.inference_utils import inbounds_visibility, _apply_oob_mask_to_vis


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


def _as_numpy(a: Any) -> np.ndarray:
    return np.asarray(a)


def _maybe_transpose_trajectories_to_tn2(arr: np.ndarray, expected_n: int) -> Optional[np.ndarray]:
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


def load_gt_trajectories_vis_from_file(
        path: Path,
        *,
        expected_n: int,
        trajectory_key: str = "",
        vis_key: str = "",
        query_key: str = "",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    meta: Dict[str, Any] = {"path": str(path)}

    if path.suffix.lower() == ".npy":
        arr = _as_numpy(np.load(str(path)))
        trajectories = _maybe_transpose_trajectories_to_tn2(arr, expected_n=expected_n)
        if trajectories is None:
            raise ValueError(f"Could not interpret GT trajectories from {path} with shape {arr.shape}")
        meta["keys"] = None
        return trajectories, None, None, meta

    if path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported GT extension: {path.suffix} (expected .npz/.npy)")

    z = np.load(str(path))
    keys = list(z.files)
    meta["keys"] = keys

    trajectory_arr: Optional[np.ndarray] = None
    if trajectory_key:
        if trajectory_key not in keys:
            raise KeyError(f"{path}: trajectory_key='{trajectory_key}' not found. keys={keys}")
        trajectory_arr = _as_numpy(z[trajectory_key])
        meta["tracks_key"] = trajectory_key
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
            raise ValueError(f"{path}: no [*,*,2] trajectory candidate found. keys={keys}")
        candidates.sort(key=lambda x: x[0], reverse=True)
        trajectory_arr = candidates[0][2]
        meta["tracks_key"] = candidates[0][1]

    trajectories_tn2 = _maybe_transpose_trajectories_to_tn2(trajectory_arr, expected_n=expected_n)
    if trajectories_tn2 is None:
        raise ValueError(f"{path}: selected trajectory array has unexpected shape {trajectory_arr.shape}")

    T, N, _ = trajectories_tn2.shape
    if N != expected_n:
        raise ValueError(f"{path}: GT trajectories N={N} != expected_n={expected_n}. Check grid settings.")

    vis_tn: Optional[np.ndarray] = None
    if vis_key:
        if vis_key not in keys:
            raise KeyError(f"{path}: vis_key='{vis_key}' not found. keys={keys}")
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
        if query_key not in keys:
            raise KeyError(f"{path}: query_key='{query_key}' not found. keys={keys}")
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

    return trajectories_tn2.astype(np.float32), (
        vis_tn.astype(np.uint8) if vis_tn is not None else None), query_xy, meta


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


def reorder_gt_to_match_query(
        gt_trajectories_tn2: np.ndarray,
        gt_vis_tn: Optional[np.ndarray],
        gt_query_xy: Optional[np.ndarray],
        query_xy_ref: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if gt_query_xy is None:
        return gt_trajectories_tn2, gt_vis_tn

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

    gt_trajectories_tn2 = gt_trajectories_tn2[:, perm]
    if gt_vis_tn is not None:
        gt_vis_tn = gt_vis_tn[:, perm]
    return gt_trajectories_tn2, gt_vis_tn


def _resolve_full_gt_path(ex_dir: Path, meta: Dict) -> Path:
    gt_meta = meta.get("gt_meta", {})
    if isinstance(gt_meta, dict):
        p = gt_meta.get("path", "")
        if p and Path(str(p)).exists():
            return Path(str(p))

    for cand in [
        ex_dir / "gt_full.npz",
        ex_dir / "gt_full.npy",
        ex_dir / "gt_tracks_full.npz",
        ex_dir / "gt_tracks_full.npy",
        ex_dir / "gt_tracks_all.npz",
        ex_dir / "gt_tracks_all.npy",
    ]:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        "Could not find FULL GT trajectory file. Expected meta.json['gt_meta']['path'] to exist, "
        "or a local gt_full(.npz/.npy) in the example directory."
    )


def _load_gt_history_from_full(
        *,
        ex_dir: Path,
        meta: Dict,
        query_xy_ref: np.ndarray,
        H: int,
        W: int,
        cond_index: Optional[int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    gt_meta = meta.get("gt_meta", {})
    if not isinstance(gt_meta, dict):
        gt_meta = {}

    gt_path = _resolve_full_gt_path(ex_dir, meta)
    expected_n = int(query_xy_ref.shape[0])

    trajectory_key = str(gt_meta.get("tracks_key", "") or "")
    vis_key = str(gt_meta.get("vis_key", "") or "")
    query_key = str(gt_meta.get("query_key", "") or "")

    gt_trajectories_full, gt_vis_full, gt_query_xy, _gt_load_meta = load_gt_trajectories_vis_from_file(
        gt_path,
        expected_n=expected_n,
        trajectory_key=trajectory_key,
        vis_key=vis_key,
        query_key=query_key,
    )

    coord_mode = str(gt_meta.get("gt_coord_mode", "pixel") or "pixel")
    swap_xy = bool(gt_meta.get("gt_swap_xy", False))
    grid_stride = int(meta.get("grid_stride", 32))
    grid_origin_default = str(meta.get("grid_origin", "topleft"))
    grid_origin_used = str(gt_meta.get("gt_grid_origin_used", grid_origin_default) or grid_origin_default)

    lb256 = _build_letterbox256(H, W)

    gt_trajectories_full_px = _apply_gt_transform(
        gt_trajectories_full,
        H=H,
        W=W,
        mode=coord_mode,
        swap_xy=swap_xy,
        grid_stride=grid_stride,
        grid_origin=grid_origin_used,
        lb256=lb256,
    ).astype(np.float32)

    gt_query_xy_px: Optional[np.ndarray] = None
    if gt_query_xy is not None:
        gt_query_xy_px = _apply_gt_transform(
            gt_query_xy,
            H=H,
            W=W,
            mode=coord_mode,
            swap_xy=swap_xy,
            grid_stride=grid_stride,
            grid_origin=grid_origin_used,
            lb256=lb256,
        ).astype(np.float32)

    gt_trajectories_full_px, gt_vis_full = reorder_gt_to_match_query(
        gt_trajectories_full_px,
        gt_vis_full,
        gt_query_xy_px,
        query_xy_ref,
    )

    if "gt_hist_slice" in gt_meta and isinstance(gt_meta["gt_hist_slice"], (list, tuple)) and len(
            gt_meta["gt_hist_slice"]) == 2:
        t0 = int(gt_meta["gt_hist_slice"][0])
        t1 = int(gt_meta["gt_hist_slice"][1])
    else:

        hist_len = int(meta.get("hist_len", 0) or 0)
        if hist_len <= 0:
            if cond_index is None:
                raise ValueError("Need meta.hist_len or --cond_index to derive history slice.")
            hist_len = int(cond_index)
        gt_time0 = int(gt_meta.get("gt_time0_used", 0) or 0)
        if cond_index is None:
            raise ValueError("Need cond_index (meta.json or --cond_index) to derive history slice.")
        hist_start = int(cond_index) - int(hist_len)
        t0 = int(hist_start) - int(gt_time0)
        t1 = int(t0 + int(hist_len))

    if t0 < 0 or t1 > int(gt_trajectories_full_px.shape[0]) or t1 <= t0:
        raise ValueError(
            f"Invalid gt_hist_slice [{t0},{t1}) for GT length {gt_trajectories_full_px.shape[0]}. "
            f"Check meta.json gt_meta."
        )

    gt_history_trajectories = gt_trajectories_full_px[t0:t1].astype(np.float32)

    if gt_vis_full is not None:
        gt_history_vis = gt_vis_full[t0:t1].astype(np.uint8)
        gt_history_vis = _apply_oob_mask_to_vis(gt_history_trajectories, gt_history_vis, W=W, H=H).astype(np.uint8)
    else:
        gt_history_vis = inbounds_visibility(gt_history_trajectories, H=H, W=W).astype(np.uint8)

    eps = 1e-3
    gt_history_trajectories = gt_history_trajectories.copy()
    gt_history_trajectories[..., 0] = np.clip(gt_history_trajectories[..., 0], 0.0, float(W - 1) - eps)
    gt_history_trajectories[..., 1] = np.clip(gt_history_trajectories[..., 1], 0.0, float(H - 1) - eps)

    return gt_history_trajectories, gt_history_vis


def _parse_rgb_color(s: str) -> Tuple[int, int, int]:
    ss = str(s).strip().lower()
    if ss in ("black", "k"):
        return 0, 0, 0
    if ss in ("gray", "grey"):
        return 60, 60, 60
    if ss in ("white", "w"):
        return 255, 255, 255
    if "," in ss:
        parts = [p.strip() for p in ss.split(",") if p.strip()]
        if len(parts) == 3:
            r, g, b = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
            r = int(np.clip(r, 0, 255))
            g = int(np.clip(g, 0, 255))
            b = int(np.clip(b, 0, 255))
            return r, g, b
    raise ValueError(f"Could not parse --box_color '{s}'. Use black/gray/white or 'R,G,B'.")


def _draw_border_box(
        img_rgb: np.ndarray,
        *,
        thickness: int,
        color_rgb: Tuple[int, int, int],
        inset: int,
) -> np.ndarray:
    t = int(max(1, thickness))
    inset = int(max(0, inset))
    H, W = img_rgb.shape[:2]
    x0, y0 = inset, inset
    x1, y1 = W - 1 - inset, H - 1 - inset
    if x1 <= x0 or y1 <= y0:
        return img_rgb

    out = img_rgb.copy()
    col = np.array(color_rgb, dtype=np.uint8)

    out[y0: min(y0 + t, H), x0: x1 + 1] = col
    out[max(y1 - t + 1, 0): y1 + 1, x0: x1 + 1] = col

    out[y0: y1 + 1, x0: min(x0 + t, W)] = col
    out[y0: y1 + 1, max(x1 - t + 1, 0): x1 + 1] = col
    return out


def _load_npy(path: Path) -> np.ndarray:
    return np.load(str(path))


def _load_npy_optional(path: Path) -> Optional[np.ndarray]:
    return np.load(str(path)) if path.exists() else None


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _infer_hw_from_trajectories(trajectories_tn2: np.ndarray) -> Tuple[int, int]:
    x = trajectories_tn2[..., 0]
    y = trajectories_tn2[..., 1]
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return 480, 832
    x = x[finite]
    y = y[finite]
    W = int(np.ceil(float(x.max()))) + 1
    H = int(np.ceil(float(y.max()))) + 1
    return max(H, 1), max(W, 1)


def _find_video_by_id(video_dir: Path, video_id: str, video_glob: str) -> Optional[Path]:
    if not video_dir.exists():
        return None

    cand = video_dir / f"{video_id}.mp4"
    if cand.exists():
        return cand

    for p in sorted(video_dir.glob(f"{video_id}.*")):
        if p.is_file():
            return p

    for p in sorted(video_dir.glob(video_glob)):
        if p.is_file() and p.stem == video_id:
            return p
    return None


def _read_video_frame_bgr(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return bgr


def _lighten_to_white(img_rgb: np.ndarray, amount: float) -> np.ndarray:
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 0:
        return img_rgb
    out = img_rgb.astype(np.float32) * (1.0 - a) + 255.0 * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _parse_rgb_triplet(s: str) -> Tuple[int, int, int]:
    s = str(s).strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB like '0,0,0' but got: {s}")
    rgb = [int(float(p)) for p in parts]
    rgb = [max(0, min(255, v)) for v in rgb]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _make_background_rgb(
        *,
        mode: str,
        H: int,
        W: int,
        video_path: Optional[Path],
        cond_index: Optional[int],
        bg_frame_index: Optional[int],
        bg_lighten: float,
        bg_solid_rgb: Tuple[int, int, int],
        bg_solid_strength: float,
) -> np.ndarray:
    mode = str(mode).lower().strip()

    if mode in ("solid", "gray", "black"):
        if mode == "black":
            rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            s = 1.0
        else:
            rgb = np.array(bg_solid_rgb, dtype=np.float32)
            s = float(np.clip(bg_solid_strength, 0.0, 1.0))

        col = 255.0 * (1.0 - s) + rgb * s
        out = np.empty((H, W, 3), dtype=np.uint8)
        out[...] = np.clip(col, 0, 255).astype(np.uint8)
        return out

    if mode == "white":
        return np.full((H, W, 3), 255, dtype=np.uint8)

    if mode == "video_last":
        if video_path is None:
            raise ValueError("background=video_last requires a video path")
        if cond_index is None:
            raise ValueError("background=video_last requires cond_index (meta.json or --cond_index)")
        frame_idx = max(int(cond_index) - 1, 0)
        bgr = _read_video_frame_bgr(video_path, frame_idx)
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _lighten_to_white(rgb, bg_lighten)

    if mode == "video_frame":
        if video_path is None:
            raise ValueError("background=video_frame requires a video path")
        if bg_frame_index is None:
            raise ValueError("background=video_frame requires --bg_frame_index")
        bgr = _read_video_frame_bgr(video_path, int(bg_frame_index))
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _lighten_to_white(rgb, bg_lighten)

    raise ValueError(f"Unknown background mode: {mode}")


def _resolve_gt_video_start_idx(meta: Dict[str, Any], cond_index: Optional[int]) -> Optional[int]:
    gt_meta = meta.get("gt_meta", {})
    if isinstance(gt_meta, dict):
        hist_slice = gt_meta.get("gt_hist_slice")
        gt_time0 = gt_meta.get("gt_time0_used")
        if isinstance(hist_slice, (list, tuple)) and len(hist_slice) == 2 and gt_time0 is not None:
            return int(gt_time0) + int(hist_slice[0])

    if cond_index is None:
        return None

    hist_len = int(meta.get("hist_len", 0) or 0)
    if hist_len <= 0:
        hist_len = int(cond_index)
    return int(cond_index) - int(hist_len)


def _save_context_frames(
        *,
        out_root: Path,
        video_path: Optional[Path],
        H: int,
        W: int,
        cond_index: Optional[int],
) -> None:
    if video_path is None:
        return
    if not Path(video_path).exists():
        print(f"[warn] video_path does not exist, skip context frames: {video_path}")
        return

    ctx_dir = out_root / "context"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    def _save_one(idx: int, name: str) -> None:
        try:
            bgr = _read_video_frame_bgr(Path(video_path), int(idx))
            if bgr.shape[:2] != (H, W):
                bgr_r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                bgr_r = bgr
            rgb = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
            imageio.imwrite(str(ctx_dir / name), rgb)
        except Exception as e:
            print(f"[warn] failed saving context frame idx={idx} to {name}: {e}")

    _save_one(0, "frame_first.png")

    meta_out = {
        "video_path": str(video_path),
        "H": int(H),
        "W": int(W),
        "cond_index": (None if cond_index is None else int(cond_index)),
        "saved": {
            "frame_first": 0,
        },
    }

    if cond_index is not None:
        hist_last = max(int(cond_index) - 1, 0)
        fut_first = max(int(cond_index), 0)
        _save_one(hist_last, "frame_hist_last.png")
        _save_one(fut_first, "frame_future_first.png")
        meta_out["saved"].update(
            {
                "frame_hist_last": hist_last,
                "frame_future_first": fut_first,
            }
        )

    try:
        (ctx_dir / "context_meta.json").write_text(json.dumps(meta_out, indent=2))
    except Exception:
        pass


def _get_cmap(name: str):
    name = str(name).strip().lower()

    if name.endswith("_y"):
        name = name[:-2]

    return matplotlib.colormaps.get_cmap(name)


def _colors_rgba_from_query(
        query_xy: np.ndarray,
        *,
        H: int,
        W: int,
        color_by: str,
        cmap_name: str,
        mix_white: float,
        gamma: float,
) -> np.ndarray:
    q = query_xy.astype(np.float32)
    N = q.shape[0]

    color_by = str(color_by).lower().strip()
    cmap = _get_cmap(cmap_name)

    if color_by in ("y", "ypos", "ypos_only"):
        y01 = np.clip(q[:, 1] / max(float(H - 1), 1.0), 0.0, 1.0)
        y01 = np.power(y01, float(gamma))
        rgba = cmap(y01)

    elif color_by in ("x", "xpos"):
        x01 = np.clip(q[:, 0] / max(float(W - 1), 1.0), 0.0, 1.0)
        x01 = np.power(x01, float(gamma))
        rgba = cmap(x01)

    elif color_by in ("xy_hsv", "hsv"):

        x01 = np.clip(q[:, 0] / max(float(W - 1), 1.0), 0.0, 1.0)
        y01 = np.clip(q[:, 1] / max(float(H - 1), 1.0), 0.0, 1.0)
        h = x01
        s = np.full_like(h, 0.85)
        v = 0.55 + 0.40 * y01
        hsv = np.stack([h, s, v], axis=-1)
        rgb = mcolors.hsv_to_rgb(hsv)
        rgba = np.concatenate([rgb, np.ones((N, 1), dtype=np.float32)], axis=-1)

    else:
        raise ValueError(f"Unknown --color_by {color_by}. Use y|x|xy_hsv")

    rgba = rgba.astype(np.float32)

    mw = float(np.clip(mix_white, 0.0, 1.0))
    rgba[:, :3] = rgba[:, :3] * (1.0 - mw) + 1.0 * mw
    rgba[:, :3] = np.clip(rgba[:, :3], 0.0, 1.0)
    rgba[:, 3] = 1.0
    return rgba


def _sanitize_vis(vis_tn: Optional[np.ndarray], T: int, N: int) -> np.ndarray:
    if vis_tn is None:
        return np.ones((T, N), dtype=bool)
    v = vis_tn
    if v.ndim == 3 and v.shape[-1] == 1:
        v = v[..., 0]
    if v.shape != (T, N):

        if v.shape == (N, T):
            v = v.T
        else:
            raise ValueError(f"vis shape {v.shape} incompatible with trajectories [T,N]=[{T},{N}]")
    return (v > 0).astype(bool)


def _finite_mask(xy: np.ndarray) -> np.ndarray:
    return np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])


def _build_segments_progressive(
        trajectories_tn2: np.ndarray,
        vis_tn: Optional[np.ndarray],
        colors_rgba: np.ndarray,
        point_idx: np.ndarray,
        t: int,
        *,
        trail_window: int,
        taper: bool,
        linewidth_head: float,
        linewidth_tail: float,
        taper_power: float,
        time_fade: bool,
        alpha_line: float,
        min_alpha: float,
        fade_gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, N, _ = trajectories_tn2.shape
    assert 0 <= t < T

    pts = point_idx
    xy = trajectories_tn2[:, pts, :].astype(np.float32)
    M = xy.shape[1]

    vis = _sanitize_vis(vis_tn, T=T, N=N)[:, pts]
    fin = _finite_mask(xy)

    K = int(max(trail_window, 0))
    if K <= 1 or t <= 0:
        return (np.zeros((0, 2, 2), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32))

    t0 = max(0, t - K)
    seg_list: List[np.ndarray] = []
    rgba_list: List[np.ndarray] = []
    lw_list: List[np.ndarray] = []

    denom = max(1, K - 1)

    for tau in range(t0, t):

        if tau + 1 > t:
            break
        vmask = vis[tau] & vis[tau + 1] & fin[tau] & fin[tau + 1]
        if not np.any(vmask):
            continue

        p0 = xy[tau][vmask]
        p1 = xy[tau + 1][vmask]
        seg = np.stack([p0, p1], axis=1)

        age = float(t - (tau + 1))
        u = 1.0 - age / float(denom)
        u = float(np.clip(u, 0.0, 1.0))

        if taper:
            lw = float(linewidth_tail) + (float(linewidth_head) - float(linewidth_tail)) * (u ** float(taper_power))
        else:
            lw = float(linewidth_head)
        lw = max(0.05, lw)

        if time_fade:
            a = float(min_alpha) + (float(alpha_line) - float(min_alpha)) * (u ** float(fade_gamma))
        else:
            a = float(alpha_line)
        a = float(np.clip(a, 0.0, 1.0))

        col = colors_rgba[pts][vmask].copy()
        col[:, 3] *= a

        seg_list.append(seg)
        rgba_list.append(col)
        lw_list.append(np.full((seg.shape[0],), lw, dtype=np.float32))

    if not seg_list:
        return (np.zeros((0, 2, 2), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32))

    segments = np.concatenate(seg_list, axis=0)
    rgba = np.concatenate(rgba_list, axis=0)
    lws = np.concatenate(lw_list, axis=0)
    return segments, rgba, lws


def _build_segments_full(
        trajectories_tn2: np.ndarray,
        vis_tn: Optional[np.ndarray],
        colors_rgba: np.ndarray,
        point_idx: np.ndarray,
        *,
        taper: bool,
        linewidth_head: float,
        linewidth_tail: float,
        taper_power: float,
        time_fade: bool,
        alpha_line: float,
        min_alpha: float,
        fade_gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, N, _ = trajectories_tn2.shape
    pts = point_idx
    xy = trajectories_tn2[:, pts, :].astype(np.float32)

    vis = _sanitize_vis(vis_tn, T=T, N=N)[:, pts]
    fin = _finite_mask(xy)

    seg_list: List[np.ndarray] = []
    rgba_list: List[np.ndarray] = []
    lw_list: List[np.ndarray] = []

    denom = max(1, T - 1)

    for tau in range(0, T - 1):
        vmask = vis[tau] & vis[tau + 1] & fin[tau] & fin[tau + 1]
        if not np.any(vmask):
            continue

        p0 = xy[tau][vmask]
        p1 = xy[tau + 1][vmask]
        seg = np.stack([p0, p1], axis=1)

        u = float((tau + 1) / float(denom))

        if taper:
            lw = float(linewidth_tail) + (float(linewidth_head) - float(linewidth_tail)) * (u ** float(taper_power))
        else:
            lw = float(linewidth_head)
        lw = max(0.05, lw)

        if time_fade:
            a = float(min_alpha) + (float(alpha_line) - float(min_alpha)) * (u ** float(fade_gamma))
        else:
            a = float(alpha_line)
        a = float(np.clip(a, 0.0, 1.0))

        col = colors_rgba[pts][vmask].copy()
        col[:, 3] *= a

        seg_list.append(seg)
        rgba_list.append(col)
        lw_list.append(np.full((seg.shape[0],), lw, dtype=np.float32))

    if not seg_list:
        return (np.zeros((0, 2, 2), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32))

    segments = np.concatenate(seg_list, axis=0)
    rgba = np.concatenate(rgba_list, axis=0)
    lws = np.concatenate(lw_list, axis=0)
    return segments, rgba, lws


def _render_canvas_rgb(
        *,
        background_rgb: np.ndarray,
        H: int,
        W: int,
        segments: np.ndarray,
        seg_rgba: np.ndarray,
        seg_lw: np.ndarray,
        head_xy: Optional[np.ndarray],
        head_rgba: Optional[np.ndarray],
        head_radius: float,
        head_alpha: float,
        render_scale: int,
        dpi: int,
) -> np.ndarray:
    rs = int(max(1, render_scale))
    Hs, Ws = int(H * rs), int(W * rs)

    fig = plt.figure(figsize=(Ws / float(dpi), Hs / float(dpi)), dpi=int(dpi))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(background_rgb, extent=[0, W, H, 0], interpolation="bilinear")

    if segments.shape[0] > 0:
        lc = LineCollection(
            segments,
            colors=seg_rgba,
            linewidths=seg_lw * float(rs),
            capstyle="round",
            joinstyle="round",
            antialiased=True,
        )
        ax.add_collection(lc)

    if head_xy is not None and head_rgba is not None and head_xy.size > 0:
        r_px = float(head_radius)
        r_pts = r_px * 72.0 / float(dpi) * float(rs)
        s = (r_pts ** 2) * np.pi

        col = head_rgba.copy()
        col[:, 3] *= float(head_alpha)
        ax.scatter(
            head_xy[:, 0],
            head_xy[:, 1],
            s=s,
            c=col,
            marker="o",
            linewidths=0,
            edgecolors="none",
        )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    fig.canvas.draw()

    if hasattr(fig.canvas, "buffer_rgba"):
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    else:
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = buf.reshape(Hs, Ws, 4)[:, :, 1:].copy()
    plt.close(fig)

    if rs != 1:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img


def _write_mp4(frames_rgb: List[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=int(fps), codec="libx264", quality=8)
    try:
        for fr in frames_rgb:
            writer.append_data(fr)
    finally:
        writer.close()


def _save_frames(frames_rgb: List[np.ndarray], frames_dir: Path, stem: str, every: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    k = int(max(1, every))
    for i, fr in enumerate(frames_rgb):
        if (i % k) != 0:
            continue
        imageio.imwrite(str(frames_dir / f"{stem}_{i:06d}.png"), fr)


def _make_montage(frames_rgb: List[np.ndarray], idxs: np.ndarray, pad: int = 10) -> np.ndarray:
    assert len(frames_rgb) > 0
    H, W = frames_rgb[0].shape[:2]
    pad = int(max(0, pad))
    cols = len(idxs)
    out_W = cols * W + (cols - 1) * pad
    out = np.full((H, out_W, 3), 255, dtype=np.uint8)

    x = 0
    for j, ti in enumerate(idxs.tolist()):
        fr = frames_rgb[int(ti)]
        out[:, x: x + W] = fr
        x += W + pad
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Render future trajectories for paper figures (white bg / last-frame bg).",
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--example_dir", type=str, default="",
                   help="Path to one example folder (<root>/<video_id>).")
    g.add_argument("--root_dir", type=str, default="",
                   help="Root dir that contains per-video subfolders.")
    p.add_argument("--video_id", type=str, default="", help="Video stem name (used with --root_dir).")

    p.add_argument("--trajectories", dest="trajectory_set", type=str, default="both_side",
                   choices=["pred", "gt", "both_side"], help="Which trajectory set to render.")
    p.add_argument("--mode", type=str, default="progressive", choices=["progressive", "static"],
                   help="progressive=video/frames, static=single images")

    p.add_argument(
        "--num_frames",
        type=int,
        default=-1,
        help="Limit rendered trajectory to first K frames (<=T). Example: 24.",
    )

    p.add_argument(
        "--background",
        type=str,
        default="white, video_last",
        help="Comma-separated list: white,video_last,video_frame. Example: white,video_last",
    )
    p.add_argument("--bg_lighten", type=float, default=0.25,
                   help="Blend video background towards white (0..1).")
    p.add_argument("--bg_frame_index", type=int, default=-1,
                   help="Used for background=video_frame")

    p.add_argument("--video_path", type=str, default="", help="Explicit video path (optional).")
    p.add_argument("--video_dir", type=str, default="", help="Directory of source videos (optional).")
    p.add_argument("--video_glob", type=str, default="*.mp4")
    p.add_argument("--cond_index", type=int, default=-1, help="Override cond_index if meta.json missing.")

    p.add_argument("--out_dir", type=str, default="", help="Output dir (default: <example_dir>/viz_traj)")

    p.add_argument("--max_points", type=int, default=390)
    p.add_argument("--sample_mode", type=str, default="linspace", choices=["linspace", "random"])
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--trail_window", type=int, default=27,
                   help="Tail length (in frames) for progressive mode.")
    p.add_argument("--taper", action="store_true", default=True, help="Taper linewidth from tail->head.")
    p.add_argument("--linewidth_head", type=float, default=3)
    p.add_argument("--linewidth_tail", type=float, default=1.2)

    p.add_argument("--thickness", type=float, default=None,
                   help="Alias for --linewidth_head (old script).")
    p.add_argument("--taper_power", type=float, default=1.6,
                   help="Higher => thinner tail, smoother easing.")

    p.add_argument("--time_fade", action="store_true", default=True, help="Fade alpha from tail->head.")
    p.add_argument("--fade_gamma", type=float, default=0.8)
    p.add_argument("--alpha_line", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=None, help="Alias for --alpha_line (old script).")
    p.add_argument("--halo", type=int, default=0, help="Ignored (old script compatibility).")
    p.add_argument("--halo_alpha", type=float, default=0.0, help="Ignored (old script compatibility).")
    p.add_argument("--temporal_smooth", type=int, default=0,
                   help="Ignored (we do NOT smooth trajectories).")
    p.add_argument("--min_alpha", type=float, default=0.20,
                   help="Minimum alpha for oldest tail segments.")

    p.add_argument("--draw_head", action="store_true", default=True, help="Draw head dots.")
    p.add_argument("--head_radius", type=float, default=2.0, help="Head dot radius in pixels.")
    p.add_argument("--head_alpha", type=float, default=1.0)
    p.add_argument("--start_radius", type=float, default=None,
                   help="Alias for --head_radius (old script).")
    p.add_argument("--start_outline", type=int, default=0, help="Ignored (old script compatibility).")
    p.add_argument("--draw_start", action="store_true", help="Alias for --draw_head (old script).")

    p.add_argument("--color_by", type=str, default="y", choices=["y", "x", "xy_hsv"])
    p.add_argument("--cmap", type=str, default="turbo")
    p.add_argument("--palette_mix_white", type=float, default=0.08)
    p.add_argument("--palette_gamma", type=float, default=1.0)

    p.add_argument("--render_scale", type=int, default=2,
                   help="Render at Nx resolution then downsample.")
    p.add_argument("--dpi", type=int, default=120)

    p.add_argument(
        "--box",
        dest="box",
        action="store_true",
        default=True,
        help="Draw a thin border box around each panel (default: on).",
    )
    p.add_argument(
        "--no_box",
        dest="box",
        action="store_false",
        help="Disable the border box.",
    )
    p.add_argument("--box_thickness", type=int, default=2, help="Border thickness in pixels.")
    p.add_argument("--box_inset", type=int, default=0,
                   help="Inset the box from the image edge (pixels).")
    p.add_argument(
        "--box_color",
        type=str,
        default="black",
        help="Border color: black/gray/white or 'R,G,B'.",
    )

    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--save_video", action="store_true", default=True, )
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--frames_every", type=int, default=1)

    p.add_argument("--montage_num", type=int, default=3)
    p.add_argument("--montage_pad", type=int, default=10)

    p.add_argument(
        "--bg_solid_rgb",
        type=str,
        default="0,0,0",
        help="RGB used for background=solid/gray/black, blended with white. Example: '0,0,0' (black).",
    )
    p.add_argument(
        "--bg_solid_strength",
        type=float,
        default=0.05,
        help="Blend strength in [0,1]. 0=white, 1=bg_solid_rgb. For subtle off-white try 0.03~0.06.",
    )

    return p.parse_args()


def _resolve_example_dir(args: argparse.Namespace) -> Path:
    if args.example_dir:
        return Path(args.example_dir)
    root = Path(args.root_dir)
    if not args.video_id:
        raise ValueError("--video_id is required when using --root_dir")
    return root / str(args.video_id)


def _pick_points(N: int, max_points: int, mode: str, seed: int) -> np.ndarray:
    if max_points is None or int(max_points) < 0 or N <= int(max_points):
        return np.arange(N, dtype=np.int64)
    k = int(max_points)
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(N, size=k, replace=False).astype(np.int64))
    return np.linspace(0, N - 1, k).astype(np.int64)


def main() -> None:
    args = parse_args()

    if getattr(args, 'thickness', None) is not None:
        args.linewidth_head = float(args.thickness)
    if getattr(args, 'alpha', None) is not None:
        args.alpha_line = float(args.alpha)
    if getattr(args, 'start_radius', None) is not None:
        args.head_radius = float(args.start_radius)
    if getattr(args, 'draw_start', False):
        args.draw_head = True

    box_color_rgb: Tuple[int, int, int] = (0, 0, 0)
    if bool(getattr(args, "box", True)):
        box_color_rgb = _parse_rgb_color(str(args.box_color))
    ex_dir = _resolve_example_dir(args)
    if not ex_dir.exists():
        raise FileNotFoundError(f"example_dir not found: {ex_dir}")

    meta = _read_json(ex_dir / "meta.json")

    pred_trajectories = _load_npy_optional(ex_dir / "pred_tracks.npy")
    pred_visibility = _load_npy_optional(ex_dir / "pred_visibility.npy")

    gt_trajectories: Optional[np.ndarray] = None
    gt_visibility: Optional[np.ndarray] = None

    if args.trajectory_set in ("pred", "both_side") and pred_trajectories is None:
        raise FileNotFoundError(f"Missing pred_tracks.npy in {ex_dir}")

    H = int(meta.get("H", 0) or 0)
    W = int(meta.get("W", 0) or 0)
    if H <= 0 or W <= 0:
        ref = pred_trajectories if pred_trajectories is not None else gt_trajectories
        assert ref is not None
        H, W = _infer_hw_from_trajectories(ref)
        print(f"[warn] meta.json missing H/W. Inferred HxW={H}x{W}")

    cond_index: Optional[int] = None
    if int(args.cond_index) >= 0:
        cond_index = int(args.cond_index)
    elif "cond_index" in meta:
        cond_index = int(meta.get("cond_index"))

    query_xy = _load_npy_optional(ex_dir / "query_xy.npy")
    if query_xy is None:
        ref = pred_trajectories if pred_trajectories is not None else gt_trajectories
        assert ref is not None
        query_xy = ref[0].astype(np.float32)
        print("[warn] query_xy.npy missing. Using first trajectory step for coloring.")
    else:
        query_xy = query_xy.astype(np.float32)

    N = int(query_xy.shape[0])

    if args.trajectory_set in ("gt", "both_side"):
        gt_trajectories, gt_visibility = _load_gt_history_from_full(
            ex_dir=ex_dir,
            meta=meta,
            query_xy_ref=query_xy,
            H=H,
            W=W,
            cond_index=cond_index,
        )
    point_idx = _pick_points(N, int(args.max_points), str(args.sample_mode), int(args.seed))

    colors_rgba = _colors_rgba_from_query(
        query_xy,
        H=H,
        W=W,
        color_by=str(args.color_by),
        cmap_name=str(args.cmap),
        mix_white=float(args.palette_mix_white),
        gamma=float(args.palette_gamma),
    )

    video_path: Optional[Path] = None
    if args.video_path:
        video_path = Path(args.video_path)
    else:
        mp = meta.get("video_path", "")
        if mp and Path(str(mp)).exists():
            video_path = Path(str(mp))
        elif args.video_dir and (args.video_id or ex_dir.name):
            vid = args.video_id if args.video_id else ex_dir.name
            video_path = _find_video_by_id(Path(args.video_dir), vid, str(args.video_glob))

    solid_rgb = _parse_rgb_triplet(args.bg_solid_rgb)
    solid_strength = float(args.bg_solid_strength)

    bg_modes = [m.strip() for m in str(args.background).split(",") if m.strip()]
    if not bg_modes:
        bg_modes = ["white"]

    bg_frame_index = None if int(args.bg_frame_index) < 0 else int(args.bg_frame_index)

    out_root = Path(args.out_dir) if args.out_dir else (ex_dir / "viz_traj")
    out_root.mkdir(parents=True, exist_ok=True)

    _save_context_frames(out_root=out_root, video_path=video_path, H=H, W=W, cond_index=cond_index)
    gt_video_start_idx = _resolve_gt_video_start_idx(meta, cond_index) if args.trajectory_set in ("gt",
                                                                                                  "both_side") else None

    def render_trajectory_set(name: str, trajectories: np.ndarray, visibility: Optional[np.ndarray],
                              bg_mode: str) -> None:
        bg_dir = out_root / f"bg_{bg_mode}"
        bg_dir.mkdir(parents=True, exist_ok=True)

        bg_rgb = _make_background_rgb(
            mode=bg_mode,
            H=H,
            W=W,
            video_path=video_path,
            cond_index=cond_index,
            bg_frame_index=bg_frame_index,
            bg_lighten=float(args.bg_lighten),
            bg_solid_rgb=solid_rgb,
            bg_solid_strength=solid_strength,
        )

        T_full = int(trajectories.shape[0])
        T = T_full
        if int(getattr(args, "num_frames", -1)) > 0:
            T = min(T_full, int(args.num_frames))
            trajectories = trajectories[:T]
            if visibility is not None:
                visibility = visibility[:T]

        seg_full, col_full, lw_full = _build_segments_full(
            trajectories,
            visibility,
            colors_rgba,
            point_idx,
            taper=bool(args.taper),
            linewidth_head=float(args.linewidth_head),
            linewidth_tail=float(args.linewidth_tail),
            taper_power=float(args.taper_power),
            time_fade=bool(args.time_fade),
            alpha_line=float(args.alpha_line),
            min_alpha=float(args.min_alpha),
            fade_gamma=float(args.fade_gamma),
        )
        full_rgb = _render_canvas_rgb(
            background_rgb=bg_rgb,
            H=H,
            W=W,
            segments=seg_full,
            seg_rgba=col_full,
            seg_lw=lw_full,
            head_xy=None,
            head_rgba=None,
            head_radius=float(args.head_radius),
            head_alpha=float(args.head_alpha),
            render_scale=int(args.render_scale),
            dpi=int(args.dpi),
        )

        if bool(getattr(args, "box", True)):
            full_rgb = _draw_border_box(
                full_rgb,
                thickness=int(args.box_thickness),
                color_rgb=box_color_rgb,
                inset=int(args.box_inset),
            )
        imageio.imwrite(str(bg_dir / f"traj_{name}_full.png"), full_rgb)

        if args.mode == "static":
            imageio.imwrite(str(bg_dir / f"traj_{name}.png"), full_rgb)
            return

        frames: List[np.ndarray] = []
        use_gt_video_background = (
                name == "gt"
                and bg_mode == "video_last"
                and video_path is not None
                and gt_video_start_idx is not None
        )
        for t in range(T):
            seg, col, lw = _build_segments_progressive(
                trajectories,
                visibility,
                colors_rgba,
                point_idx,
                t,
                trail_window=int(args.trail_window),
                taper=bool(args.taper),
                linewidth_head=float(args.linewidth_head),
                linewidth_tail=float(args.linewidth_tail),
                taper_power=float(args.taper_power),
                time_fade=bool(args.time_fade),
                alpha_line=float(args.alpha_line),
                min_alpha=float(args.min_alpha),
                fade_gamma=float(args.fade_gamma),
            )

            head_xy = None
            head_col = None
            if bool(args.draw_head):
                pts = point_idx
                xy_t = trajectories[t, pts, :].astype(np.float32)
                vmask = _sanitize_vis(visibility, T=T, N=trajectories.shape[1])[
                    t, pts] if visibility is not None else np.ones(
                    (pts.shape[0],), dtype=bool)
                fin = _finite_mask(xy_t)
                keep = vmask & fin
                head_xy = xy_t[keep]
                head_col = colors_rgba[pts][keep]

            frame_bg_rgb = bg_rgb
            if use_gt_video_background:
                frame_bg_rgb = _make_background_rgb(
                    mode="video_frame",
                    H=H,
                    W=W,
                    video_path=video_path,
                    cond_index=cond_index,
                    bg_frame_index=int(gt_video_start_idx) + t,
                    bg_lighten=float(args.bg_lighten),
                    bg_solid_rgb=solid_rgb,
                    bg_solid_strength=solid_strength,
                )

            fr = _render_canvas_rgb(
                background_rgb=frame_bg_rgb,
                H=H,
                W=W,
                segments=seg,
                seg_rgba=col,
                seg_lw=lw,
                head_xy=head_xy,
                head_rgba=head_col,
                head_radius=float(args.head_radius),
                head_alpha=float(args.head_alpha),
                render_scale=int(args.render_scale),
                dpi=int(args.dpi),
            )

            if bool(getattr(args, "box", True)):
                fr = _draw_border_box(
                    fr,
                    thickness=int(args.box_thickness),
                    color_rgb=box_color_rgb,
                    inset=int(args.box_inset),
                )
            frames.append(fr)

        imageio.imwrite(str(bg_dir / f"traj_{name}_last.png"), frames[-1])
        imageio.imwrite(str(bg_dir / f"traj_{name}.png"), frames[-1])

        if int(args.montage_num) > 0:
            idxs = np.linspace(0, T - 1, int(args.montage_num)).round().astype(np.int64)
            montage = _make_montage(frames, idxs, pad=int(args.montage_pad))
            imageio.imwrite(str(bg_dir / f"traj_{name}_montage.png"), montage)

        if bool(args.save_frames):
            _save_frames(frames, bg_dir / "frames", stem=f"traj_{name}", every=int(args.frames_every))

        if bool(args.save_video):
            _write_mp4(frames, bg_dir / f"traj_{name}.mp4", fps=int(args.fps))

    for bg_mode in bg_modes:
        if args.trajectory_set == "pred":
            assert pred_trajectories is not None
            render_trajectory_set("pred", pred_trajectories, pred_visibility, bg_mode)

        elif args.trajectory_set == "gt":
            assert gt_trajectories is not None
            render_trajectory_set("gt", gt_trajectories, gt_visibility, bg_mode)

        elif args.trajectory_set == "both_side":
            assert pred_trajectories is not None and gt_trajectories is not None

            render_trajectory_set("gt", gt_trajectories, gt_visibility, bg_mode)
            render_trajectory_set("pred", pred_trajectories, pred_visibility, bg_mode)

    print(f"[ok] Saved outputs to: {out_root}")


if __name__ == "__main__":
    main()
