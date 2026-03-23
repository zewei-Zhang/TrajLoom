"""
Run TrajLoom VAE reconstruction on the standard TrajLoom patch grid.

This script reconstructs future dense trajectories with the current
TrajLoom VAE and exports per-video benchmark artifacts on the 15x26
patch grid used across the TrajLoom evaluation flow.

Expected inputs:
  - models/trajloom_vae.py
  - a JSON config containing `data` and `trajloom_vae`
  - ground-truth track files stored as `.npz` or `.npy`

Example usage:
  Replace the placeholder paths below with your local paths.

  python "/path/to/TrajLoom/run_trajloom_vae_recon.py" \
    --config "/path/to/TrajLoom/configs/trajloom_vae_config.json" \
    --video_dir "/path/to/videos/" \
    --video_glob "*.mp4" \
    --gt_dir "/path/to/ground_truth/tracks/" \
    --out_dir "/path/to/output/" \
    --pred_len 81 \
    --save_video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
for _p in [THIS_DIR, PARENT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange

from models.trajloom_vae import TrajLoomVAE

from utils.inference_utils import (
    _apply_oob_mask_to_vis,
    auto_convert_gt_tracks_to_video_px,
    denormalize_points_torch,
    infer_gt_time0,
    inbounds_visibility,
    load_gt_tracks_vis_from_file,
    make_query_grid_xy,
    maybe_resize_frames,
    normalize_points_torch,
    parse_hw,
    read_video_rgb,
    reorder_gt_to_match_query,
    sample_dense_at_query_xy,
    save_json,
)
from utils.load_utils import (
    load_json as _load_json,
    load_trajloom_vae_from_config_file as load_trajloom_vae_from_config,
    parse_dtype as _parse_dtype,
)
from utils.trajloom_vae_utils import flat_to_dense_raw

GRID_ORIGIN = "topleft"


def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device_str} but CUDA is not available")
        return torch.device(device_str)
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Visualization helpers
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


def _write_mp4_rgb(frames_rgb: List[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=int(fps), codec="libx264", quality=8)
    try:
        for fr in frames_rgb:
            writer.append_data(fr)
    finally:
        writer.close()


def make_2col_video(
    *,
    out_mp4: Path,
    fut_frames_rgb: List[np.ndarray],
    gt_tracks_tn2: np.ndarray,
    gt_vis_tn: np.ndarray,
    pr_tracks_tn2: np.ndarray,
    pr_vis_tn: np.ndarray,
    query_xy: np.ndarray,
    fps: int,
    radius: int,
    viz_max_points: int,
    pred_label: str = "Pred",
) -> None:
    """Render a side-by-side GT vs prediction overlay video."""

    T = len(fut_frames_rgb)
    H, W = fut_frames_rgb[0].shape[:2]

    N = pr_tracks_tn2.shape[1]
    sel = np.arange(N, dtype=np.int64)
    if viz_max_points > 0 and N > viz_max_points:
        sel = np.linspace(0, N - 1, viz_max_points).astype(np.int64)

    colors_bgr = _colors_from_query_xy(query_xy[sel], W=W, H=H)

    frames_out: List[np.ndarray] = []
    for t in range(T):
        base_bgr = cv2.cvtColor(fut_frames_rgb[t], cv2.COLOR_RGB2BGR)

        col1 = _draw_tracks_one_frame_bgr(base_bgr, gt_tracks_tn2[t, sel], gt_vis_tn[t, sel], colors_bgr, radius)
        col1 = _put_label_bgr(col1, "GT")

        col2 = _draw_tracks_one_frame_bgr(base_bgr, pr_tracks_tn2[t, sel], pr_vis_tn[t, sel], colors_bgr, radius)
        col2 = _put_label_bgr(col2, pred_label)

        concat_bgr = np.concatenate([col1, col2], axis=1)
        frames_out.append(cv2.cvtColor(concat_bgr, cv2.COLOR_BGR2RGB))

    _write_mp4_rgb(frames_out, out_mp4, fps=fps)


# -----------------------------------------------------------------------------
# Per-video run
# -----------------------------------------------------------------------------


@torch.inference_mode()
def run_one_video(
        *,
        video_path: Path,
        out_root: Path,
        vae: TrajLoomVAE,
        vae_cfg: Dict[str, Any],
        device: torch.device,
        model_dtype: torch.dtype,
        raw_height: int,
        raw_width: int,
        args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run TrajLoom VAE reconstruction for a single video."""

    if int(args.grid_border) != 0:
        raise ValueError("--grid_border must be 0 to match dataset flat_to_dense_raw.")

    frames = read_video_rgb(video_path, max_frames=int(args.max_video_frames))
    resize_hw = parse_hw(args.resize_hw) if args.resize_hw else None
    frames = maybe_resize_frames(frames, resize_hw)

    T_vid = len(frames)
    H, W = frames[0].shape[:2]

    patch_size = int(args.grid_stride) if int(args.grid_stride) > 0 else int(vae_cfg["patch_size"])
    hist_len = int(args.hist_len) if int(args.hist_len) > 0 else int(vae_cfg.get("num_frames_in", 81))

    if int(args.cond_index) >= 0:
        cond_index = int(args.cond_index)
    else:
        use_overlap = bool(args.use_overlap)
        cond_index = (hist_len - 1) if use_overlap else hist_len

    pred_len = int(args.pred_len) if int(args.pred_len) > 0 else int(vae_cfg.get("num_frames_in", 81))

    if cond_index < 0 or cond_index >= T_vid:
        raise ValueError(f"cond_index={cond_index} out of range for video length {T_vid}")
    if cond_index + pred_len > T_vid:
        raise ValueError(f"Need frames[{cond_index}:{cond_index + pred_len}] but video length is {T_vid}")

    fut_frames = frames[cond_index: cond_index + pred_len]

    # Build the fixed query grid in video pixel coordinates.
    query_xy_hw2 = make_query_grid_xy(
        H=H,
        W=W,
        stride=patch_size,
        border=int(args.grid_border),
        origin=GRID_ORIGIN,
    )
    query_xy = query_xy_hw2.reshape(-1, 2)
    grid_idx = (query_xy[:, 1] * W + query_xy[:, 0]).astype(np.int64)

    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "query_xy.npy", query_xy.astype(np.int64))
    np.save(out_dir / "grid_idx.npy", grid_idx.astype(np.int64))

    # Load GT and reorder it to the same query-grid layout used for export.
    gt_dir = Path(args.gt_dir)
    cand_npz = gt_dir / f"{video_path.stem}.npz"
    cand_npy = gt_dir / f"{video_path.stem}.npy"
    gt_path = cand_npz if cand_npz.exists() else (cand_npy if cand_npy.exists() else None)
    if gt_path is None:
        raise FileNotFoundError(f"GT not found for {video_path.stem} under {gt_dir} (.npz/.npy)")

    gt_tracks_full, gt_vis_full, gt_query_xy, gt_load_meta = load_gt_tracks_vis_from_file(
        gt_path,
        expected_n=int(query_xy.shape[0]),
        tracks_key=str(args.gt_tracks_key),
        vis_key=str(args.gt_vis_key),
        query_key=str(args.gt_query_key),
    )

    gt_tracks_full_px, gt_query_xy_px, gt_coord_meta = auto_convert_gt_tracks_to_video_px(
        gt_tracks_full,
        gt_query_xy,
        H=H,
        W=W,
        grid_stride=patch_size,
        grid_origin=GRID_ORIGIN,
        prefer_mode=str(args.gt_coord_mode),
        prefer_swap=str(args.gt_swap_xy),
    )
    gt_tracks_full_px, gt_vis_full = reorder_gt_to_match_query(
        gt_tracks_full_px,
        gt_vis_full,
        gt_query_xy_px,
        query_xy_ref=query_xy,
    )

    gt_time0 = int(args.gt_time0)
    if gt_time0 < 0:
        gt_time0 = infer_gt_time0(gt_T=int(gt_tracks_full_px.shape[0]), vid_T=int(T_vid), cond_index=int(cond_index))

    fut_t0 = cond_index - gt_time0
    if fut_t0 < 0 or (fut_t0 + pred_len) > gt_tracks_full_px.shape[0]:
        raise ValueError(
            f"GT arrays too short for future slice: need gt[{fut_t0}:{fut_t0 + pred_len}] "
            f"but gt_T={gt_tracks_full_px.shape[0]}"
        )

    hist_start_vid = cond_index - hist_len
    hist_t0 = hist_start_vid - gt_time0
    if hist_start_vid < 0 or hist_t0 < 0 or (hist_t0 + hist_len) > gt_tracks_full_px.shape[0]:
        raise ValueError(
            f"GT arrays too short for history slice: need gt[{hist_t0}:{hist_t0 + hist_len}] "
            f"but gt_T={gt_tracks_full_px.shape[0]}"
        )

    gt_fut_tracks = gt_tracks_full_px[fut_t0:fut_t0 + pred_len].astype(np.float32)
    gt_hist_tracks = gt_tracks_full_px[hist_t0:hist_t0 + hist_len].astype(np.float32)

    if gt_vis_full is not None:
        gt_fut_vis = gt_vis_full[fut_t0:fut_t0 + pred_len].astype(np.uint8)
        gt_hist_vis = gt_vis_full[hist_t0:hist_t0 + hist_len].astype(np.uint8)
    else:
        gt_fut_vis = inbounds_visibility(gt_fut_tracks, H=H, W=W)
        gt_hist_vis = inbounds_visibility(gt_hist_tracks, H=H, W=W)

    gt_fut_vis = _apply_oob_mask_to_vis(gt_fut_tracks, gt_fut_vis, W=W, H=H).astype(np.uint8)
    gt_hist_vis = _apply_oob_mask_to_vis(gt_hist_tracks, gt_hist_vis, W=W, H=H).astype(np.uint8)

    np.save(out_dir / "gt_tracks.npy", gt_fut_tracks.astype(np.float32))
    np.save(out_dir / "gt_visibility.npy", gt_fut_vis.astype(np.uint8))

    # Expand sparse patch-grid tracks into a dense field before VAE reconstruction.
    fut_tracks_nt2 = torch.from_numpy(np.transpose(gt_fut_tracks, (1, 0, 2))).to(device=device, dtype=model_dtype)
    fut_vis_nt = torch.from_numpy(np.transpose(gt_fut_vis, (1, 0)).astype(np.bool_)).to(device=device)
    fut_dense_px_t, fut_dense_vis = flat_to_dense_raw(
        tracks_nt2=fut_tracks_nt2,
        vis_nt=fut_vis_nt,
        raw_hw=(H, W),
        grid_stride=patch_size,
    )
    fut_dense_n_t = normalize_points_torch(fut_dense_px_t, max_width=int(raw_width), max_height=int(raw_height))
    fut_x = rearrange(fut_dense_n_t, "t h w c -> 1 t c h w")

    # Reconstruct with the TrajLoom VAE posterior mean for deterministic export.
    q = vae.encode(fut_x)
    z = q.loc
    x_hat_n = vae.decode(z).clamp(-1.0, 1.0)

    pred_dense_px_t = denormalize_points_torch(
        rearrange(x_hat_n, "b t c h w -> b t h w c"),
        max_width=int(raw_width),
        max_height=int(raw_height),
    )[0]
    pred_dense_px = pred_dense_px_t.detach().to(dtype=torch.float32).cpu().numpy()

    pred_tracks = sample_dense_at_query_xy(pred_dense_px, query_xy)
    pred_vis = inbounds_visibility(pred_tracks, H=H, W=W)

    np.save(out_dir / "pred_tracks.npy", pred_tracks.astype(np.float32))
    np.save(out_dir / "pred_visibility.npy", pred_vis.astype(np.uint8))
    if bool(args.save_dense_debug):
        np.save(out_dir / "pred_tracks_dense.npy", pred_dense_px.astype(np.float32))

    if bool(args.save_video):
        make_2col_video(
            out_mp4=out_dir / "compare_2col.mp4",
            fut_frames_rgb=fut_frames,
            gt_tracks_tn2=gt_fut_tracks,
            gt_vis_tn=gt_fut_vis,
            pr_tracks_tn2=pred_tracks,
            pr_vis_tn=pred_vis,
            query_xy=query_xy,
            fps=int(args.fps),
            radius=int(args.viz_radius),
            viz_max_points=int(args.viz_max_points),
            pred_label="TrajLoom VAE",
        )

    meta: Dict[str, Any] = {
        "video_path": str(video_path),
        "video_len": int(T_vid),
        "H": int(H),
        "W": int(W),
        "cond_index": int(cond_index),
        "hist_len": int(hist_len),
        "pred_len": int(pred_len),
        "grid_stride": int(patch_size),
        "grid_border": int(args.grid_border),
        "grid_origin": GRID_ORIGIN,
        "num_points": int(query_xy.shape[0]),
        "baseline": "trajloom_vae_recon",
        "vae_config": vae_cfg,
        "gt_meta": {
            **gt_load_meta,
            **gt_coord_meta,
            "gt_time0_used": int(gt_time0),
            "gt_hist_slice": [int(hist_t0), int(hist_t0 + hist_len)],
            "gt_fut_slice": [int(fut_t0), int(fut_t0 + pred_len)],
        },
    }
    save_json(out_dir / "meta.json", meta)

    print(
        f"[GT] {video_path.stem}: coord_mode={meta['gt_meta'].get('gt_coord_mode')} "
        f"swap={meta['gt_meta'].get('gt_swap_xy')} gt_time0={gt_time0}"
    )
    return {"vid": video_path.stem, "out_dir": str(out_dir)}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for TrajLoom VAE reconstruction export."""

    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run TrajLoom VAE reconstruction and export TrajLoom benchmark files.",
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON config containing the current `data` and `trajloom_vae` sections.",
    )
    p.add_argument("--vae_ckpt", type=str, default="", help="Optional override for the VAE checkpoint.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--amp_dtype", type=str, default="fp16", choices=["no", "fp16", "bf16", "fp32"])

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video_path", type=str, default="")
    g.add_argument("--video_dir", type=str, default="")
    p.add_argument("--video_glob", type=str, default="*.mp4")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--max_video_frames", type=int, default=-1)
    p.add_argument("--resize_hw", type=str, default="")

    p.add_argument("--hist_len", type=int, default=-1)
    p.add_argument("--use_overlap", action="store_true")
    p.add_argument("--cond_index", type=int, default=-1)
    p.add_argument("--pred_len", type=int, default=-1)

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--grid_stride", type=int, default=-1)
    p.add_argument("--grid_border", type=int, default=0)

    p.add_argument("--gt_dir", type=str, required=True)
    p.add_argument("--gt_tracks_key", type=str, default="")
    p.add_argument("--gt_vis_key", type=str, default="")
    p.add_argument("--gt_query_key", type=str, default="")
    p.add_argument("--gt_time0", type=int, default=-1)
    p.add_argument(
        "--gt_coord_mode",
        type=str,
        default="auto",
        choices=["auto", "pixel", "unit", "norm", "grid", "base256_resize", "base256_letterbox"],
    )
    p.add_argument("--gt_swap_xy", type=str, default="0", choices=["auto", "0", "1"])

    p.add_argument("--save_video", action="store_true")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--viz_radius", type=int, default=2)
    p.add_argument("--viz_max_points", type=int, default=1024)
    p.add_argument("--save_dense_debug", action="store_true")
    return p.parse_args()


def main() -> None:
    """Load config and run TrajLoom VAE reconstruction for one or more videos."""

    args = parse_args()
    device = _resolve_device(args.device)
    model_dtype = _parse_dtype(args.amp_dtype)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    vae, vae_cfg = load_trajloom_vae_from_config(Path(args.config), args.vae_ckpt, device)
    if model_dtype in (torch.float16, torch.bfloat16):
        vae = vae.to(dtype=model_dtype)

    payload = _load_json(Path(args.config))
    if isinstance(payload, dict) and ("data" in payload) and isinstance(payload["data"], dict):
        raw_height = int(payload["data"].get("raw_height", vae_cfg.get("input_height", 480)))
        raw_width = int(payload["data"].get("raw_width", vae_cfg.get("input_width", 832)))
    else:
        raw_height = int(vae_cfg.get("input_height", 480))
        raw_width = int(vae_cfg.get("input_width", 832))

    if args.video_path:
        videos = [Path(args.video_path)]
    else:
        vids = sorted(Path(args.video_dir).glob(str(args.video_glob)))
        if not vids:
            raise FileNotFoundError(f"No videos found under {args.video_dir} with glob {args.video_glob}")
        start = max(0, int(args.start))
        vids = vids[start:]
        if int(args.limit) > 0:
            vids = vids[: int(args.limit)]
        videos = vids

    results: List[Dict[str, Any]] = []
    for vp in videos:
        out_dir = out_root / vp.stem
        if bool(args.skip_existing) and (out_dir / "pred_tracks.npy").exists():
            print(f"[skip] {vp.stem}")
            continue
        print(f"[run] {vp}")
        res = run_one_video(
            video_path=vp,
            out_root=out_root,
            vae=vae,
            vae_cfg=vae_cfg,
            device=device,
            model_dtype=model_dtype,
            raw_height=raw_height,
            raw_width=raw_width,
            args=args,
        )
        results.append(res)

    print(json.dumps({"num_done": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
