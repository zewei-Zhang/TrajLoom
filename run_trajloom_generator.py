"""
Run TrajLoom inference on the standard patch grid.

This script loads the TrajLoom generator, predicts future
trajectories and visibility on the fixed patch grid, and exports per-video
benchmark artifacts.

Expected inputs:
  - a generator config JSON
  - a visibility predictor config JSON
  - generator and visibility predictor checkpoints
  - input videos
  - ground-truth track files stored as `.npz` or `.npy`

Example usage:
  Replace the placeholder paths below with your local paths.

  python "/path/to/TrajLoom/run_trajloom_generator.py" \
    --gen_config "/path/to/TrajLoom/configs/trajloom_generator_config.json" \
    --gen_ckpt "/path/to/checkpoints/trajloom_generator.pt" \
    --vis_config "/path/to/TrajLoom/configs/vis_predictor_config.json" \
    --vis_ckpt "/path/to/checkpoints/visibility_predictor.pt" \
    --video_dir "/path/to/videos/" \
    --video_glob "*.mp4" \
    --gt_dir "/path/to/ground_truth/tracks/" \
    --out_dir "/path/to/output/" \
    --pred_len 81
"""
import argparse
import json
import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
for _p in [THIS_DIR, PARENT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from configs.generator_configs import FullConfig
from models.trajloom_generator import TrajLoomGenerator
from models.visibility_predictor import TrajLoomVisibilityPredictor
from utils.inference_utils import (
    _apply_oob_mask_to_vis,
    auto_convert_gt_tracks_to_video_px,
    normalize_points_torch,
    denormalize_points_torch,
    infer_gt_time0,
    inbounds_visibility,
    load_gt_tracks_vis_from_file,
    make_query_grid_xy,
    maybe_resize_frames,
    parse_hw,
    read_video_rgb,
    reorder_gt_to_match_query,
)
from utils.load_utils import (
    load_full_generator_config,
    load_state_dict_from_ckpt,
    load_trajloom_vae_from_checkpoint_cfg,
    load_vis_cfg_and_params,
    load_wan_t5,
    load_wan_video_vae,
)
from utils.dataset_utils import (
    GRID_ORIGIN,
    resize_video_bcthw,
    encode_video_latents_wan,
    encode_text_t5_pooled,
    repeat_21_to_81,
    resolve_device,
    dtype_from_arg,
    find_torch_module,
    infer_module_dtype,
    maybe_eval,
    maybe_to_device_dtype,
)
from utils.trajloom_vae_utils import (
    flat_to_dense_raw,
    init_latent_scaler,
    scale_latents,
    unscale_latents,
)


@torch.no_grad()
def _sample_euler(velocity_fn, z0: torch.Tensor, *, steps: int) -> torch.Tensor:
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    z = z0
    dt = 1.0 / float(steps)
    for i in range(steps):
        t = torch.full((), float(i) / float(steps), device=z.device, dtype=z.dtype)
        z = z + dt * velocity_fn(t, z)
    return z


@torch.no_grad()
def sample_rectified_flow(
        model: TrajLoomGenerator,
        z_hist: torch.Tensor,
        z_video: Optional[torch.Tensor],
        text_cond: Optional[torch.Tensor],
        shape: Tuple[int, int, int, int],
        hist_vis_lat: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
        noise_seed: Optional[int] = None,
        steps: int = 100,
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        fix_first_frame: bool = False,
        cfg: Optional[FullConfig] = None,
        vel_mask_lat: Optional[torch.Tensor] = None,
        return_traj: bool = False,
        t_eval: Optional[torch.Tensor] = None,
):
    device = z_hist.device
    B = int(shape[0])
    state_dtype = torch.float32
    try:
        model_dtype = next(iter(model.parameters())).dtype
    except StopIteration:
        model_dtype = torch.float32

    z_hist_md = z_hist.to(dtype=model_dtype)
    z_video_md = z_video.to(dtype=model_dtype) if z_video is not None else None
    text_cond_md = text_cond.to(dtype=model_dtype) if text_cond is not None else None

    if z0 is None:
        mode = getattr(cfg.train, "x0_mode", "noise") if cfg is not None else "noise"
        g = None
        if noise_seed is not None:
            g = torch.Generator(device=device).manual_seed(int(noise_seed))
        if mode == "noise":
            z0 = torch.randn(shape, device=device, dtype=state_dtype, generator=g)
        elif mode == "anchor_first":
            z0 = torch.randn(shape, device=device, dtype=state_dtype, generator=g)
            anchor = z_hist.to(dtype=state_dtype)[:, -1:, ...]
            noise_std = float(getattr(cfg.train, "x0_noise_std", 0.1)) if cfg is not None else 0.1
            anchor = anchor + torch.randn(anchor.shape, device=device, dtype=state_dtype, generator=g) * noise_std
            z0[:, 0:1, ...] = anchor
        else:
            raise ValueError(f"Unknown x0_mode: {mode}. Expected 'noise' or 'anchor_first'.")

    fixed0 = None
    mask = None
    if fix_first_frame:
        fixed0 = z_hist[:, -1:, ...].to(dtype=state_dtype)
        z0[:, 0:1, ...] = fixed0
        mask = torch.ones_like(z0)
        mask[:, 0:1, ...] = 0.0

    vel_mask = None
    if vel_mask_lat is not None:
        vel_mask = vel_mask_lat
        if vel_mask.dim() == 3:
            vel_mask = vel_mask.unsqueeze(-1)
        if vel_mask.dim() != 4:
            raise ValueError(f"vel_mask_lat must be [B,T,N] or [B,T,N,1], got {tuple(vel_mask.shape)}")
        if vel_mask.shape[0] != B or vel_mask.shape[1] != shape[1] or vel_mask.shape[2] != shape[2]:
            raise ValueError(
                f"vel_mask_lat shape mismatch: got {tuple(vel_mask.shape)} "
                f"expected [B={B},T={shape[1]},N={shape[2]},1]"
            )
        vel_mask = vel_mask.to(device=device, dtype=state_dtype)

    if t_eval is None:
        t_eval = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=state_dtype)
    else:
        t_eval = t_eval.to(device=device, dtype=state_dtype)
        if t_eval.dim() != 1:
            raise ValueError(f"t_eval must be 1D, got shape {tuple(t_eval.shape)}")
        if int(t_eval.numel()) < 2:
            raise ValueError(f"t_eval must have at least 2 time points, got {int(t_eval.numel())}")
        if not torch.all(t_eval[1:] >= t_eval[:-1]):
            raise ValueError("t_eval must be monotonically non-decreasing")

    hist_vis_md = None
    if torch.is_tensor(hist_vis_lat):
        hist_vis_md = hist_vis_lat.to(device=device)

    def f(t_scalar: torch.Tensor, z_state: torch.Tensor) -> torch.Tensor:
        if fixed0 is not None:
            z_in = z_state.clone()
            z_in[:, 0:1, ...] = fixed0
        else:
            z_in = z_state
        t = t_scalar.to(device=device, dtype=torch.float32).expand(B)
        t = t.clamp(1e-5, 1.0 - 1e-5)
        if vel_mask is not None:
            z_in = z_in * vel_mask
        v = model(
            z_t=z_in.to(dtype=model_dtype),
            t=t,
            z_hist_track=z_hist_md,
            hist_vis=hist_vis_md,
            z_hist_video=z_video_md,
            text_cond=text_cond_md,
        ).to(dtype=state_dtype)
        if mask is not None:
            v = v * mask
        if vel_mask is not None:
            v = v * vel_mask
        return v

    method = str(method).lower()
    if method == "euler":
        try:
            from torchdiffeq import odeint
            z_traj = odeint(f, z0, t_eval, method=method, atol=atol, rtol=rtol)
            z1 = z_traj[-1]
        except ImportError:
            z1 = _sample_euler(f, z0, steps=int(steps))
            z_traj = None
    else:
        try:
            from torchdiffeq import odeint
        except ImportError as exc:
            raise ImportError(
                f"sample_method='{method}' requires torchdiffeq. Install it or use --solver euler") from exc
        z_traj = odeint(f, z0, t_eval, method=method, atol=atol, rtol=rtol)
        z1 = z_traj[-1]

    if fixed0 is not None:
        z1[:, 0:1, ...] = fixed0

    z1_out = z1.to(dtype=z_hist.dtype)
    if return_traj:
        if z_traj is None:
            return z1_out, None, t_eval
        return z1_out, z_traj, t_eval
    return z1_out


# -----------------------------------------------------------------------------
# Core per-video run
# -----------------------------------------------------------------------------

@torch.no_grad()
def run_one_video(
        *,
        video_path: Path,
        out_root: Path,
        cfg_gen: Any,
        track_vae: torch.nn.Module,
        generator: TrajLoomGenerator,
        vis_predictor: TrajLoomVisibilityPredictor,
        vis_patch_size: int,
        vis_temp_stride: int,
        vis_threshold: float,
        video_vae: Optional[torch.nn.Module],
        t5: Optional[torch.nn.Module],
        device: torch.device,
        dtype: torch.dtype,
        args: argparse.Namespace,
        gt_dir: Path,
) -> Dict[str, Any]:
    if int(args.grid_border) != 0:
        raise ValueError("--grid_border must be 0 to match dataset flat_to_dense_raw.")

    frames = read_video_rgb(video_path, max_frames=int(args.max_video_frames))
    resize_hw = parse_hw(args.resize_hw) if args.resize_hw else None
    frames = maybe_resize_frames(frames, resize_hw)

    T_vid = len(frames)
    H, W = frames[0].shape[:2]

    cond_index = int(args.cond_index)
    if cond_index < 0 or cond_index >= T_vid:
        raise ValueError(f"--cond_index {cond_index} out of range for video length {T_vid}")

    # generator config
    hist_len = int(args.hist_len) if int(getattr(args, "hist_len", -1)) > 0 else int(
        getattr(cfg_gen.data, "hist_len", 81))
    future_len = int(getattr(cfg_gen.data, "future_len", 81))
    use_overlap = bool(getattr(cfg_gen.data, "use_overlap", False))

    save_offset = 1 if use_overlap else 0
    max_save_len = int(future_len - save_offset)
    if max_save_len <= 0:
        raise ValueError(
            f"Invalid future_len={future_len} with use_overlap={use_overlap} => max_save_len={max_save_len}")

    pred_len = int(args.pred_len) if int(args.pred_len) > 0 else max_save_len
    if pred_len > max_save_len:
        raise ValueError(
            f"--pred_len {pred_len} > max_save_len {max_save_len} "
            f"(future_len={future_len}, use_overlap={use_overlap})."
        )

    # history slice in VIDEO indices (end is cond_index exclusive)
    hist_start = cond_index - hist_len
    hist_end = cond_index
    if hist_start < 0:
        raise ValueError(f"Need hist_start={hist_start} >= 0. Either lower hist_len or increase cond_index.")

    # model's future slice in VIDEO indices (may include overlap)
    fut_start = cond_index - 1 if use_overlap else cond_index
    fut_end = fut_start + future_len
    if fut_end > T_vid:
        raise ValueError(
            f"Need future frames [{fut_start}:{fut_end}] but video length is {T_vid}. "
            f"(cond_index={cond_index}, future_len={future_len}, use_overlap={use_overlap})"
        )

    # evaluation/saved future frames (WHN convention)
    eval_start = cond_index
    eval_end = cond_index + pred_len
    if eval_end > T_vid:
        raise ValueError(f"Need eval frames [{eval_start}:{eval_end}] but video length is {T_vid}.")

    # Build query grid in VIDEO coords
    query_xy_hw2 = make_query_grid_xy(
        H=H,
        W=W,
        stride=int(args.grid_stride),
        border=int(args.grid_border),
        origin=GRID_ORIGIN,
    )
    query_xy = query_xy_hw2.reshape(-1, 2)
    grid_idx = (query_xy[:, 1] * W + query_xy[:, 0]).astype(np.int64)
    N = int(query_xy.shape[0])

    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save query info
    np.save(out_dir / "query_xy.npy", query_xy.astype(np.int64))
    np.save(out_dir / "grid_idx.npy", grid_idx.astype(np.int64))

    # -------------------------
    # Load GT (full)
    # -------------------------
    cand_npz = gt_dir / f"{video_path.stem}.npz"
    cand_npy = gt_dir / f"{video_path.stem}.npy"
    gt_path = cand_npz if cand_npz.exists() else (cand_npy if cand_npy.exists() else None)
    if gt_path is None:
        raise FileNotFoundError(f"GT not found for {video_path.stem}: tried {cand_npz} and {cand_npy}")

    gt_tracks_full, gt_vis_full, gt_query_xy, gt_load_meta = load_gt_tracks_vis_from_file(
        gt_path,
        expected_n=N,
        tracks_key=str(args.gt_tracks_key),
        vis_key=str(args.gt_vis_key),
        query_key=str(args.gt_query_key),
    )

    # Convert coords to VIDEO px
    gt_tracks_full_px, gt_query_xy_px, gt_coord_meta = auto_convert_gt_tracks_to_video_px(
        gt_tracks_full,
        gt_query_xy,
        H=H,
        W=W,
        grid_stride=int(args.grid_stride),
        grid_origin=GRID_ORIGIN,
        prefer_mode=str(args.gt_coord_mode),
        prefer_swap=str(args.gt_swap_xy),
    )

    # Reorder if possible
    gt_tracks_full_px, gt_vis_full = reorder_gt_to_match_query(
        gt_tracks_full_px,
        gt_vis_full,
        gt_query_xy_px,
        query_xy_ref=query_xy,
    )

    # Time alignment
    gt_time0 = int(args.gt_time0)
    if gt_time0 < 0:
        gt_time0 = infer_gt_time0(gt_T=int(gt_tracks_full_px.shape[0]), vid_T=int(T_vid), cond_index=int(cond_index))

    # history slice in GT indices
    gt_hist_t0 = hist_start - gt_time0
    gt_hist_t1 = gt_hist_t0 + hist_len

    # model future slice in GT indices
    gt_fut_t0 = fut_start - gt_time0
    gt_fut_t1 = gt_fut_t0 + future_len

    # eval slice in GT indices
    gt_eval_t0 = eval_start - gt_time0
    gt_eval_t1 = gt_eval_t0 + pred_len

    # Validate availability
    gt_T = int(gt_tracks_full_px.shape[0])
    if gt_hist_t0 < 0 or gt_hist_t1 > gt_T:
        raise ValueError(
            f"GT does not cover required history slice: need gt[{gt_hist_t0}:{gt_hist_t1}] but gt_T={gt_T}. "
            f"(hist_start={hist_start}, gt_time0={gt_time0})"
        )
    if gt_fut_t0 < 0 or gt_fut_t1 > gt_T:
        raise ValueError(
            f"GT does not cover required future slice: need gt[{gt_fut_t0}:{gt_fut_t1}] but gt_T={gt_T}. "
            f"(fut_start={fut_start}, gt_time0={gt_time0})"
        )
    if gt_eval_t0 < 0 or gt_eval_t1 > gt_T:
        raise ValueError(
            f"GT does not cover required eval slice: need gt[{gt_eval_t0}:{gt_eval_t1}] but gt_T={gt_T}. "
            f"(eval_start={eval_start}, gt_time0={gt_time0})"
        )

    gt_hist_tracks = gt_tracks_full_px[gt_hist_t0:gt_hist_t1].astype(np.float32)  # [hist_len,N,2]
    gt_fut_tracks = gt_tracks_full_px[gt_fut_t0:gt_fut_t1].astype(np.float32)  # [future_len,N,2]
    gt_eval_tracks = gt_tracks_full_px[gt_eval_t0:gt_eval_t1].astype(np.float32)  # [pred_len,N,2]

    if gt_vis_full is not None:
        gt_hist_vis = gt_vis_full[gt_hist_t0:gt_hist_t1].astype(np.uint8)
        gt_fut_vis = gt_vis_full[gt_fut_t0:gt_fut_t1].astype(np.uint8)
        gt_eval_vis = gt_vis_full[gt_eval_t0:gt_eval_t1].astype(np.uint8)
    else:
        gt_hist_vis = inbounds_visibility(gt_hist_tracks, H=H, W=W)
        gt_fut_vis = inbounds_visibility(gt_fut_tracks, H=H, W=W)
        gt_eval_vis = inbounds_visibility(gt_eval_tracks, H=H, W=W)

    gt_hist_vis = _apply_oob_mask_to_vis(gt_hist_tracks, gt_hist_vis, W=W, H=H).astype(np.uint8)
    gt_fut_vis = _apply_oob_mask_to_vis(gt_fut_tracks, gt_fut_vis, W=W, H=H).astype(np.uint8)
    gt_eval_vis = _apply_oob_mask_to_vis(gt_eval_tracks, gt_eval_vis, W=W, H=H).astype(np.uint8)

    # Save GT slice aligned to saved predictions
    np.save(out_dir / "gt_tracks.npy", gt_eval_tracks.astype(np.float32))
    np.save(out_dir / "gt_visibility.npy", gt_eval_vis.astype(np.uint8))

    # -------------------------
    # Prepare conditioning inputs
    # -------------------------
    # Build dense history track field from sparse patch grid
    hist_tracks_nt2 = torch.from_numpy(np.transpose(gt_hist_tracks, (1, 0, 2))).to(device=device, dtype=dtype)
    hist_vis_nt = torch.from_numpy(np.transpose(gt_hist_vis, (1, 0)).astype(np.bool_)).to(device=device)
    dense_hist_px, _ = flat_to_dense_raw(
        tracks_nt2=hist_tracks_nt2,
        vis_nt=hist_vis_nt,
        raw_hw=(H, W),
        grid_stride=int(args.grid_stride),
    )
    dense_hist_px = dense_hist_px.unsqueeze(0)  # [1,hist_len,H,W,2]

    raw_w = int(getattr(cfg_gen.data, "raw_width", W))
    raw_h = int(getattr(cfg_gen.data, "raw_height", H))
    if raw_w != W or raw_h != H:
        print(
            f"[warn] video size HxW={H}x{W} != cfg_gen.data.raw_height/raw_width={raw_h}x{raw_w}. "
            "Using config values for normalize/denormalize."
        )

    dense_hist_n = normalize_points_torch(dense_hist_px, max_width=int(raw_w), max_height=int(raw_h))
    hist_x = rearrange(dense_hist_n, "b t h w c -> b t c h w").contiguous()  # [1,hist_len,2,H,W]

    # Encode history -> scaled latents
    q_hist = track_vae.encode(hist_x)
    z_hist_unscaled = q_hist.loc if hasattr(q_hist, "loc") else q_hist  # type: ignore
    z_hist = scale_latents(z_hist_unscaled)
    if isinstance(z_hist, torch.Tensor) and z_hist.dtype != dtype:
        z_hist = z_hist.to(dtype=dtype)

    # Video conditioning (optional)
    z_video = None
    if bool(getattr(cfg_gen.video_vae, "enable", False)) and video_vae is not None and (
    not bool(getattr(args, "no_video_cond", False))):
        hist_frames = frames[hist_start:hist_end]
        if len(hist_frames) != hist_len:
            raise ValueError(f"hist_frames len {len(hist_frames)} != hist_len {hist_len}")
        vid = torch.from_numpy(np.stack(hist_frames, axis=0)).to(device=device, dtype=torch.float32)  # [T,H,W,3]
        vid = vid / 127.5 - 1.0
        hist_video_cthw = vid.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # [1,3,T,H,W]

        # Choose input dtype that matches the WAN video VAE (some wrappers stay fp32).
        video_dtype = infer_module_dtype(video_vae) or torch.float32
        hist_video_cthw = hist_video_cthw.to(dtype=video_dtype)

        hist_video_small = resize_video_bcthw(hist_video_cthw, float(cfg_gen.video_vae.resize_factor))
        z_video = encode_video_latents_wan(video_vae, hist_video_small)

        # Cast to the generator/track latent dtype for downstream ops.
        if isinstance(z_video, torch.Tensor) and z_video.dtype != dtype:
            z_video = z_video.to(dtype=dtype)
        if z_video.shape[2] != z_hist.shape[2]:
            raise ValueError(
                f"WAN video tokens N={z_video.shape[2]} != track tokens N={z_hist.shape[2]}. "
                f"Check video_vae.resize_factor vs track tokens."
            )

    # Text conditioning (optional)
    text_cond = None
    if bool(getattr(cfg_gen.text_encoder, "enable", False)) and t5 is not None and (
    not bool(getattr(args, "no_text_cond", False))):
        caption = str(getattr(args, "caption", ""))
        text_cond = encode_text_t5_pooled(t5, [caption], device=device)
        if isinstance(text_cond, torch.Tensor) and text_cond.dtype != dtype:
            text_cond = text_cond.to(dtype=dtype)

    # -------------------------
    # Sample generator in latent space
    # -------------------------
    # NOTE: when running fp16/bf16, torchdiffeq calls the model without AMP by default.
    # Wrapping sampling in autocast prevents dtype mismatches (Float vs Half) inside the generator.
    autocast_enabled = (device.type == "cuda" and dtype in (torch.float16, torch.bfloat16))
    cast_ctx = torch.autocast(device_type="cuda", dtype=dtype) if autocast_enabled else contextlib.nullcontext()
    with cast_ctx:
        z_pred = sample_rectified_flow(
            model=generator,
            z_hist=z_hist,
            z_video=z_video,
            text_cond=text_cond,
            shape=tuple(z_hist.shape),
            noise_seed=int(args.noise_seed),
            method=str(args.solver),
            steps=int(args.steps),
            fix_first_frame=bool(args.fix_first_frame),
            cfg=cfg_gen,
        )  # [1,T_lat,N,C] scaled
    if isinstance(z_pred, torch.Tensor) and z_pred.dtype != dtype:
        z_pred = z_pred.to(dtype=dtype)

    z_use = z_pred

    # Decode predicted tracks to dense pixel space
    z_unscaled_dec = unscale_latents(z_use)
    # Keep dtype consistent with track VAE parameters (avoids Float/Half matmul errors)
    try:
        vae_dtype = next(track_vae.parameters()).dtype
    except StopIteration:
        vae_dtype = z_unscaled_dec.dtype
    if z_unscaled_dec.dtype != vae_dtype:
        z_unscaled_dec = z_unscaled_dec.to(dtype=vae_dtype)
    x_pred_n = track_vae.decode(z_unscaled_dec).clamp(-1.0, 1.0)  # [1,T_out,2,H,W]

    x_pred_thwc = rearrange(x_pred_n, "b t c h w -> b t h w c").contiguous()
    dense_pred_px = denormalize_points_torch(x_pred_thwc, max_width=int(raw_w),
                                             max_height=int(raw_h))  # [1,T_out,H,W,2]

    T_out = int(dense_pred_px.shape[1])
    if T_out < future_len:
        raise ValueError(f"track_vae.decode returned T_out={T_out} < future_len={future_len}.")

    # Visibility prediction (dense)
    try:
        vis_model_dtype = next(vis_predictor.parameters()).dtype
    except StopIteration:
        vis_model_dtype = z_use.dtype
    vis_logits_21 = vis_predictor(z_fut=z_use.to(dtype=vis_model_dtype))
    vis_prob_21 = torch.sigmoid(vis_logits_21)
    vis_prob_T_tok = repeat_21_to_81(vis_prob_21, stride=int(vis_temp_stride), out_len=int(future_len))  # [1,T,N]

    Hp = H // int(vis_patch_size)
    Wp = W // int(vis_patch_size)
    if Hp * Wp != int(vis_prob_T_tok.shape[2]):
        raise ValueError(
            f"vis token count mismatch: Hp*Wp={Hp * Wp} but vis_prob_T_tok.shape[2]={vis_prob_T_tok.shape[2]}. "
            f"Check vis_patch_size={vis_patch_size} vs raw H,W={H},{W}."
        )

    vis_prob_raw = vis_prob_T_tok.view(1, int(future_len), Hp, Wp)
    vis_prob_raw = vis_prob_raw.repeat_interleave(int(vis_patch_size), dim=2)
    vis_prob_raw = vis_prob_raw.repeat_interleave(int(vis_patch_size), dim=3)  # [1,future_len,H,W]

    pred_vis_raw_bool = (vis_prob_raw > float(vis_threshold)).to(torch.bool)

    # Sample dense preds/vis at query points
    flat_pred = dense_pred_px[0].reshape(T_out, -1, 2)
    flat_vis = pred_vis_raw_bool[0].reshape(int(future_len), -1)
    flat_vis_prob = vis_prob_raw[0].reshape(int(future_len), -1)

    pred_tracks_all = flat_pred[:future_len, grid_idx].detach().cpu().numpy().astype(np.float32)  # [future_len,N,2]
    pred_vis_all = flat_vis[:, grid_idx].detach().cpu().numpy().astype(np.uint8)  # [future_len,N]
    pred_vis_prob_all = flat_vis_prob[:, grid_idx].detach().cpu().numpy().astype(np.float32)  # [future_len,N]

    # Drop overlap frame if needed and slice to pred_len
    pred_tracks = pred_tracks_all[save_offset: save_offset + pred_len]
    pred_vis = pred_vis_all[save_offset: save_offset + pred_len]
    pred_vis_prob = pred_vis_prob_all[save_offset: save_offset + pred_len]

    # Sanitize pred vis with OOB coords (and clamp coords for downstream safety)
    pred_vis = _apply_oob_mask_to_vis(pred_tracks, pred_vis, W=W, H=H).astype(np.uint8)

    eps = 1e-3
    pred_tracks = pred_tracks.copy()
    pred_tracks[..., 0] = np.clip(pred_tracks[..., 0], 0.0, float(W - 1) - eps)
    pred_tracks[..., 1] = np.clip(pred_tracks[..., 1], 0.0, float(H - 1) - eps)

    # Save predictions
    np.save(out_dir / "pred_tracks.npy", pred_tracks.astype(np.float32))
    np.save(out_dir / "pred_visibility.npy", pred_vis.astype(np.uint8))
    if not bool(args.no_prob):
        np.save(out_dir / "pred_visibility_prob.npy", pred_vis_prob.astype(np.float16))

    meta: Dict[str, Any] = {
        "video_path": str(video_path),
        "video_len": int(T_vid),
        "H": int(H),
        "W": int(W),
        "cond_index": int(cond_index),
        "hist_len": int(hist_len),
        "future_len": int(future_len),
        "use_overlap": bool(use_overlap),
        "save_offset": int(save_offset),
        "pred_len": int(pred_len),
        "grid_stride": int(args.grid_stride),
        "grid_border": int(args.grid_border),
        "grid_origin": GRID_ORIGIN,
        "num_points": int(N),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "noise_seed": int(args.noise_seed),
        "solver": str(args.solver),
        "steps": int(args.steps),
        "fix_first_frame": bool(args.fix_first_frame),
        "vis_threshold": float(vis_threshold),
        "vis_patch_size": int(vis_patch_size),
        "vis_temp_stride": int(vis_temp_stride),
        "caption": str(getattr(args, "caption", "")),
        "gt_meta": {
            **gt_load_meta,
            **gt_coord_meta,
            "gt_time0_used": int(gt_time0),
            "gt_hist_slice": [int(gt_hist_t0), int(gt_hist_t1)],
            "gt_fut_slice": [int(gt_fut_t0), int(gt_fut_t1)],
            "gt_eval_slice": [int(gt_eval_t0), int(gt_eval_t1)],
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[done] {video_path.stem} | pred_len={pred_len} hist_len={hist_len} future_len={future_len} "
        f"| gt_coord_mode={gt_coord_meta.get('gt_coord_mode')} swap={gt_coord_meta.get('gt_swap_xy')} "
        f"gt_time0={gt_time0}"
    )
    return {"vid": video_path.stem, "out_dir": str(out_dir)}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run TrajLoom generator inference and export standard patch-grid outputs.",
    )

    # TrajLoom configs + checkpoints
    p.add_argument("--gen_config", "--config", dest="gen_config", type=str, required=True,
                   help="Path to generator config JSON.")
    p.add_argument(
        "--gen_ckpt", "--generator_ckpt",
        type=str,
        default=None,
        help="Path to generator checkpoint (.pt). If omitted, uses cfg.train.resume from gen_config.",
    )

    p.add_argument("--vis_config", type=str, required=True,
                   help="Path to visibility predictor config JSON.")
    p.add_argument("--vis_ckpt", type=str, required=True,
                   help="Path to visibility predictor checkpoint (.pt).")

    # input videos
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video_path", type=str, default="")
    g.add_argument("--video_dir", type=str, default="")
    p.add_argument("--video_glob", type=str, default="*.mp4")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--max_video_frames", type=int, default=-1)
    p.add_argument("--resize_hw", type=str, default="")

    # benchmark slice
    p.add_argument("--hist_len", type=int, default=-1)
    p.add_argument("--cond_index", type=int, default=81)
    p.add_argument(
        "--pred_len",
        type=int,
        default=-1,
        help="How many future frames to SAVE starting at cond_index. -1 uses cfg_gen.data.future_len (minus overlap).",
    )

    # query grid
    p.add_argument("--grid_stride", type=int, default=32)
    p.add_argument("--grid_border", type=int, default=0)

    # GT (required)
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
    p.add_argument("--gt_swap_xy", type=str, default="auto", choices=["auto", "0", "1"])

    # Output
    p.add_argument("--out_dir", type=str, required=True)

    # Sampling
    p.add_argument("--noise_seed", "--seed", dest="noise_seed", type=int, default=42)
    p.add_argument("--solver", "--sample_method", dest="solver", type=str, default="euler")
    p.add_argument("--steps", "--sample_steps", dest="steps", type=int, default=100)
    p.add_argument("--fix_first_frame", action="store_true")

    # Visibility threshold
    p.add_argument("--vis_threshold", type=float, default=0.5)

    # Text (optional global caption)
    p.add_argument("--caption", type=str, default="")
    p.add_argument("--no_video_cond", action="store_true")
    p.add_argument("--no_text_cond", action="store_true")

    p.add_argument("--no_prob", action="store_true")

    # Device / dtype
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--model_dtype", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--amp_dtype", type=str, default="", help=argparse.SUPPRESS)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if getattr(args, "model_dtype", ""):
        args.dtype = args.model_dtype
    dtype = dtype_from_arg(args.dtype)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"--gt_dir not found: {gt_dir}")

    # Load generator config
    cfg_gen = load_full_generator_config(args.gen_config)

    # Init latent scaler (required for SCALE/UNSCALE)
    init_latent_scaler(cfg_gen.train, device=device)

    # Load models
    track_vae, _ = load_trajloom_vae_from_checkpoint_cfg(cfg_gen.trajloom_vae, device=device)
    generator = TrajLoomGenerator(cfg_gen.generator).to(device).eval()

    gen_ckpt = Path(args.gen_ckpt) if args.gen_ckpt is not None else Path(str(getattr(cfg_gen.train, "resume", "")))
    if not gen_ckpt.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {gen_ckpt}")
    gen_sd = load_state_dict_from_ckpt(gen_ckpt, key_candidates=["generator", "model"])
    generator.load_state_dict(gen_sd, strict=True)

    # Optional: cast to dtype for memory/perf
    if dtype in (torch.float16, torch.bfloat16):
        track_vae = track_vae.to(dtype=dtype)
        generator = generator.to(dtype=dtype)

    # WAN video VAE + T5 (optional)
    video_vae = None
    if bool(getattr(cfg_gen.video_vae, "enable", False)) and (not bool(getattr(args, "no_video_cond", False))):
        # NOTE: In some WAN installs, load_wan_video_vae returns a wrapper (e.g., WanVAE)
        # that is *not* a torch.nn.Module and therefore has no .eval()/.to().
        # We handle both cases robustly here.
        video_vae = load_wan_video_vae(cfg_gen.video_vae, device=device)
        video_vae = maybe_eval(video_vae)
        if dtype in (torch.float16, torch.bfloat16):
            video_vae = maybe_to_device_dtype(video_vae, device=device, dtype=dtype)

    t5 = None
    if bool(getattr(cfg_gen.text_encoder, "enable", False)) and (not bool(getattr(args, "no_text_cond", False))):
        t5 = load_wan_t5(cfg_gen.text_encoder, device=device)
        t5 = maybe_eval(t5)
        if dtype in (torch.float16, torch.bfloat16):
            t5 = maybe_to_device_dtype(t5, device=device, dtype=dtype)

    # Visibility predictor
    vis_cfg, vis_patch_size, vis_temp_stride = load_vis_cfg_and_params(Path(args.vis_config))
    vis_model = TrajLoomVisibilityPredictor(vis_cfg).to(device).eval()
    vis_sd = load_state_dict_from_ckpt(Path(args.vis_ckpt), key_candidates=["model", "predictor"])
    vis_model.load_state_dict(vis_sd, strict=True)
    if dtype in (torch.float16, torch.bfloat16):
        vis_model = vis_model.to(dtype=dtype)

    # Videos
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
        req = [out_dir / "pred_tracks.npy", out_dir / "pred_visibility.npy"]
        if bool(args.skip_existing) and all(p.exists() for p in req):
            print(f"[skip] {vp.stem}")
            continue

        print(f"[run] {vp}")
        res = run_one_video(
            video_path=vp,
            out_root=out_root,
            cfg_gen=cfg_gen,
            track_vae=track_vae,
            generator=generator,
            vis_predictor=vis_model,
            vis_patch_size=int(vis_patch_size),
            vis_temp_stride=int(vis_temp_stride),
            vis_threshold=float(args.vis_threshold),
            video_vae=video_vae,
            t5=t5,
            device=device,
            dtype=dtype,
            args=args,
            gt_dir=gt_dir,
        )
        results.append(res)

    print(json.dumps({"num_done": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
