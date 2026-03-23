"""
Utilities for latent statistics and trajectory processing.

This module provides functions for handling latent statistics, scaling
and unscaling latent variables, translating trajectories, and converting
between flat and dense representations.

Functions:

1. load_latent_stats_json: Load latent statistics from a JSON file.
2. init_latent_scaler: Initialize global latent scaler variables.
3. scale_latents: Scale latent variables using global statistics.
4. unscale_latents: Unscale latent variables using global statistics.
5. grid_counts: Compute the number of grid points for a given model size and stride.
6. translate_trajectories: Adjust trajectories between source and target frame sizes.
7. flat_to_dense: Convert flat TAPIR/CoTracker tracks to a dense layout.
8. flat_to_dense_raw: Inverse conversion of flat tracks to dense raw frame grids.
9. detect_axis_order: Detect axis order (yx or xy) in trajectory coordinates.
"""
import json
from typing import Literal, Tuple, Union

import numpy as np
import torch
from einops import rearrange


_LATENT_STATS_MEAN = None
_LATENT_STATS_STD = None


def load_latent_stats_json(
    path: str,
    key: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(path, "r") as f:
        js = json.load(f)
    if key not in js:
        raise KeyError(
            f"latent stats json missing key '{key}'. available={list(js.keys())}"
        )
    pc = js[key]["per_channel"]
    mean = torch.tensor(pc["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(pc["std"], dtype=torch.float32, device=device)
    return mean, std


def init_latent_scaler(train_cfg, device: torch.device) -> None:
    global _LATENT_STATS_MEAN, _LATENT_STATS_STD
    if not getattr(train_cfg, "latent_stats_json", ""):
        _LATENT_STATS_MEAN, _LATENT_STATS_STD = None, None
        return
    mean_c, std_c = load_latent_stats_json(
        train_cfg.latent_stats_json,
        getattr(train_cfg, "latent_stats_key", "z_all"),
        device=device,
    )
    _LATENT_STATS_MEAN = mean_c.view(1, 1, 1, -1)
    _LATENT_STATS_STD = std_c.view(1, 1, 1, -1)


def scale_latents(z: torch.Tensor) -> torch.Tensor:
    if _LATENT_STATS_MEAN is None or _LATENT_STATS_STD is None:
        return z
    mean = _LATENT_STATS_MEAN.to(device=z.device, dtype=torch.float32)
    std = _LATENT_STATS_STD.to(device=z.device, dtype=torch.float32).clamp_min(1e-8)
    return ((z.float() - mean) / std).to(dtype=z.dtype)


def unscale_latents(z: torch.Tensor) -> torch.Tensor:
    if _LATENT_STATS_MEAN is None or _LATENT_STATS_STD is None:
        return z
    mean = _LATENT_STATS_MEAN.to(device=z.device, dtype=torch.float32)
    std = _LATENT_STATS_STD.to(device=z.device, dtype=torch.float32)
    return (z.float() * std + mean).to(dtype=z.dtype)


def grid_counts(model_size: Union[int, Tuple[int, int]], stride: int) -> Tuple[int, int]:
    """Compute the number of grid samples along H and W.

    Args:
        model_size: Either a scalar S (square SxS) or a tuple (H_model, W_model).
        stride:     Stride used when seeding the grid (e.g., 2, 4, 8).

    Returns:
        (H_count, W_count): number of grid points along height and width.
    """
    if isinstance(model_size, int):
        h_size = w_size = model_size
    else:
        h_size, w_size = model_size

    ys = np.arange(0, h_size, stride, dtype=np.int32)
    xs = np.arange(0, w_size, stride, dtype=np.int32)
    return int(len(ys)), int(len(xs))


def translate_trajectories(
    tracks: torch.Tensor,
    source_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """Scale pixel coordinates from source_hw to target_hw.

    Args:
        tracks:    Trajectories with trailing (..., 2) as (x, y).
        source_hw: (H_src, W_src) the current coordinate frame.
        target_hw: (H_tgt, W_tgt) the desired coordinate frame.

    Returns:
        Trajectories in target pixel coordinates.
    """
    h_src, w_src = source_hw
    h_tgt, w_tgt = target_hw
    scale = torch.tensor(
        [w_tgt / w_src, h_tgt / h_src],
        dtype=tracks.dtype,
        device=tracks.device,
    )
    return tracks * scale


def flat_to_dense(
    tracks_nt2: torch.Tensor,
    vis_nt: torch.Tensor,
    source_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    model_size: int,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert TAPIR/CoTracker flat tracks to dense layout for VAE/Latte.

    This is the original WHN / 256x256 grid path.

    Args:
        tracks_nt2: [N, T, 2] tracks in pixel coordinates.
        vis_nt:     [N, T] visibility (bool or 0/1).
        source_hw:  (H_src, W_src) current track frame.
        target_hw:  (H_tgt, W_tgt) frame size used for tracks.
        model_size: model grid size used to seed query grid.
        stride:     stride used to seed query grid.

    Returns:
        tracks_thwd: [T, H', W', 2] in target pixel coordinates.
        vis_thw:     [T, H', W'] visibility grid.
    """
    h_count, w_count = grid_counts(model_size, stride)
    tracks_nt2 = translate_trajectories(tracks_nt2, source_hw, target_hw)
    tracks_thwd = rearrange(
        tracks_nt2, "(hh ww) t d -> t hh ww d", hh=h_count, ww=w_count
    )
    vis_thw = rearrange(vis_nt, "(hh ww) t -> t hh ww", hh=h_count, ww=w_count)
    return tracks_thwd, vis_thw


def flat_to_dense_raw(
    tracks_nt2: torch.Tensor,
    vis_nt: torch.Tensor,
    raw_hw: Tuple[int, int],
    grid_stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of `extract_tracks_and_visibility` for the WAN case.

    We assume the flat tracks were produced by:

        traj_maps[:, :, :, ::s, ::s]  # stride = s
          .reshape(B, T, 2, -1)       # [B, T, 2, N]
          .permute(0, 1, 3, 2)        # [B, T, N, 2]
          # then -> [N, T, 2] for tracks_nt2

    and we want a dense track video on the *raw* grid HxW, e.g. 480x832.

    Args:
        tracks_nt2: [N, T, 2] pixel coordinates in the raw frame (x, y).
        vis_nt:     [N, T] visibility flags.
        raw_hw:     (H_raw, W_raw) of the video (e.g. (480, 832)).
        grid_stride: stride `s` used in extract_tracks_and_visibility.

    Returns:
        tracks_thwd: [T, H_raw, W_raw, 2] dense (seed locations filled).
        vis_thw:     [T, H_raw, W_raw] dense visibility mask.
    """
    H_raw, W_raw = raw_hw
    s = int(grid_stride)

    if H_raw % s != 0 or W_raw % s != 0:
        raise ValueError(
            f"raw_hw={raw_hw} must be divisible by grid_stride={s}"
        )

    Hc = H_raw // s
    Wc = W_raw // s
    N_expected = Hc * Wc

    N, T, D = tracks_nt2.shape
    if D != 2:
        raise ValueError(f"Expected last dim=2 (x,y), got {D}")
    if N != N_expected:
        raise ValueError(
            f"N={N} does not match coarse grid {Hc}x{Wc} "
            f"(from raw_hw={raw_hw}, stride={s}, expected {N_expected})"
        )

    device = tracks_nt2.device
    dtype = tracks_nt2.dtype

    # 1) Flat → coarse grid [T, Hc, Wc, 2]
    tracks_thwd_coarse = rearrange(
        tracks_nt2,
        "(hh ww) t d -> t hh ww d",
        hh=Hc,
        ww=Wc,
    )
    vis_thw_coarse = rearrange(
        vis_nt,
        "(hh ww) t -> t hh ww",
        hh=Hc,
        ww=Wc,
    )

    # 2) Coarse grid → dense raw grid by repeating each cell s x s
    tracks_thwd_dense = (
        tracks_thwd_coarse.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)
    )
    vis_thw_dense = (
        vis_thw_coarse.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)
    )

    # Sanity checks
    assert tracks_thwd_dense.shape[1] == H_raw and tracks_thwd_dense.shape[2] == W_raw
    assert vis_thw_dense.shape[1] == H_raw and vis_thw_dense.shape[2] == W_raw

    return tracks_thwd_dense, vis_thw_dense


@torch.no_grad()
def detect_axis_order(
    tracks_nt2: torch.Tensor,
    model_size: int,
    stride: int,
) -> Literal["yx", "xy"]:
    """Infer whether last dim is (y,x) or (x,y) by exploiting grid regularity.

    Args:
        tracks_nt2: [N, T, 2] tracks in pixel coordinates.
        model_size: Side length S used to seed the grid during TAPIR.
        stride:     Stride used to seed the grid during TAPIR.

    Returns:
        "yx" if the last dimension stores (y,x), else "xy".
    """
    n, _, _ = tracks_nt2.shape
    hh, ww = grid_counts(model_size, stride)
    if n != hh * ww:
        raise ValueError(
            f"N={n} does not match grid {hh}x{ww} (S={model_size}, stride={stride})"
        )

    pos0 = tracks_nt2[:, 0, :]
    pos2d = rearrange(pos0, "(hh ww) d -> hh ww d", hh=hh, ww=ww)

    row_std_y = pos2d[:, :, 0].std(dim=1).mean()
    col_std_x = pos2d[:, :, 1].std(dim=0).mean()
    score_yx = row_std_y + col_std_x

    row_std_x = pos2d[:, :, 1].std(dim=1).mean()
    col_std_y = pos2d[:, :, 0].std(dim=0).mean()
    score_xy = row_std_x + col_std_y

    return "yx" if score_yx <= score_xy else "xy"


def normalize_points(
    points: torch.Tensor,
    max_width: float,
    max_height: float,
) -> torch.Tensor:
    """Map raw pixel coords to [-1, 1] range using (max_width, max_height)."""
    scale = torch.tensor([max_width, max_height], device=points.device)
    return 2 * (points / scale) - 1


def denormalize_points(
    normalized_points: torch.Tensor,
    max_width: float,
    max_height: float,
) -> torch.Tensor:
    """Inverse of normalize_points: [-1,1] → pixel coords."""
    scale = torch.tensor(
        [max_width, max_height],
        device=normalized_points.device,
    )
    return (normalized_points + 1) * 0.5 * scale


_LAT_MEAN = None
_LAT_STD = None


def _load_latent_stats_json(path: str, key: str, device: torch.device):
    with open(path, "r") as f:
        js = json.load(f)
    if key not in js:
        raise KeyError(f"latent stats json missing key '{key}'. available={list(js.keys())}")
    pc = js[key]["per_channel"]
    mean = torch.tensor(pc["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(pc["std"], dtype=torch.float32, device=device)
    return mean, std


def init_latent_scaler(train_cfg, device: torch.device):
    global _LAT_MEAN, _LAT_STD
    if not getattr(train_cfg, "latent_stats_json", ""):
        _LAT_MEAN, _LAT_STD = None, None
        return
    mean_c, std_c = _load_latent_stats_json(train_cfg.latent_stats_json, getattr(train_cfg, "latent_stats_key", "z_all"), device=device)
    _LAT_MEAN = mean_c.view(1, 1, 1, -1)
    _LAT_STD = std_c.view(1, 1, 1, -1)


def scale_latents(z: torch.Tensor) -> torch.Tensor:
    if _LAT_MEAN is None or _LAT_STD is None:
        return z
    mean = _LAT_MEAN.to(device=z.device, dtype=torch.float32)
    std = _LAT_STD.to(device=z.device, dtype=torch.float32).clamp_min(1e-8)
    return ((z.float() - mean) / std).to(dtype=z.dtype)


def unscale_latents(z: torch.Tensor) -> torch.Tensor:
    if _LAT_MEAN is None or _LAT_STD is None:
        return z
    mean = _LAT_MEAN.to(device=z.device, dtype=torch.float32)
    std = _LAT_STD.to(device=z.device, dtype=torch.float32)
    return (z.float() * std + mean).to(dtype=z.dtype)
