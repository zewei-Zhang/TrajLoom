"""Utility helpers for loading trajectories, videos, and model inputs."""

from __future__ import annotations

import csv
import os
import re
from typing import Any, Optional

import cv2
import decord
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from utils.utils import _canonical_vid

GRID_ORIGIN = "topleft"
_PROMPT_COLUMN_CANDIDATES = (
    "prompt",
    "caption",
    "cap",
    "text",
    "description",
    "Prompt",
    "Caption",
)


# ------------------------- Trajectory / prompt loading -------------------------


def load_trajectory(trajectory_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load saved trajectory and visibility arrays from a NumPy archive."""
    data = np.load(trajectory_path)
    try:
        trajectory_key = next(
            (key for key in ("trajectory", "trajectories", "tracks") if key in data),
            None,
        )
        visibility_key = next(
            (key for key in ("visibility", "visibilities", "vis") if key in data),
            None,
        )
        if trajectory_key is None or visibility_key is None:
            available = sorted(str(key) for key in data.keys())
            raise KeyError(
                f"{trajectory_path} must contain trajectory/visibility arrays; found keys={available}"
            )

        trajectories = torch.from_numpy(data[trajectory_key])
        visibles = torch.from_numpy(data[visibility_key])
    finally:
        close_fn = getattr(data, "close", None)
        if callable(close_fn):
            close_fn()

    return trajectories, visibles


def _resolve_prompt_column(headers: list[str], prompt_col: str | None) -> str:
    if prompt_col is not None:
        return prompt_col

    found = [candidate for candidate in _PROMPT_COLUMN_CANDIDATES if candidate in headers]
    if not found:
        raise ValueError(
            f"prompt_col not given and none of {_PROMPT_COLUMN_CANDIDATES} in {headers}"
        )
    return found[0]


def _build_prompt_map(rows: Any, *, id_col: str, prompt_col: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in rows:
        vid = _canonical_vid(row[id_col])
        mapping[vid] = str(row[prompt_col]).strip()
    return mapping


def load_prompt_map(
    table_path: str,
    id_col: str = "vid",
    prompt_col: str | None = None,
    encoding: str = "utf-8",
) -> dict[str, str]:
    """Return ``{video_id: prompt}`` from a CSV or Excel file."""
    ext = os.path.splitext(table_path)[1].lower()

    if ext == ".csv":
        with open(table_path, newline="", encoding=encoding) as f:
            reader = csv.DictReader(f)
            headers = [str(h).strip() for h in reader.fieldnames or []]
            reader.fieldnames = headers
            if id_col not in headers:
                raise ValueError(f"id_col '{id_col}' not found; available: {headers}")

            resolved_prompt_col = _resolve_prompt_column(headers, prompt_col)
            return _build_prompt_map(reader, id_col=id_col, prompt_col=resolved_prompt_col)

    if ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError(
                "Reading Excel requires pandas (and openpyxl for .xlsx). "
                "Install: pip install pandas openpyxl"
            ) from e

        df = pd.read_excel(table_path).rename(columns=lambda col: str(col).strip())
        headers = list(df.columns)
        if id_col not in headers:
            raise ValueError(f"id_col '{id_col}' not found; available: {headers}")

        resolved_prompt_col = _resolve_prompt_column(headers, prompt_col)
        return _build_prompt_map(
            (row for _, row in df.iterrows()),
            id_col=id_col,
            prompt_col=resolved_prompt_col,
        )

    raise ValueError(f"Unsupported table extension: {ext}")


# ------------------------- Video helpers -------------------------


def read_video_frames(
    video_path: str,
    use_type: str = "cv2",
    is_rgb: bool = True,
    info: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], float, int, int, int]:
    """Read all frames from a video with ``cv2`` or ``decord``."""
    frames: list[np.ndarray] = []

    if use_type == "decord":
        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(video_path)
        total_frames = len(reader)
        fps = float(reader.get_avg_fps())
        height, width, _ = reader[0].shape
        frames = [reader[i].asnumpy() for i in range(total_frames)]
    elif use_type == "cv2":
        capture = cv2.VideoCapture(video_path)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            frames.append(frame[..., ::-1] if is_rgb else frame)
        capture.release()
        total_frames = len(frames)
    else:
        raise ValueError(f"Unknown video type {use_type}")

    if info:
        return frames, fps, int(width), int(height), int(total_frames)
    return frames


def _natural_key(path: str) -> list[int | str]:
    """Sort helper that keeps numeric filename parts in numeric order."""
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.findall(r"\d+|\D+", os.path.basename(path))
    ]


def _to_bcthw(frames_list: list[np.ndarray]) -> torch.Tensor:
    """Convert ``[T,H,W,3]`` uint8 frames into ``[3,T,H,W]`` float32 in ``[-1, 1]``."""
    if not frames_list:
        raise ValueError("frames_list must be non-empty")

    arr = np.stack(frames_list, axis=0)
    x = torch.from_numpy(arr).float() / 255.0
    x = x.permute(3, 0, 1, 2).contiguous()
    return x * 2.0 - 1.0


def resize_video_bcthw(hist_video_bcthw: torch.Tensor, resize_factor: int) -> torch.Tensor:
    """Resize a ``[B,C,T,H,W]`` video tensor spatially by an integer factor."""
    factor = int(resize_factor)
    if factor <= 1:
        return hist_video_bcthw

    batch, channels, frames, height, width = hist_video_bcthw.shape
    if (height % factor) != 0 or (width % factor) != 0:
        raise ValueError(
            f"video H,W must be divisible by resize_factor={factor}, "
            f"got H={height}, W={width}"
        )

    out_h, out_w = height // factor, width // factor
    x = hist_video_bcthw.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return x.reshape(batch, frames, channels, out_h, out_w).permute(0, 2, 1, 3, 4).contiguous()


# ------------------------- Encoding helpers -------------------------


@torch.no_grad()
def encode_video_latents_wan(video_vae: Any, hist_video_bcthw: torch.Tensor) -> torch.Tensor:
    """Encode a batch of videos and return ``[B,T,N,C]`` WAN latents."""
    batch = hist_video_bcthw.shape[0]
    videos = [hist_video_bcthw[i] for i in range(batch)]
    latents = video_vae.encode(videos)
    first_latent = latents[0]

    if first_latent.dim() == 4:
        z = torch.stack(latents, dim=0)
    elif first_latent.dim() == 5:
        z = torch.cat(latents, dim=0)
    else:
        raise ValueError(f"Unexpected WanVAE.encode latent shape: {first_latent.shape}")

    return rearrange(z, "b c t h w -> b t (h w) c")


@torch.no_grad()
def encode_text_t5_pooled(t5: Any, captions: list[str], device: torch.device) -> torch.Tensor:
    """Encode captions and mean-pool the token embeddings."""
    ctx_list = t5(captions, device=device)
    pooled = [ctx.mean(dim=0) for ctx in ctx_list]
    return torch.stack(pooled, dim=0)


@torch.no_grad()
def repeat_21_to_81(
    logits_btn: torch.Tensor,
    *,
    stride: int = 4,
    out_len: int = 81,
) -> torch.Tensor:
    """Expand ``[B,21,N]`` logits to a dense temporal grid."""
    if logits_btn.dim() != 3:
        raise ValueError(f"expected [B,21,N], got {tuple(logits_btn.shape)}")

    x = logits_btn.repeat_interleave(int(stride), dim=1)
    if x.size(1) > out_len:
        return x[:, :out_len]
    if x.size(1) < out_len:
        return F.pad(x, (0, 0, 0, out_len - x.size(1)))
    return x


# ------------------------- Device / dtype -------------------------


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string with an explicit CUDA availability check."""
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device_str} but CUDA is not available.")
        return torch.device(device_str)
    return torch.device("cpu")


def dtype_from_arg(s: str) -> torch.dtype:
    """Map a short dtype string to a torch dtype."""
    s = str(s).lower()
    if s == "fp32":
        return torch.float32
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    raise ValueError(s)


# ------------------------- WAN wrapper helpers -------------------------


def find_torch_module(obj: Any) -> Optional[torch.nn.Module]:
    """Try to find an ``nn.Module`` inside common WAN wrappers."""
    if obj is None:
        return None
    if isinstance(obj, torch.nn.Module):
        return obj

    for attr in ("model", "module", "vae", "net", "encoder", "decoder"):
        inner = getattr(obj, attr, None)
        if isinstance(inner, torch.nn.Module):
            return inner
    return None


def infer_module_dtype(obj: Any) -> Optional[torch.dtype]:
    """Best-effort: return the dtype of the first parameter or buffer."""
    module = find_torch_module(obj)
    if module is None:
        return None

    for param in module.parameters(recurse=True):
        return param.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.dtype
    return None


def maybe_eval(obj: Any) -> Any:
    """Call ``.eval()`` when available, including through common wrappers."""
    if obj is None:
        return obj

    if hasattr(obj, "eval") and callable(getattr(obj, "eval")):
        try:
            obj.eval()  # type: ignore[call-arg]
            return obj
        except Exception:
            pass

    module = find_torch_module(obj)
    if module is not None:
        module.eval()
    return obj


def maybe_to_device_dtype(obj: Any, *, device: torch.device, dtype: torch.dtype) -> Any:
    """Best-effort move/cast helper for wrappers that may not be ``nn.Module``."""
    if obj is None:
        return obj

    if hasattr(obj, "to") and callable(getattr(obj, "to")):
        try:
            obj.to(device=device, dtype=dtype)  # type: ignore[misc]
            return obj
        except Exception:
            try:
                obj.to(device=device)  # type: ignore[misc]
            except Exception:
                pass
            try:
                obj.to(dtype=dtype)  # type: ignore[misc]
            except Exception:
                pass

    module = find_torch_module(obj)
    if module is not None:
        module.to(device=device, dtype=dtype)

    return obj
