"""
This module provides utilities for video processing and reproducibility in machine learning workflows.

The module includes functionality to set random seeds for deterministic behavior, save batches of
video tensors as files, and handle potential dependencies on external packages. Additionally, it
offers a utility method to extract canonical video IDs from file paths.
"""
import csv
import os
import glob
import random
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
try:
    from wan.utils.utils import cache_video  # real function
    HAS_WAN = True
except Exception:
    HAS_WAN = False

    def cache_video(*args, **kwargs):
        """Fallback no-op when `wan` isn't installed."""
        # You can also `warnings.warn(...)` here if you want a notice.
        return None


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def save_batch_videos_with_ids(videos, vids, out_dir="./generated", fps=16):
    """
    videos: list[Tensor [3,T,H,W]] in [-1,1] returned by generate()
    vids:   list[str] ids to name files
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for v, vid in zip(videos, vids):
        # cache_video expects [N,C,T,H,W]; use N=1 and nrow=1 to avoid tiling
        ten = v.unsqueeze(0)  # [1,3,T,H,W]
        out_path = os.path.join(out_dir, f"{vid}.mp4")
        cache_video(
            tensor=ten, save_file=out_path, fps=fps,
            nrow=1, normalize=True, value_range=(-1, 1)
        )
        paths.append(out_path)
    return paths


def _canonical_vid(v):
    """Drop folders & extensions so '.../6764402.mp4' -> '6764402'."""
    base = os.path.basename(str(v))
    stem, _ = os.path.splitext(base)
    return stem
