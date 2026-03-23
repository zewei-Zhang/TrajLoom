"""
A collection of functions and utilities for loading and configuring models,
parsing configurations, and managing state dictionaries within the TrajLoom framework.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from configs.generator_configs import (
    FullConfig,
    GeneratorDatasetConfig,
    TrajLoomVAECheckpointConfig,
    TrainConfig,
    TrajLoomGeneratorConfig,
    WanT5Config,
    WanVideoVAEConfig,
)
from models.trajloom_generator import TrajLoomGenerator
from models.trajloom_vae import TrajLoomVAE
from models.visibility_predictor import VisPredictorConfig


def parse_dtype(s: str) -> torch.dtype:
    s = (s or "").lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def canonicalize_trajloom_vae_config(d: Dict[str, Any]) -> Dict[str, Any]:
    if ("encoder_depth" in d) and ("decoder_depth" in d):
        allowed = {
            "input_height", "input_width", "patch_size",
            "in_channels", "latent_channels", "hidden_size",
            "encoder_depth", "decoder_depth", "num_heads", "mlp_ratio",
            "num_frames_in", "num_frames_latent",
            "learn_sigma", "pos_embed_dim",
            "beta", "use_temp_compress", "temp_stride", "use_offsets",
            "use_dino", "frame_features_dim", "num_frame_tokens", "temp_method", "use_refine3d",
        }
        return {k: v for k, v in d.items() if k in allowed}

    out: Dict[str, Any] = {}
    out["input_height"] = d.get("input_height", 480)
    out["input_width"] = d.get("input_width", 832)
    out["patch_size"] = d.get("patch_size", 32)
    out["in_channels"] = d.get("in_channels", d.get("input_channels", 2))
    out["latent_channels"] = d.get("latent_channels", 16)
    out["hidden_size"] = d.get("hidden_size", 384)

    n_layers = d.get("num_layers", 12)
    out["encoder_depth"] = d.get("encoder_depth", n_layers)
    out["decoder_depth"] = d.get("decoder_depth", n_layers)
    out["num_heads"] = d.get("num_heads", 6)
    out["mlp_ratio"] = d.get("mlp_ratio", 4.0)

    out["num_frames_in"] = d.get("num_frames_in", d.get("in_frames", 81))
    out["num_frames_latent"] = d.get("num_frames_latent", d.get("latent_frames", 21))

    out["use_offsets"] = bool(d.get("use_offsets", False))
    out["temp_stride"] = int(d.get("temp_stride", 4))
    out["use_temp_compress"] = bool(d.get("use_temp_compress", True))

    if "beta" in d:
        out["beta"] = d["beta"]
    elif "kl_weight" in d:
        out["beta"] = d["kl_weight"]

    for k in ("learn_sigma", "use_dino", "frame_features_dim",
              "num_frame_tokens", "pos_embed_dim", "temp_method", "use_refine3d"):
        if k in d:
            out[k] = d[k]
    return out


def _extract_state_dict(ckpt: Any, *, primary_key: str) -> Any:
    if isinstance(ckpt, dict) and primary_key in ckpt:
        return ckpt[primary_key]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def _freeze_eval(module: nn.Module, device: torch.device) -> nn.Module:
    module.eval().to(device)
    for p in module.parameters():
        p.requires_grad = False
    return module


def load_full_generator_config(path: Path) -> FullConfig:
    d = load_json(path)
    if hasattr(FullConfig, "from_dict"):
        return FullConfig.from_dict(d)

    trajloom_vae = d.get("trajloom_vae", d.get("track_vae"))
    if trajloom_vae is None:
        raise KeyError(f"Expected 'trajloom_vae' (preferred) or 'track_vae' in config: {path}")

    return FullConfig(
        data=GeneratorDatasetConfig(**d["data"]),
        trajloom_vae=TrajLoomVAECheckpointConfig(**trajloom_vae),
        video_vae=WanVideoVAEConfig(**d.get("video_vae", {})),
        text_encoder=WanT5Config(**d.get("text_encoder", {})),
        generator=TrajLoomGeneratorConfig(**d["generator"]),
        train=TrainConfig(**d.get("train", {})),
    )


def load_trajloom_vae_from_checkpoint_cfg(
    cfg: TrajLoomVAECheckpointConfig,
    device: torch.device,
) -> Tuple[TrajLoomVAE, Dict[str, Any]]:
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    raw_cfg = dict(cfg.config or {})
    if not raw_cfg:
        if isinstance(ckpt, dict) and isinstance(ckpt.get("cfg"), dict):
            raw_cfg = dict(ckpt["cfg"].get("vae", {}))
    if not raw_cfg:
        raise ValueError("Generator config must define trajloom_vae.config or provide cfg['vae'] in the checkpoint.")

    vae_cfg = canonicalize_trajloom_vae_config(raw_cfg or {})
    vae = TrajLoomVAE(**vae_cfg)
    state = _extract_state_dict(ckpt, primary_key="vae")
    vae.load_state_dict(state, strict=True)
    vae = _freeze_eval(vae, device)
    return vae, vae_cfg


def load_trajloom_vae_from_config_file(
    config_path: Path,
    ckpt_override: str,
    device: torch.device,
) -> Tuple[TrajLoomVAE, Dict[str, Any]]:
    payload = load_json(config_path)
    if "trajloom_vae" not in payload:
        raise KeyError(f"Expected 'trajloom_vae' in config: {config_path}")

    sec = payload["trajloom_vae"]
    ckpt_path = ckpt_override
    vae_cfg_raw: Dict[str, Any] = dict(sec.get("config", {}))
    ckpt_path = ckpt_path or str(sec.get("ckpt_path", ""))

    if not ckpt_path:
        raise ValueError(
            "Could not resolve VAE checkpoint path. Pass --vae_ckpt or set trajloom_vae.ckpt_path in the config."
        )
    if not vae_cfg_raw:
        raise ValueError(f"Expected trajloom_vae.config in config: {config_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    vae_cfg = canonicalize_trajloom_vae_config(vae_cfg_raw or {})
    vae = TrajLoomVAE(**vae_cfg)
    state = _extract_state_dict(ckpt, primary_key="vae")
    vae.load_state_dict(state, strict=True)
    vae = _freeze_eval(vae, device)
    return vae, vae_cfg


def load_generator_model(
    config: TrajLoomGeneratorConfig,
    ckpt_path: Path,
    device: torch.device,
    model_dtype: str,
) -> TrajLoomGenerator:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt, primary_key="generator")

    model = TrajLoomGenerator(config)
    model.load_state_dict(state, strict=True)
    if device.type == "cuda" and model_dtype.lower() in ("bf16", "fp16"):
        model = model.to(dtype=parse_dtype(model_dtype))
    model.eval().to(device)
    return model


def load_wan_video_vae(cfg: WanVideoVAEConfig, device: torch.device) -> Optional[nn.Module]:
    if not cfg.enable:
        return None
    if cfg.wan2_repo_dir and cfg.wan2_repo_dir not in sys.path:
        sys.path.insert(0, cfg.wan2_repo_dir)
    from wan.modules.vae import WanVAE  # type: ignore

    dtype = parse_dtype(getattr(cfg, "dtype", "fp32"))
    ckpt_path = cfg.ckpt_dir
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, cfg.ckpt_name)
    return WanVAE(vae_pth=ckpt_path, device=str(device), dtype=dtype)


def load_wan_t5(cfg: WanT5Config, device: torch.device) -> Optional[nn.Module]:
    if not cfg.enable:
        return None
    if cfg.wan2_repo_dir and cfg.wan2_repo_dir not in sys.path:
        sys.path.insert(0, cfg.wan2_repo_dir)
    from wan.modules.t5 import T5EncoderModel  # type: ignore

    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)
    return T5EncoderModel(
        text_len=cfg.max_length,
        checkpoint_path=ckpt_path,
        tokenizer_path=cfg.tokenizer_name,
        device=str(device),
    )


def load_vis_cfg_and_params(vis_config_path: Path) -> Tuple[VisPredictorConfig, int, int]:
    d = load_json(vis_config_path)

    if "predictor" in d:
        pred_cfg = VisPredictorConfig(**d["predictor"])
        patch_size = int(d.get("patch_size", 32))
        temp_stride = int(d.get("temp_stride", 4))
        return pred_cfg, patch_size, temp_stride

    pred_cfg = VisPredictorConfig(**d)
    patch_size = int(d.get("patch_size", 32))
    temp_stride = int(d.get("temp_stride", 4))
    return pred_cfg, patch_size, temp_stride


def load_state_dict_from_ckpt(ckpt_path: Path, key_candidates: list[str]) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict):
        for k in key_candidates:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")
