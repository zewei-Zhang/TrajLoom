"""
Module providing utilities for model manipulation and optimization.

This module contains several functions to facilitate working with PyTorch models,
including handling distributed data parallel (DDP) wrapping, activation checkpointing,
and mixed precision casting.
"""
from torch import nn
import torch
from functools import partial


def unwrap_ddp(m: nn.Module) -> nn.Module:
    """Return underlying module if wrapped by DDP (so we can call custom methods)."""
    return m.module if hasattr(m, "module") else m


def _is_backbone_block(mod: nn.Module) -> bool:
    name = mod.__class__.__name__.lower()
    return (
            ("block" in name or "transformer" in name)
            and hasattr(mod, "forward")
            and not isinstance(mod, nn.Sequential)
    )


def enable_activation_ckpt(root: nn.Module):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )

    wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(root, checkpoint_wrapper_fn=wrapper, check_fn=_is_backbone_block)


def cast_model_bf16_keep_norm_fp32(model: nn.Module):
    model.to(dtype=torch.bfloat16)
    for m in model.modules():
        if isinstance(
                m,
                (
                        nn.LayerNorm,
                        nn.GroupNorm,
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.RMSNorm,
                ),
        ):
            m.float()
