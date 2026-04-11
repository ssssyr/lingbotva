# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""Helpers for loading Hazard scheduler checkpoints in non-training runtimes."""

from __future__ import annotations

import torch

try:
    from ..modules.hazard_scheduler import HazardSchedulerHead
except ImportError:  # pragma: no cover - script-style imports in wan_va_server.py
    from modules.hazard_scheduler import HazardSchedulerHead


def get_model_hidden_dim(transformer) -> int:
    """Return the hidden width of the video backbone features."""
    config = transformer.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    return int(config.num_attention_heads) * int(config.attention_head_dim)


def get_scheduler_hidden_dim(config, default_dim: int) -> int:
    """Resolve the scheduler hidden width from config or fallback to backbone width."""
    model_cfg = getattr(config, "model", None)
    configured_dim = getattr(model_cfg, "hazard_hidden_dim", None)
    if configured_dim is None:
        configured_dim = getattr(config, "hazard_hidden_dim", None)
    if configured_dim is None:
        return int(default_dim)
    configured_dim = int(configured_dim)
    if configured_dim <= 0:
        raise ValueError(f"hazard_hidden_dim must be positive, got {configured_dim}")
    return configured_dim


def normalize_scheduler_state_dict(state_dict):
    """Strip a leading DDP ``module.`` prefix when present."""
    keys = list(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_hazard_scheduler_head(transformer, config, checkpoint_path, device, dtype):
    """Instantiate and load a Hazard scheduler head checkpoint."""
    checkpoint_path = str(checkpoint_path)
    video_feature_dim = get_model_hidden_dim(transformer)
    scheduler_hidden_dim = get_scheduler_hidden_dim(config, video_feature_dim)
    scheduler_head = HazardSchedulerHead(
        feature_dim=video_feature_dim,
        hidden_dim=scheduler_hidden_dim,
        output_dim=1,
        use_context=False,
    ).to(device=device, dtype=dtype)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    scheduler_head.load_state_dict(
        normalize_scheduler_state_dict(checkpoint["scheduler_head_state_dict"])
    )
    scheduler_head.eval()
    return scheduler_head, checkpoint
