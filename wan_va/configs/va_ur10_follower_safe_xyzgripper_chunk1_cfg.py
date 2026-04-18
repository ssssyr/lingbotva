# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

from copy import deepcopy

from .va_ur10_follower_safe_xyzgripper_cfg import (
    va_ur10_follower_safe_xyzgripper_cfg,
)


va_ur10_follower_safe_xyzgripper_chunk1_cfg = deepcopy(
    va_ur10_follower_safe_xyzgripper_cfg
)
va_ur10_follower_safe_xyzgripper_chunk1_cfg.__name__ = (
    "Config: VA UR10 follower safe xyz+gripper chunk1"
)

# Match the current chunk-level training mask with online rollout:
# one observed video frame slot predicts one action frame slot.
va_ur10_follower_safe_xyzgripper_chunk1_cfg.frame_chunk_size = 1
