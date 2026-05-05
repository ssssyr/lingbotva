# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from copy import deepcopy

from .va_ur10_follower_safe_xyzgripper_cfg import (
    va_ur10_follower_safe_xyzgripper_cfg,
)


va_ur10_follower_safe_xyzgripper_relative_cfg = deepcopy(
    va_ur10_follower_safe_xyzgripper_cfg
)
va_ur10_follower_safe_xyzgripper_relative_cfg.__name__ = (
    "Config: VA UR10 follower safe relative xyz+gripper"
)
va_ur10_follower_safe_xyzgripper_relative_cfg.action_representation = (
    "relative_chunk_anchor"
)
va_ur10_follower_safe_xyzgripper_relative_cfg.relative_action_base = (
    "chunk_anchor"
)

# Computed from /home/syr/code/outputs/ur10_gamepad_tcp_20hz_drawer_block_trainready_20260412
# by subtracting each segment's first tcp.xyz before aggregating quantiles.
va_ur10_follower_safe_xyzgripper_relative_cfg.norm_stat = {
    "q01": [
        -0.27176573872566223,
        -0.20350971817970276,
        -0.22701884806156158,
    ]
    + [0.0] * 25
    + [-1.0, 0.0],
    "q99": [
        0.23846584558486938,
        0.1785377413034439,
        0.14233073592185974,
    ]
    + [0.0] * 25
    + [1.0, 0.0],
}
