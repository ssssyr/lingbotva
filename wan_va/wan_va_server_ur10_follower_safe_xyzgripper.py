# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

from . import wan_va_server as base_server
from .configs.va_ur10_follower_safe_xyzgripper_cfg import (
    va_ur10_follower_safe_xyzgripper_cfg,
)

base_server.VA_CONFIGS["ur10_follower_safe_xyzgripper"] = (
    va_ur10_follower_safe_xyzgripper_cfg
)


if __name__ == "__main__":
    base_server.init_logger()
    base_server.main()
