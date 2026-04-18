# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_ur10_follower_safe_xyzgripper_relative_train_cfg import (
    va_ur10_follower_safe_xyzgripper_relative_train_cfg,
)

va_ur10_follower_safe_xyzgripper_relative_max512_train_cfg = EasyDict(
    __name__="Config: VA UR10 follower safe relative xyz+gripper train max512"
)
va_ur10_follower_safe_xyzgripper_relative_max512_train_cfg.update(
    va_ur10_follower_safe_xyzgripper_relative_train_cfg
)
va_ur10_follower_safe_xyzgripper_relative_max512_train_cfg.max_bucket_key = int(
    os.environ.get("LINGBOT_VA_MAX_BUCKET_KEY", "512")
)
