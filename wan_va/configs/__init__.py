# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_i2va import va_libero_i2va_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_mt50_cfg import va_mt50_cfg
from .va_mt50_train_cfg import va_mt50_train_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_ur10_follower_safe_xyzgripper_cfg import (
    va_ur10_follower_safe_xyzgripper_cfg,
)
from .va_ur10_follower_safe_xyzgripper_chunk1_cfg import (
    va_ur10_follower_safe_xyzgripper_chunk1_cfg,
)
from .va_ur10_follower_safe_xyzgripper_relative_cfg import (
    va_ur10_follower_safe_xyzgripper_relative_cfg,
)
from .va_ur10_follower_safe_xyzgripper_relative_train_cfg import (
    va_ur10_follower_safe_xyzgripper_relative_train_cfg,
)
from .va_ur10_follower_safe_xyzgripper_relative_max512_train_cfg import (
    va_ur10_follower_safe_xyzgripper_relative_max512_train_cfg,
)

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'libero': va_libero_cfg,
    'mt50': va_mt50_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'libero_i2av': va_libero_i2va_cfg,
    'mt50_train': va_mt50_train_cfg,
    'libero_train': va_libero_train_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'ur10_follower_safe_xyzgripper': va_ur10_follower_safe_xyzgripper_cfg,
    'ur10_follower_safe_xyzgripper_chunk1': va_ur10_follower_safe_xyzgripper_chunk1_cfg,
    'ur10_follower_safe_xyzgripper_relative': va_ur10_follower_safe_xyzgripper_relative_cfg,
    'ur10_follower_safe_xyzgripper_relative_train': va_ur10_follower_safe_xyzgripper_relative_train_cfg,
    'ur10_follower_safe_xyzgripper_relative_max512_train': va_ur10_follower_safe_xyzgripper_relative_max512_train_cfg,
}
