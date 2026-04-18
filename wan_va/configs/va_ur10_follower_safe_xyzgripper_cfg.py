# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg

va_ur10_follower_safe_xyzgripper_cfg = EasyDict(
    __name__="Config: VA UR10 follower safe xyz+gripper"
)
va_ur10_follower_safe_xyzgripper_cfg.update(va_shared_cfg)

va_ur10_follower_safe_xyzgripper_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_VA_MODEL_PATH",
    "/path/to/lingbot-va-base",
)

va_ur10_follower_safe_xyzgripper_cfg.attn_window = int(
    os.environ.get("LINGBOT_VA_ATTN_WINDOW", "30")
)
va_ur10_follower_safe_xyzgripper_cfg.frame_chunk_size = int(
    os.environ.get("LINGBOT_VA_FRAME_CHUNK_SIZE", "4")
)
va_ur10_follower_safe_xyzgripper_cfg.env_type = "none"

va_ur10_follower_safe_xyzgripper_cfg.height = int(
    os.environ.get("LINGBOT_VA_FRAME_HEIGHT", "256")
)
va_ur10_follower_safe_xyzgripper_cfg.width = int(
    os.environ.get("LINGBOT_VA_FRAME_WIDTH", "320")
)
va_ur10_follower_safe_xyzgripper_cfg.action_dim = 30
va_ur10_follower_safe_xyzgripper_cfg.action_per_frame = int(
    os.environ.get("LINGBOT_VA_ACTION_PER_FRAME", "4")
)
va_ur10_follower_safe_xyzgripper_cfg.action_representation = "absolute"
va_ur10_follower_safe_xyzgripper_cfg.relative_action_base = ""
va_ur10_follower_safe_xyzgripper_cfg.obs_cam_keys = [
    item.strip()
    for item in os.environ.get(
        "LINGBOT_VA_UR10_CAMERA_KEYS",
        "observation.images.third,observation.images.wrist",
    ).split(",")
    if item.strip()
]
va_ur10_follower_safe_xyzgripper_cfg.guidance_scale = 5
va_ur10_follower_safe_xyzgripper_cfg.action_guidance_scale = 1

va_ur10_follower_safe_xyzgripper_cfg.num_inference_steps = 5
va_ur10_follower_safe_xyzgripper_cfg.video_exec_step = -1
va_ur10_follower_safe_xyzgripper_cfg.action_num_inference_steps = 10

va_ur10_follower_safe_xyzgripper_cfg.snr_shift = 5.0
va_ur10_follower_safe_xyzgripper_cfg.action_snr_shift = 1.0

# Only learn x/y/z and gripper. Orientation is fixed by the runtime bridge.
va_ur10_follower_safe_xyzgripper_cfg.used_action_channel_ids = [0, 1, 2, 28]
inverse_used_action_channel_ids = [
    len(va_ur10_follower_safe_xyzgripper_cfg.used_action_channel_ids)
] * va_ur10_follower_safe_xyzgripper_cfg.action_dim
for i, j in enumerate(va_ur10_follower_safe_xyzgripper_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_ur10_follower_safe_xyzgripper_cfg.inverse_used_action_channel_ids = (
    inverse_used_action_channel_ids
)

va_ur10_follower_safe_xyzgripper_cfg.action_norm_method = "quantiles"
va_ur10_follower_safe_xyzgripper_cfg.norm_stat = {
    "q01": [
        0.6175350761827103,
        -0.15494709144558863,
        0.34905938286342525,
    ]
    + [0.0] * 25
    + [-1.000000013351432e-10, 0.0],
    "q99": [
        0.8777034230434345,
        0.08205861096422079,
        0.6177889849397484,
    ]
    + [0.0] * 25
    + [0.9999970714593885, 0.0],
}
