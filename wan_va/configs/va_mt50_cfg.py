# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg

va_mt50_cfg = EasyDict(__name__='Config: VA MetaWorld MT50')
va_mt50_cfg.update(va_shared_cfg)

va_mt50_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_VA_MODEL_PATH",
    "/path/to/lingbot-va-base",
)

va_mt50_cfg.attn_window = 30
va_mt50_cfg.frame_chunk_size = int(os.environ.get("LINGBOT_VA_FRAME_CHUNK_SIZE", "4"))
va_mt50_cfg.env_type = 'none'

va_mt50_cfg.height = int(os.environ.get("LINGBOT_VA_FRAME_HEIGHT", "256"))
va_mt50_cfg.width = int(os.environ.get("LINGBOT_VA_FRAME_WIDTH", "256"))
va_mt50_cfg.action_dim = 30
va_mt50_cfg.action_per_frame = int(os.environ.get("LINGBOT_VA_ACTION_PER_FRAME", "1"))
va_mt50_cfg.obs_cam_keys = [
    item.strip()
    for item in os.environ.get(
        "LINGBOT_VA_MT50_CAMERA_KEYS",
        "observation.images.main,observation.images.wrist",
    ).split(",")
    if item.strip()
]
va_mt50_cfg.guidance_scale = 5
va_mt50_cfg.action_guidance_scale = 1

va_mt50_cfg.num_inference_steps = 5
va_mt50_cfg.video_exec_step = -1
va_mt50_cfg.action_num_inference_steps = 10

va_mt50_cfg.snr_shift = 5.0
va_mt50_cfg.action_snr_shift = 1.0

# MT50 actions are typically 4-D: dx, dy, dz, gripper.
# The first three channels map to the single-arm Cartesian slots and the
# gripper maps to LingBot's canonical gripper channel 28.
va_mt50_cfg.used_action_channel_ids = [0, 1, 2, 28]
inverse_used_action_channel_ids = [
    len(va_mt50_cfg.used_action_channel_ids)
] * va_mt50_cfg.action_dim
for i, j in enumerate(va_mt50_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_mt50_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_mt50_cfg.action_norm_method = 'quantiles'
va_mt50_cfg.norm_stat = {
    "q01": [-1.0, -1.0, -1.0] + [0.0] * 25 + [-1.0, 0.0],
    "q99": [1.0, 1.0, 1.0] + [0.0] * 25 + [1.0, 0.0],
}
