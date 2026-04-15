# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robotwin_cfg = EasyDict(__name__='Config: VA robotwin')
va_robotwin_cfg.update(va_shared_cfg)

va_robotwin_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_VA_MODEL_PATH",
    "/home/syr/code/models/lingbot-va-posttrain-robotwin",
)

va_robotwin_cfg.attn_window = 72
va_robotwin_cfg.frame_chunk_size = 2
va_robotwin_cfg.env_type = 'robotwin_tshape'

va_robotwin_cfg.height = 256
va_robotwin_cfg.width = 320
va_robotwin_cfg.action_dim = 30
va_robotwin_cfg.action_per_frame = 16
va_robotwin_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_robotwin_cfg.guidance_scale = 5
va_robotwin_cfg.action_guidance_scale = 1

va_robotwin_cfg.num_inference_steps = 25
va_robotwin_cfg.video_exec_step = -1
va_robotwin_cfg.action_num_inference_steps = 50
va_robotwin_cfg.online_scheduler_mode = "fixed"
va_robotwin_cfg.fixed_video_steps = 25
va_robotwin_cfg.enable_hazard_scheduler_runtime = False
va_robotwin_cfg.hazard_checkpoint_path = ""
va_robotwin_cfg.hazard_hidden_dim = 512
va_robotwin_cfg.hazard_policy_variant = "stop_only_v1"
va_robotwin_cfg.hazard_K_max = 25
va_robotwin_cfg.hazard_K_min = 3
va_robotwin_cfg.hazard_eta = 0.8
va_robotwin_cfg.hazard_sigma_min = 0.01
va_robotwin_cfg.jump_logprob_scale = 1.0
va_robotwin_cfg.hazard_feature_source = "cond"
va_robotwin_cfg.hazard_return_metadata = True
va_robotwin_cfg.save_debug_artifacts = True

va_robotwin_cfg.snr_shift = 5.0
va_robotwin_cfg.action_snr_shift = 1.0

va_robotwin_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_robotwin_cfg.used_action_channel_ids)
] * va_robotwin_cfg.action_dim
for i, j in enumerate(va_robotwin_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robotwin_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robotwin_cfg.action_norm_method = 'quantiles'
va_robotwin_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}
