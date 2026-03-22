# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './train_out'

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = True

# Action residual adapter defaults
va_shared_cfg.enable_action_residual_adapter = False
va_shared_cfg.action_adapter_dim = 256
va_shared_cfg.action_adapter_dropout = 0.0

# Trainable module switches
va_shared_cfg.freeze_backbone = False
va_shared_cfg.freeze_embeddings = False
va_shared_cfg.train_action_adapter = False
va_shared_cfg.train_action_head = False
va_shared_cfg.train_video_heads = False
va_shared_cfg.train_time_embedder = False
