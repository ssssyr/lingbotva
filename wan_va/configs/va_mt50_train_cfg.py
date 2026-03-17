# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_mt50_cfg import va_mt50_cfg

va_mt50_train_cfg = EasyDict(__name__='Config: VA MetaWorld MT50 train')
va_mt50_train_cfg.update(va_mt50_cfg)

resume_from = os.environ.get("LINGBOT_VA_RESUME_FROM")
if resume_from:
    va_mt50_train_cfg.resume_from = resume_from

va_mt50_train_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_VA_TRAIN_MODEL_PATH",
    va_mt50_train_cfg.wan22_pretrained_model_name_or_path,
)
va_mt50_train_cfg.dataset_path = os.environ.get(
    "LINGBOT_VA_DATASET_PATH",
    "/path/to/mt50_lerobot_dataset",
)
va_mt50_train_cfg.empty_emb_path = os.path.join(
    va_mt50_train_cfg.dataset_path,
    'empty_emb.pt',
)
va_mt50_train_cfg.enable_wandb = os.environ.get("LINGBOT_VA_ENABLE_WANDB", "0") == "1"
va_mt50_train_cfg.load_worker = int(os.environ.get("LINGBOT_VA_LOAD_WORKERS", "8"))
va_mt50_train_cfg.save_interval = int(os.environ.get("LINGBOT_VA_SAVE_INTERVAL", "500"))
va_mt50_train_cfg.gc_interval = int(os.environ.get("LINGBOT_VA_GC_INTERVAL", "50"))
va_mt50_train_cfg.cfg_prob = float(os.environ.get("LINGBOT_VA_CFG_PROB", "0.1"))

va_mt50_train_cfg.learning_rate = float(os.environ.get("LINGBOT_VA_LR", "1e-5"))
va_mt50_train_cfg.beta1 = 0.9
va_mt50_train_cfg.beta2 = 0.95
va_mt50_train_cfg.weight_decay = float(os.environ.get("LINGBOT_VA_WEIGHT_DECAY", "0.1"))
va_mt50_train_cfg.warmup_steps = int(os.environ.get("LINGBOT_VA_WARMUP_STEPS", "10"))
va_mt50_train_cfg.batch_size = int(os.environ.get("LINGBOT_VA_BATCH_SIZE", "1"))
va_mt50_train_cfg.gradient_accumulation_steps = int(
    os.environ.get("LINGBOT_VA_GRAD_ACCUM_STEPS", "4")
)
va_mt50_train_cfg.num_steps = int(os.environ.get("LINGBOT_VA_NUM_STEPS", "2000"))
