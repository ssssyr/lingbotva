#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""
Training entry point for Hazard scheduler.

This script sets up the training environment and launches the REINFORCE
training loop for learning adaptive video denoising schedules.
"""

import argparse
import os
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wan_va.modules.model import WanTransformer3DModel
from wan_va.modules.hazard_scheduler import HazardSchedulerHead
from wan_va.modules.utils import load_transformer
from wan_va.configs import VA_CONFIGS
from wan_va.utils.hazard_trainer import HazardTrainer
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from wan_va.utils import init_logger, logger
from wan_va.utils.hazard_monitor import HazardRunMonitor
from torch.utils.data import DataLoader, DistributedSampler, Subset


class ConfigNamespace(dict):
    def __init__(self, d):
        super().__init__()
        for key, value in d.items():
            if isinstance(value, dict):
                value = ConfigNamespace(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value
        super().__setattr__(key, value)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hazard scheduler")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    return parser.parse_args()


def _to_plain_python(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, torch.dtype):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def load_config(config_path):
    """Load YAML config and merge it into a base LingBot config."""
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    launcher = raw_config.get("launcher", {})
    profile_name = launcher.get("config_name", "robotwin_train")
    base_name = profile_name
    if base_name not in VA_CONFIGS and base_name.endswith("_hazard_train"):
        candidate = base_name.replace("_hazard_train", "_train")
        if candidate in VA_CONFIGS:
            base_name = candidate
    if base_name not in VA_CONFIGS:
        raise KeyError(
            f"Unknown base config {profile_name!r}. "
            f"Available base configs: {sorted(VA_CONFIGS.keys())}"
        )

    merged = dict(deepcopy(VA_CONFIGS[base_name]))
    merged["launcher"] = raw_config.get("launcher", {})
    merged["paths"] = raw_config.get("paths", {})
    merged["logging"] = raw_config.get("logging", {})
    merged["model"] = raw_config.get("model", {})
    merged["trainable"] = raw_config.get("trainable", {})
    merged["training"] = raw_config.get("training", {})
    merged["hazard"] = raw_config.get("hazard", {})
    for key, value in raw_config.items():
        if key not in {"launcher", "paths", "logging", "model", "trainable", "training", "hazard"}:
            merged[key] = value

    paths = merged["paths"]
    if "train_model_path" in paths:
        merged["wan22_pretrained_model_name_or_path"] = paths["train_model_path"]
    if "dataset_path" in paths:
        merged["dataset_path"] = paths["dataset_path"]
        merged["empty_emb_path"] = os.path.join(paths["dataset_path"], "empty_emb.pt")
    if "save_root" in paths:
        merged["save_root"] = paths["save_root"]

    logging_cfg = merged["logging"]
    if "enable_wandb" in logging_cfg:
        merged["enable_wandb"] = logging_cfg["enable_wandb"]

    model_cfg = merged["model"]
    for key in [
        "enable_action_residual_adapter",
        "action_adapter_dim",
        "action_adapter_dropout",
    ]:
        if key in model_cfg:
            merged[key] = model_cfg[key]

    trainable_cfg = merged["trainable"]
    for key in [
        "freeze_backbone",
        "freeze_embeddings",
        "train_action_adapter",
        "train_action_head",
        "train_video_heads",
        "train_time_embedder",
    ]:
        if key in trainable_cfg:
            merged[key] = trainable_cfg[key]

    training_cfg = merged["training"]
    compatibility_map = {
        "load_workers": "load_worker",
        "init_workers": "init_worker",
        "save_interval": "save_interval",
        "gc_interval": "gc_interval",
        "cfg_prob": "cfg_prob",
        "batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "num_steps": "num_steps",
    }
    for src_key, dst_key in compatibility_map.items():
        if src_key in training_cfg:
            merged[dst_key] = training_cfg[src_key]

    return ConfigNamespace(merged)


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def resolve_transformer_path(model_root: str) -> str:
    """Accept either a model root or an explicit transformer directory."""
    model_root = str(model_root)
    transformer_dir = os.path.join(model_root, "transformer")
    if os.path.isdir(transformer_dir):
        return transformer_dir
    return model_root


def get_model_hidden_dim(transformer: WanTransformer3DModel) -> int:
    """Return the backbone hidden size used by scheduler features."""
    config = transformer.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    return int(config.num_attention_heads) * int(config.attention_head_dim)


def get_scheduler_hidden_dim(config, default_dim: int) -> int:
    """Return the configured scheduler hidden width."""
    model_cfg = getattr(config, "model", None)
    configured_dim = getattr(model_cfg, "hazard_hidden_dim", None)
    if configured_dim is None:
        return int(default_dim)
    configured_dim = int(configured_dim)
    if configured_dim <= 0:
        raise ValueError(f"hazard_hidden_dim must be positive, got {configured_dim}")
    return configured_dim


def resolve_run_name(config, rank: int) -> str:
    logging_cfg = getattr(config, "logging", None)
    env_name = os.environ.get("LINGBOT_HAZARD_RUN_NAME")
    configured_name = getattr(logging_cfg, "run_name", None)
    configured_prefix = getattr(logging_cfg, "run_name_prefix", None)
    if env_name:
        return str(env_name)
    return str(
        configured_name
        or f"{configured_prefix or getattr(getattr(config, 'launcher', None), 'config_name', 'hazard')}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def prepare_run_directory(config, rank: int, world_size: int) -> Path:
    base_save_root = Path(config.paths.save_root)
    run_name = resolve_run_name(config, rank)
    run_dir = base_save_root / "runs" / run_name

    if rank == 0:
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        base_save_root.mkdir(parents=True, exist_ok=True)
        (base_save_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
        latest_link = base_save_root / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(run_dir, target_is_directory=True)
        except OSError:
            pass

    config.run_name = run_name
    config.run_dir = str(run_dir)
    config.paths.base_save_root = str(base_save_root)
    config.paths.save_root = str(run_dir)
    return run_dir


def split_dataset_for_train(dataset, val_ratio: float = 0.1):
    """Simple holdout split so training does not consume the full dataset."""
    total = len(dataset)
    if total <= 1:
        return dataset

    val_size = max(1, int(total * val_ratio))
    train_size = total - val_size
    if train_size <= 0:
        train_size = total
    return Subset(dataset, range(train_size))


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    config.rank = rank
    config.world_size = world_size
    config.local_rank = local_rank

    run_dir = prepare_run_directory(config, rank=rank, world_size=world_size)
    init_logger(
        log_file=run_dir / "logs" / f"train_rank{rank}.log",
        rank=rank,
        console=(rank == 0),
        force=True,
    )

    # Setup device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    logger.info(f"Rank {rank}/{world_size} on device {device}")
    logger.info(f"Run name: {config.run_name}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Profile: {args.config}")

    monitor = HazardRunMonitor(
        run_dir=run_dir,
        run_name=config.run_name,
        config=config,
        rank=rank,
        world_size=world_size,
    )
    monitor.maybe_heartbeat(
        step=0,
        micro_step=0,
        max_steps=int(getattr(config.training, "num_steps", 0)),
        note="loading_transformer",
        force=True,
    )

    if int(config.training.batch_size) != 1:
        raise ValueError(
            "Hazard V1 currently only supports training.batch_size=1 "
            f"(got {config.training.batch_size})."
        )

    # Load transformer model (frozen)
    transformer_path = resolve_transformer_path(config.paths.train_model_path)
    logger.info(f"Loading transformer from {transformer_path}")
    model_overrides = {
        "enable_action_residual_adapter": getattr(config.model, "enable_action_residual_adapter", False),
        "action_adapter_dim": getattr(config.model, "action_adapter_dim", 256),
        "action_adapter_dropout": getattr(config.model, "action_adapter_dropout", 0.0),
    }
    transformer = load_transformer(
        transformer_path,
        torch_dtype=torch.float32,
        torch_device='cpu',
        model_overrides=model_overrides,
    ).to(device=device, dtype=dtype)
    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad = False
    monitor.maybe_heartbeat(
        step=0,
        micro_step=0,
        max_steps=int(getattr(config.training, "num_steps", 0)),
        note="creating_scheduler_head",
        force=True,
    )

    # Create Hazard scheduler head
    logger.info("Creating HazardSchedulerHead")
    video_feature_dim = get_model_hidden_dim(transformer)
    scheduler_hidden_dim = get_scheduler_hidden_dim(config, video_feature_dim)
    logger.info(
        "HazardSchedulerHead dims: feature_dim=%d hidden_dim=%d",
        video_feature_dim,
        scheduler_hidden_dim,
    )
    scheduler_head = HazardSchedulerHead(
        feature_dim=video_feature_dim,
        hidden_dim=scheduler_hidden_dim,
        output_dim=1,
        use_context=False,
    ).to(device=device, dtype=torch.float32)

    # Keep the scheduler head as a plain module. We synchronize its gradients
    # manually in HazardTrainer to avoid DDP constructor collectives stalling
    # on this environment during parameter verification.
    if world_size > 1 and rank == 0:
        logger.info("Scheduler head gradient sync mode: manual_all_reduce")

    # Create dataset
    monitor.maybe_heartbeat(
        step=0,
        micro_step=0,
        max_steps=int(getattr(config.training, "num_steps", 0)),
        note="loading_dataset",
        force=True,
    )
    logger.info(f"Loading dataset from {config.paths.dataset_path}")
    dataset = MultiLatentLeRobotDataset(
        config=config,
        num_init_worker=config.training.init_workers,
    )
    val_ratio = getattr(getattr(config, "hazard", None), "val_split", 0.1)
    dataset = split_dataset_for_train(dataset, val_ratio=val_ratio)
    logger.info(f"Using training subset with {len(dataset)} samples")

    # Create dataloader
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.training.load_workers,
        pin_memory=True,
        drop_last=True,
    )
    monitor.maybe_heartbeat(
        step=0,
        micro_step=0,
        max_steps=int(getattr(config.training, "num_steps", 0)),
        note="initializing_logging_backends",
        force=True,
    )

    # Setup WandB (only on rank 0)
    wandb = None
    if rank == 0 and getattr(config.logging, 'enable_wandb', False):
        import wandb as wandb_module

        wandb = wandb_module
        wandb_mode = getattr(config.logging, "wandb_mode", os.environ.get("WANDB_MODE", "offline"))
        wandb_base_url = (
            getattr(config.logging, "wandb_base_url", None)
            or os.environ.get("WANDB_BASE_URL")
            or None
        )
        wandb_api_key = (
            getattr(config.logging, "wandb_api_key", None)
            or os.environ.get("WANDB_API_KEY")
            or None
        )
        wandb_dir = run_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        login_kwargs = {}
        if wandb_base_url:
            login_kwargs["host"] = wandb_base_url
        if wandb_api_key:
            login_kwargs["key"] = wandb_api_key
        if wandb_mode == "online" and login_kwargs:
            wandb.login(**login_kwargs)
        init_kwargs = dict(
            project=config.logging.wandb_project,
            config=_to_plain_python(dict(config)),
            name=config.run_name,
            dir=str(wandb_dir),
            mode=wandb_mode,
        )
        if wandb_base_url:
            init_kwargs["settings"] = wandb.Settings(base_url=wandb_base_url)
        wandb_entity = getattr(config.logging, "wandb_team_name", None) or None
        if wandb_entity:
            init_kwargs["entity"] = wandb_entity
        wandb.init(**init_kwargs)
        logger.info("WandB initialized (mode=%s)", wandb_mode)

    # Create trainer
    save_dir = Path(config.paths.save_root) / "checkpoints"
    trainer = HazardTrainer(
        transformer=transformer,
        scheduler_head=scheduler_head,
        config=config,
        device=device,
        dtype=dtype,
        train_loader=train_loader,
        save_dir=save_dir,
        wandb=wandb,
        monitor=monitor,
    )

    try:
        trainer.train()
        logger.info("Training finished!")
    except Exception as exc:
        logger.exception("Hazard training failed")
        monitor.mark_failed(
            step=trainer.step,
            micro_step=getattr(trainer, "micro_step", 0),
            max_steps=trainer.max_steps,
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )
        raise
    finally:
        if wandb is not None:
            wandb.finish()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
