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
from copy import deepcopy
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
from wan_va.utils import logger
from torch.utils.data import DataLoader, DistributedSampler, Subset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hazard scheduler")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    return parser.parse_args()


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

    # Convert to namespace for easier access
    class ConfigNamespace(dict):
        def __init__(self, d):
            super().__init__()
            for key, value in d.items():
                if isinstance(value, dict):
                    value = ConfigNamespace(value)
                else:
                    value = value
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

    # Setup device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    logger.info(f"Rank {rank}/{world_size} on device {device}")

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

    # Create Hazard scheduler head
    logger.info("Creating HazardSchedulerHead")
    video_feature_dim = get_model_hidden_dim(transformer)
    scheduler_head = HazardSchedulerHead(
        hidden_dim=video_feature_dim,
        output_dim=1,
        use_context=False,
    ).to(device, dtype=dtype)

    # Only the trainable scheduler head needs DDP. The transformer stays frozen
    # and can be replicated per rank without DistributedDataParallel.
    if world_size > 1:
        scheduler_head = torch.nn.parallel.DistributedDataParallel(
            scheduler_head,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # Create dataset
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

    # Setup WandB (only on rank 0)
    wandb = None
    if rank == 0 and getattr(config.logging, 'enable_wandb', False):
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(
            project=config.logging.wandb_project,
            config=vars(config),
        )

    # Create trainer
    save_dir = Path(config.paths.save_root)
    trainer = HazardTrainer(
        transformer=transformer,
        scheduler_head=scheduler_head,
        config=config,
        device=device,
        dtype=dtype,
        train_loader=train_loader,
        save_dir=save_dir,
        wandb=wandb,
    )

    # Start training
    trainer.train()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

    logger.info("Training finished!")


if __name__ == '__main__':
    main()
