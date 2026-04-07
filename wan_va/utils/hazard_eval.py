#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""
Offline evaluation script for Hazard scheduler.

This script evaluates a trained HazardScheduler on the validation set,
comparing its performance against fixed-K baselines.
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from wan_va.modules.model import WanTransformer3DModel
from wan_va.modules.hazard_scheduler import HazardSchedulerHead
from wan_va.modules.utils import load_transformer
from wan_va.configs import VA_CONFIGS
from wan_va.utils.hazard_runtime import HazardRolloutRunner
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from wan_va.utils import logger
from torch.utils.data import DataLoader, Subset


class HazardEvaluator:
    """
    Offline evaluator for Hazard scheduler.

    Evaluates on validation set and compares:
    - Hazard scheduler (adaptive)
    - Fixed K baselines (K=5, 10, 15, 20)
    """

    def __init__(
        self,
        transformer,
        scheduler_head: HazardSchedulerHead,
        config,
        device,
        dtype,
        val_loader: DataLoader,
    ):
        self.transformer = transformer
        self.scheduler_head = scheduler_head
        self.config = config
        self.device = device
        self.dtype = dtype
        self.val_loader = val_loader

        # Evaluation settings
        # Read from nested config: hazard.*
        hazard_config = getattr(config, 'hazard', None)
        if hazard_config is None:
            # Fallback to top-level attributes
            self.lambda_cost = getattr(config, 'lambda_cost', 0.1)
            self.baseline_Ks = getattr(config, 'baseline_Ks', [5, 10, 15, 20])
        else:
            self.lambda_cost = getattr(hazard_config, 'lambda_cost', 0.1)
            self.baseline_Ks = getattr(hazard_config, 'baseline_Ks', [5, 10, 15, 20])

        # Rollout runner
        self.rollout_runner = HazardRolloutRunner(
            transformer=transformer,
            scheduler_head=scheduler_head,
            config=config,
            device=device,
            dtype=dtype,
        )

        # Set to eval mode
        self.transformer.eval()
        self.scheduler_head.eval()

        logger.info("HazardEvaluator initialized")
        logger.info(f"  Lambda cost: {self.lambda_cost}")
        logger.info(f"  Baseline Ks: {self.baseline_Ks}")

    def _prepare_rollout_inputs(self, batch):
        """Prepare inputs for rollout from batch."""
        latents = batch['latents'].to(self.device, dtype=self.dtype)
        actions = batch['actions'].to(self.device, dtype=self.dtype)
        text_emb = batch['text_emb'].to(self.device, dtype=self.dtype)

        video_noise = torch.randn_like(latents)
        action_noise = torch.randn_like(actions)
        latent_cond = latents[:, :, 0:1].clone()

        return {
            'video_noise': video_noise,
            'action_noise': action_noise,
            'text_emb': text_emb,
            'gt_action': actions,
            'latent_cond': latent_cond,
        }

    @torch.no_grad()
    def evaluate_sample(self, batch) -> Dict:
        """
        Evaluate a single sample.

        Returns:
            Dict with metrics for Hazard and all baselines
        """
        rollout_inputs = self._prepare_rollout_inputs(batch)

        # Evaluate Hazard scheduler
        trajectory, pred_action, reward = self.rollout_runner.rollout_single_sample(
            video_noise=rollout_inputs['video_noise'],
            action_noise=rollout_inputs['action_noise'],
            text_emb=rollout_inputs['text_emb'],
            gt_action=rollout_inputs['gt_action'],
            latent_cond=rollout_inputs['latent_cond'],
            mode='eval',
        )

        results = {
            'hazard': {
                'reward': reward,
                'video_steps': len(trajectory),
                'action_loss': -reward + self.lambda_cost * len(trajectory),
            }
        }

        # Evaluate baselines
        for K in self.baseline_Ks:
            baseline_trajectory, baseline_pred_action, baseline_reward = self.rollout_runner.rollout_fixed_K(
                video_noise=rollout_inputs['video_noise'],
                action_noise=rollout_inputs['action_noise'],
                text_emb=rollout_inputs['text_emb'],
                gt_action=rollout_inputs['gt_action'],
                K=K,
                latent_cond=rollout_inputs['latent_cond'],
            )

            results[f'baseline_K{K}'] = {
                'reward': baseline_reward,
                'video_steps': K,
                'action_loss': -baseline_reward + self.lambda_cost * K,
            }

        return results

    def evaluate(self, num_samples: int = None) -> Dict:
        """
        Evaluate on validation set.

        Args:
            num_samples: Number of samples to evaluate (None = all)

        Returns:
            Dict with aggregated metrics
        """
        logger.info("Starting evaluation...")

        all_results = []
        num_evaluated = 0

        pbar = tqdm(self.val_loader, desc="Evaluating", disable=(self.config.rank != 0))

        for batch in pbar:
            results = self.evaluate_sample(batch)
            all_results.append(results)
            num_evaluated += 1

            if num_samples is not None and num_evaluated >= num_samples:
                break

        pbar.close()

        # Aggregate results
        aggregated = self._aggregate_results(all_results)

        # Print summary
        self._print_summary(aggregated)

        return aggregated

    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across all samples."""
        aggregated = {}

        # Get all method names (hazard, baseline_K5, etc.)
        method_names = list(all_results[0].keys())

        for method in method_names:
            rewards = [r[method]['reward'] for r in all_results]
            video_steps = [r[method]['video_steps'] for r in all_results]
            action_losses = [r[method]['action_loss'] for r in all_results]

            aggregated[method] = {
                'reward_mean': sum(rewards) / len(rewards),
                'reward_std': torch.tensor(rewards).std().item(),
                'video_steps_mean': sum(video_steps) / len(video_steps),
                'video_steps_std': torch.tensor(video_steps).float().std().item(),
                'action_loss_mean': sum(action_losses) / len(action_losses),
                'action_loss_std': torch.tensor(action_losses).std().item(),
            }

        return aggregated

    def _print_summary(self, aggregated: Dict):
        """Print evaluation summary."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)

        # Print Hazard results
        hazard = aggregated['hazard']
        logger.info(f"\nHazard Scheduler:")
        logger.info(f"  Reward:       {hazard['reward_mean']:.4f} ± {hazard['reward_std']:.4f}")
        logger.info(f"  Video Steps:  {hazard['video_steps_mean']:.2f} ± {hazard['video_steps_std']:.2f}")
        logger.info(f"  Action Loss:  {hazard['action_loss_mean']:.4f} ± {hazard['action_loss_std']:.4f}")

        # Print baseline results
        logger.info(f"\nBaselines:")
        for method in sorted(aggregated.keys()):
            if method.startswith('baseline_'):
                baseline = aggregated[method]
                K = method.split('_K')[1]
                logger.info(f"  K={K}:")
                logger.info(f"    Reward:       {baseline['reward_mean']:.4f} ± {baseline['reward_std']:.4f}")
                logger.info(f"    Action Loss:  {baseline['action_loss_mean']:.4f} ± {baseline['action_loss_std']:.4f}")

        # Compute improvement over best baseline
        # Reward is defined as: R = -L_act - lambda_cost * K
        # Higher reward is better, so we want the MAX reward baseline
        best_baseline_method = max(
            [m for m in aggregated.keys() if m.startswith('baseline_')],
            key=lambda m: aggregated[m]['reward_mean']
        )
        best_baseline_reward = aggregated[best_baseline_method]['reward_mean']
        hazard_reward = aggregated['hazard']['reward_mean']
        improvement = ((hazard_reward - best_baseline_reward) / abs(best_baseline_reward)) * 100

        logger.info(f"\nImprovement over best baseline ({best_baseline_method}):")
        logger.info(f"  Reward improvement: {improvement:+.2f}%")

        logger.info("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hazard scheduler")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to scheduler checkpoint")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output file for results (JSON)")
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


def resolve_transformer_path(model_root: str) -> str:
    model_root = str(model_root)
    transformer_dir = Path(model_root) / "transformer"
    return str(transformer_dir) if transformer_dir.is_dir() else model_root


def get_model_hidden_dim(transformer: WanTransformer3DModel) -> int:
    config = transformer.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    return int(config.num_attention_heads) * int(config.attention_head_dim)


def split_dataset_for_eval(dataset, val_ratio: float = 0.1):
    total = len(dataset)
    if total <= 1:
        return dataset

    val_size = max(1, int(total * val_ratio))
    start = max(0, total - val_size)
    return Subset(dataset, range(start, total))


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    config.rank = 0  # Single GPU evaluation

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Load transformer model
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

    # Create Hazard scheduler head
    logger.info("Creating HazardSchedulerHead")
    video_feature_dim = get_model_hidden_dim(transformer)
    scheduler_head = HazardSchedulerHead(
        hidden_dim=video_feature_dim,
        output_dim=1,
        use_context=False,
    ).to(device, dtype=dtype)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    scheduler_head.load_state_dict(checkpoint['scheduler_head_state_dict'])
    logger.info(f"Loaded checkpoint from step {checkpoint['step']}")

    # Create validation dataset
    logger.info(f"Loading validation dataset from {config.paths.dataset_path}")
    # Note: MultiLatentLeRobotDataset doesn't have split parameter, loads all data
    val_dataset = MultiLatentLeRobotDataset(
        config=config,
        num_init_worker=4,
    )
    val_ratio = getattr(getattr(config, "hazard", None), "val_split", 0.1)
    val_dataset = split_dataset_for_eval(val_dataset, val_ratio=val_ratio)
    logger.info(f"Using evaluation subset with {len(val_dataset)} samples")

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Evaluate one sample at a time
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create evaluator
    evaluator = HazardEvaluator(
        transformer=transformer,
        scheduler_head=scheduler_head,
        config=config,
        device=device,
        dtype=dtype,
        val_loader=val_loader,
    )

    # Run evaluation
    results = evaluator.evaluate(num_samples=args.num_samples)

    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
