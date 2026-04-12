#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""
Offline evaluation script for Hazard scheduler.

This script evaluates a trained HazardScheduler on the validation set,
comparing its performance against fixed-K baselines.
"""

import argparse
import os
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
from wan_va.utils.hazard_rollout_inputs import prepare_chunked_rollout_inputs
from wan_va.utils.hazard_runtime import HazardRolloutRunner
from wan_va.utils.hazard_reward import DualAnchorRewardEvaluator
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from wan_va.utils import init_logger, logger
from torch.utils.data import DataLoader, Subset


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

        hazard_config = getattr(config, 'hazard', None)
        if hazard_config is None:
            self.baseline_Ks = getattr(config, 'baseline_Ks', [5, 10, 15, 20])
        else:
            self.baseline_Ks = getattr(hazard_config, 'baseline_Ks', [5, 10, 15, 20])

        # Rollout runner
        self.rollout_runner = HazardRolloutRunner(
            transformer=transformer,
            scheduler_head=scheduler_head,
            config=config,
            device=device,
            dtype=dtype,
        )
        self.reward_evaluator = DualAnchorRewardEvaluator(
            config=config,
            device=device,
        )

        # Set to eval mode
        self.transformer.eval()
        self.scheduler_head.eval()

        logger.info("HazardEvaluator initialized")
        logger.info(f"  Baseline Ks: {self.baseline_Ks}")

    def _prepare_rollout_inputs(self, batch):
        """Prepare rollout inputs aligned with real cached inference."""
        return prepare_chunked_rollout_inputs(
            batch=batch,
            device=self.device,
            dtype=self.dtype,
            frame_chunk_size=int(self.config.frame_chunk_size),
            sample_history=False,
        )

    @torch.no_grad()
    def evaluate_sample(self, batch) -> Dict:
        """
        Evaluate a single sample.

        Returns:
            Dict with metrics for Hazard and all baselines
        """
        rollout_inputs = self._prepare_rollout_inputs(batch)

        # Evaluate Hazard scheduler
        hazard_result = self.rollout_runner.rollout_single_sample(
            video_noise=rollout_inputs['video_noise'],
            action_noise=rollout_inputs['action_noise'],
            text_emb=rollout_inputs['text_emb'],
            clean_history_latents=rollout_inputs['clean_history_latents'],
            clean_history_actions=rollout_inputs['clean_history_actions'],
            latent_cond=rollout_inputs['latent_cond'],
            action_cond=rollout_inputs['action_cond'],
            mode='eval',
        )
        anchor_lo_result = self.rollout_runner.rollout_fixed_K(
            video_noise=rollout_inputs['video_noise'],
            action_noise=rollout_inputs['action_noise'],
            text_emb=rollout_inputs['text_emb'],
            K=self.reward_evaluator.anchor_lo_k,
            clean_history_latents=rollout_inputs['clean_history_latents'],
            clean_history_actions=rollout_inputs['clean_history_actions'],
            latent_cond=rollout_inputs['latent_cond'],
            action_cond=rollout_inputs['action_cond'],
        )
        if self.reward_evaluator.anchor_hi_k == self.reward_evaluator.anchor_lo_k:
            anchor_hi_result = anchor_lo_result
        else:
            anchor_hi_result = self.rollout_runner.rollout_fixed_K(
                video_noise=rollout_inputs['video_noise'],
                action_noise=rollout_inputs['action_noise'],
                text_emb=rollout_inputs['text_emb'],
                K=self.reward_evaluator.anchor_hi_k,
                clean_history_latents=rollout_inputs['clean_history_latents'],
                clean_history_actions=rollout_inputs['clean_history_actions'],
                latent_cond=rollout_inputs['latent_cond'],
                action_cond=rollout_inputs['action_cond'],
            )
        hazard_breakdown = self.reward_evaluator.evaluate(
            current=hazard_result,
            anchor_lo=anchor_lo_result,
            anchor_hi=anchor_hi_result,
            gt_action=rollout_inputs['gt_action'],
            gt_action_mask=rollout_inputs['gt_action_mask'],
        )

        results = {
            'hazard': {
                'reward': hazard_breakdown.reward,
                'quality': hazard_breakdown.quality,
                'cost': hazard_breakdown.cost,
                'video_steps': hazard_result.video_steps,
                'l_seq': hazard_breakdown.l_seq_cur,
                'l_delta': hazard_breakdown.l_delta_cur,
            }
        }

        # Evaluate baselines
        for K in self.baseline_Ks:
            if K == anchor_lo_result.video_steps:
                baseline_result = anchor_lo_result
            elif K == anchor_hi_result.video_steps:
                baseline_result = anchor_hi_result
            else:
                baseline_result = self.rollout_runner.rollout_fixed_K(
                    video_noise=rollout_inputs['video_noise'],
                    action_noise=rollout_inputs['action_noise'],
                    text_emb=rollout_inputs['text_emb'],
                    K=K,
                    clean_history_latents=rollout_inputs['clean_history_latents'],
                    clean_history_actions=rollout_inputs['clean_history_actions'],
                    latent_cond=rollout_inputs['latent_cond'],
                    action_cond=rollout_inputs['action_cond'],
                )
            baseline_breakdown = self.reward_evaluator.evaluate(
                current=baseline_result,
                anchor_lo=anchor_lo_result,
                anchor_hi=anchor_hi_result,
                gt_action=rollout_inputs['gt_action'],
                gt_action_mask=rollout_inputs['gt_action_mask'],
            )

            results[f'baseline_K{K}'] = {
                'reward': baseline_breakdown.reward,
                'quality': baseline_breakdown.quality,
                'cost': baseline_breakdown.cost,
                'video_steps': baseline_result.video_steps,
                'l_seq': baseline_breakdown.l_seq_cur,
                'l_delta': baseline_breakdown.l_delta_cur,
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
            qualities = [r[method]['quality'] for r in all_results]
            costs = [r[method]['cost'] for r in all_results]
            seq_losses = [r[method]['l_seq'] for r in all_results]
            delta_losses = [r[method]['l_delta'] for r in all_results]

            aggregated[method] = {
                'reward_mean': sum(rewards) / len(rewards),
                'reward_std': torch.tensor(rewards).std().item(),
                'quality_mean': sum(qualities) / len(qualities),
                'quality_std': torch.tensor(qualities).std().item(),
                'cost_mean': sum(costs) / len(costs),
                'cost_std': torch.tensor(costs).std().item(),
                'video_steps_mean': sum(video_steps) / len(video_steps),
                'video_steps_std': torch.tensor(video_steps).float().std().item(),
                'l_seq_mean': sum(seq_losses) / len(seq_losses),
                'l_seq_std': torch.tensor(seq_losses).std().item(),
                'l_delta_mean': sum(delta_losses) / len(delta_losses),
                'l_delta_std': torch.tensor(delta_losses).std().item(),
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
        logger.info(f"  Quality:      {hazard['quality_mean']:.4f} ± {hazard['quality_std']:.4f}")
        logger.info(f"  Cost:         {hazard['cost_mean']:.4f} ± {hazard['cost_std']:.4f}")
        logger.info(f"  Video Steps:  {hazard['video_steps_mean']:.2f} ± {hazard['video_steps_std']:.2f}")
        logger.info(f"  L_seq:        {hazard['l_seq_mean']:.4f} ± {hazard['l_seq_std']:.4f}")
        logger.info(f"  L_delta:      {hazard['l_delta_mean']:.4f} ± {hazard['l_delta_std']:.4f}")

        # Print baseline results
        logger.info(f"\nBaselines:")
        for method in sorted(aggregated.keys()):
            if method.startswith('baseline_'):
                baseline = aggregated[method]
                K = method.split('_K')[1]
                logger.info(f"  K={K}:")
                logger.info(f"    Reward:       {baseline['reward_mean']:.4f} ± {baseline['reward_std']:.4f}")
                logger.info(f"    Quality:      {baseline['quality_mean']:.4f} ± {baseline['quality_std']:.4f}")
                logger.info(f"    Cost:         {baseline['cost_mean']:.4f} ± {baseline['cost_std']:.4f}")
                logger.info(f"    L_seq:        {baseline['l_seq_mean']:.4f} ± {baseline['l_seq_std']:.4f}")
                logger.info(f"    L_delta:      {baseline['l_delta_mean']:.4f} ± {baseline['l_delta_std']:.4f}")

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


def get_scheduler_hidden_dim(config, default_dim: int) -> int:
    model_cfg = getattr(config, "model", None)
    configured_dim = getattr(model_cfg, "hazard_hidden_dim", None)
    if configured_dim is None:
        return int(default_dim)
    configured_dim = int(configured_dim)
    if configured_dim <= 0:
        raise ValueError(f"hazard_hidden_dim must be positive, got {configured_dim}")
    return configured_dim


def normalize_scheduler_state_dict(state_dict):
    keys = list(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def split_dataset_for_eval(dataset, val_ratio: float = 0.1):
    total = len(dataset)
    if total <= 1:
        return dataset

    val_size = max(1, int(total * val_ratio))
    start = max(0, total - val_size)
    return Subset(dataset, range(start, total))


def main():
    init_logger(console=True, force=True)
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
    ).to(device, dtype=dtype)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    scheduler_head.load_state_dict(
        normalize_scheduler_state_dict(checkpoint['scheduler_head_state_dict'])
    )
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
