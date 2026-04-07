# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""
REINFORCE trainer for HazardScheduler.

This module implements the policy gradient training loop for learning
adaptive video denoising schedules using the REINFORCE algorithm.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ..modules.hazard_scheduler import HazardSchedulerHead
from .hazard_runtime import HazardRolloutRunner
from . import logger


def _to_plain_python(value):
    """Recursively convert config namespaces / tensors to plain Python containers."""
    if isinstance(value, dict):
        return {k: _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_python(v) for v in value]
    return value


class HazardTrainer:
    """
    REINFORCE trainer for HazardScheduler.

    Training loop:
    1. Sample batch from dataset
    2. Execute rollout with HazardScheduler (collect trajectory)
    3. Execute baseline rollout with fixed K
    4. Compute advantage: A = R_hazard - R_baseline
    5. Compute policy gradient loss: L = -sum(log_prob * A)
    6. Add KL regularization to prevent drift
    7. Backprop and update
    """

    def __init__(
        self,
        transformer,
        scheduler_head: HazardSchedulerHead,
        config,
        device,
        dtype,
        train_loader: DataLoader,
        save_dir: Path,
        wandb=None,
    ):
        """
        Args:
            transformer: WanTransformer3DModel (frozen or trainable)
            scheduler_head: HazardSchedulerHead module (trainable)
            config: Configuration object
            device: torch device
            dtype: torch dtype
            train_loader: DataLoader for training data
            save_dir: Directory to save checkpoints
            wandb: WandB instance for logging (optional)
        """
        self.transformer = transformer
        self.scheduler_head = scheduler_head
        self.config = config
        self.device = device
        self.dtype = dtype
        self.train_loader = train_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = wandb

        # Training hyperparameters
        # Read from nested config: training.hazard.*
        hazard_config = getattr(config, 'hazard', None)
        if hazard_config is None:
            # Fallback to top-level attributes
            self.learning_rate = float(getattr(config, 'hazard_lr', 1e-4))
            self.lambda_cost = float(getattr(config, 'lambda_cost', 0.1))
            self.lambda_kl = float(getattr(config, 'lambda_kl', 0.01))
            self.baseline_K = int(getattr(config, 'baseline_K', 10))
            self.max_steps = int(getattr(config, 'hazard_max_steps', 10000))
            self.save_interval = int(getattr(config, 'save_interval', 1000))
            self.log_interval = int(getattr(config, 'log_interval', 10))
        else:
            # Read from hazard config section
            self.learning_rate = float(getattr(hazard_config, 'learning_rate', 1e-4))
            self.lambda_cost = float(getattr(hazard_config, 'lambda_cost', 0.1))
            self.lambda_kl = float(getattr(hazard_config, 'lambda_kl', 0.01))
            self.baseline_K = int(getattr(hazard_config, 'baseline_K', 10))
            self.max_steps = int(getattr(config.training, 'num_steps', 10000))
            self.save_interval = int(getattr(hazard_config, 'save_interval', 1000))
            self.log_interval = int(getattr(hazard_config, 'log_interval', 10))

        # Read gradient_accumulation_steps from training config
        self.gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))

        # Optimizer for scheduler_head only
        self.optimizer = torch.optim.AdamW(
            self.scheduler_head.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        # Rollout runner
        self.rollout_runner = HazardRolloutRunner(
            transformer=transformer,
            scheduler_head=scheduler_head,
            config=config,
            device=device,
            dtype=dtype,
        )

        # Training state
        self.step = 0
        self.train_loader_iter = None

        # Set transformer to eval mode (frozen during Hazard training)
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False

        # Set scheduler_head to train mode
        self.scheduler_head.train()

        logger.info("HazardTrainer initialized")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Lambda cost: {self.lambda_cost}")
        logger.info(f"  Lambda KL: {self.lambda_kl}")
        logger.info(f"  Baseline K: {self.baseline_K}")
        logger.info(f"  Max steps: {self.max_steps}")

    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)

        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                epoch = getattr(self.train_loader.sampler, "epoch", 0) + 1
                self.train_loader.sampler.set_epoch(epoch)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)

        return batch

    def _prepare_rollout_inputs(self, batch):
        """
        Prepare inputs for rollout from batch.

        Args:
            batch: Dict with keys 'latents', 'actions', 'text_emb', etc.

        Returns:
            Dict with video_noise, action_noise, text_emb, gt_action, latent_cond
        """
        # Move to device
        latents = batch['latents'].to(self.device, dtype=self.dtype)
        actions = batch['actions'].to(self.device, dtype=self.dtype)
        text_emb = batch['text_emb'].to(self.device, dtype=self.dtype)

        # Generate noise
        video_noise = torch.randn_like(latents)
        action_noise = torch.randn_like(actions)

        # Extract conditioning frame (first frame)
        latent_cond = latents[:, :, 0:1].clone()

        return {
            'video_noise': video_noise,
            'action_noise': action_noise,
            'text_emb': text_emb,
            'gt_action': actions,
            'latent_cond': latent_cond,
        }

    def _compute_kl_regularization(self, trajectory, baseline_trajectory):
        """
        Compute KL divergence between policy and baseline.

        KL(π || π_ref) = sum_k [h_k * log(h_k / h_ref_k) + (1-h_k) * log((1-h_k) / (1-h_ref_k))]

        For simplicity, we use a Monte Carlo estimate based on the sampled trajectory.
        """
        if len(baseline_trajectory) == 0:
            # Baseline used fixed K, no stochastic decisions
            return torch.tensor(0.0, device=self.device)

        kl_sum = 0.0
        min_len = min(len(trajectory), len(baseline_trajectory))

        for k in range(min_len):
            h_k = trajectory[k]['h_k']
            h_ref_k = baseline_trajectory[k]['h_k'] if k < len(baseline_trajectory) else torch.tensor(0.5, device=self.device)

            # Clamp to avoid log(0)
            h_k = torch.clamp(h_k, 1e-6, 1 - 1e-6)
            h_ref_k = torch.clamp(h_ref_k, 1e-6, 1 - 1e-6)

            # Binary KL divergence
            kl = h_k * torch.log(h_k / h_ref_k) + (1 - h_k) * torch.log((1 - h_k) / (1 - h_ref_k))
            kl_sum += kl.mean()

        return kl_sum

    def _train_step(self, batch_idx):
        """
        Execute one training step.

        CRITICAL: We need to collect log_probs WITH gradients during rollout,
        then compute the policy gradient loss.

        Returns:
            Dict with training metrics
        """
        step_t0 = time.perf_counter()

        # Get batch
        batch = self._get_next_batch()
        rollout_inputs = self._prepare_rollout_inputs(batch)

        # Execute Hazard rollout (stochastic) - KEEP GRADIENTS for log_probs
        # The transformer forward passes are still no_grad, but scheduler_head is not
        trajectory, pred_action, reward = self.rollout_runner.rollout_single_sample(
            video_noise=rollout_inputs['video_noise'],
            action_noise=rollout_inputs['action_noise'],
            text_emb=rollout_inputs['text_emb'],
            gt_action=rollout_inputs['gt_action'],
            latent_cond=rollout_inputs['latent_cond'],
            mode='train',
        )

        # Execute baseline rollout (fixed K) - no gradients needed
        with torch.no_grad():
            baseline_trajectory, baseline_pred_action, baseline_reward = self.rollout_runner.rollout_fixed_K(
                video_noise=rollout_inputs['video_noise'],
                action_noise=rollout_inputs['action_noise'],
                text_emb=rollout_inputs['text_emb'],
                gt_action=rollout_inputs['gt_action'],
                K=self.baseline_K,
                latent_cond=rollout_inputs['latent_cond'],
            )

        # Compute advantage
        # Both reward and baseline_reward are Python floats, no need to detach
        advantage = reward - baseline_reward

        # Compute policy gradient loss
        # REINFORCE: L = -sum(log_prob * advantage)
        policy_loss = 0.0
        for step_info in trajectory:
            if step_info['log_prob'] is not None:
                # log_prob should have gradients, advantage is detached
                policy_loss += -step_info['log_prob'] * advantage

        policy_loss = policy_loss / max(len(trajectory), 1)

        # Note: KL regularization is disabled because baseline uses fixed K (no stochastic policy)
        # If needed, consider using entropy bonus instead: -lambda_entropy * sum(h_k * log(h_k))
        kl_loss = 0.0

        # Total loss
        total_loss = policy_loss
        total_loss = total_loss / self.gradient_accumulation_steps

        # Backward
        total_loss.backward()

        # Optimizer step (if accumulation is done)
        should_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if should_step:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.scheduler_head.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Metrics
        metrics = {
            'policy_loss': policy_loss.item() * self.gradient_accumulation_steps,
            'kl_loss': kl_loss,
            'total_loss': total_loss.item() * self.gradient_accumulation_steps,
            'reward': reward,
            'baseline_reward': baseline_reward,
            'advantage': advantage,
            'video_steps': len(trajectory),
            'baseline_video_steps': self.baseline_K,
            'step_time': time.perf_counter() - step_t0,
        }

        return metrics

    def train(self):
        """Main training loop."""
        logger.info("Starting Hazard training...")

        pbar = tqdm(total=self.max_steps, desc="Training", disable=(self.config.rank != 0))

        for batch_idx in range(self.max_steps):
            metrics = self._train_step(batch_idx)

            # Update step counter
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.step += 1

            # Logging
            if self.step % self.log_interval == 0 and self.config.rank == 0:
                log_str = (
                    f"Step {self.step} | "
                    f"Loss: {metrics['total_loss']:.4f} | "
                    f"Reward: {metrics['reward']:.4f} | "
                    f"Baseline: {metrics['baseline_reward']:.4f} | "
                    f"Adv: {metrics['advantage']:.4f} | "
                    f"Steps: {metrics['video_steps']}/{metrics['baseline_video_steps']} | "
                    f"Time: {metrics['step_time']:.2f}s"
                )
                logger.info(log_str)

                if self.wandb is not None:
                    self.wandb.log({
                        'train/policy_loss': metrics['policy_loss'],
                        'train/kl_loss': metrics['kl_loss'],
                        'train/total_loss': metrics['total_loss'],
                        'train/reward': metrics['reward'],
                        'train/baseline_reward': metrics['baseline_reward'],
                        'train/advantage': metrics['advantage'],
                        'train/video_steps': metrics['video_steps'],
                        'train/step_time': metrics['step_time'],
                    }, step=self.step)

            # Save checkpoint
            if self.step % self.save_interval == 0 and self.config.rank == 0:
                self.save_checkpoint()

            pbar.update(1)

        pbar.close()
        logger.info("Training completed!")

        # Save final checkpoint
        if self.config.rank == 0:
            self.save_checkpoint(final=True)

    def save_checkpoint(self, final=False):
        """Save checkpoint."""
        if final:
            ckpt_path = self.save_dir / "hazard_scheduler_final.pt"
        else:
            ckpt_path = self.save_dir / f"hazard_scheduler_step_{self.step}.pt"

        checkpoint = {
            'step': self.step,
            'scheduler_head_state_dict': self.scheduler_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': _to_plain_python(dict(self.config)),
        }

        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")

    def load_checkpoint(self, ckpt_path):
        """Load checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.scheduler_head.load_state_dict(checkpoint['scheduler_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        logger.info(f"Checkpoint loaded from {ckpt_path}, resuming from step {self.step}")
