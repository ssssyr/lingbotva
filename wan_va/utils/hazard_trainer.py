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
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ..modules.hazard_scheduler import HazardSchedulerHead
from .hazard_rollout_inputs import prepare_chunked_rollout_inputs
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
        monitor=None,
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
        self.monitor = monitor

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
        self.micro_step = 0
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
        return prepare_chunked_rollout_inputs(
            batch=batch,
            device=self.device,
            dtype=self.dtype,
            frame_chunk_size=int(self.config.frame_chunk_size),
            sample_history=True,
        )

    @staticmethod
    def _unwrap_scheduler_head(module):
        return module.module if hasattr(module, "module") else module

    @staticmethod
    def _normalize_scheduler_state_dict(state_dict):
        keys = list(state_dict.keys())
        if keys and all(key.startswith("module.") for key in keys):
            return {key[len("module."):]: value for key, value in state_dict.items()}
        return state_dict

    def _compute_kl_regularization(self, trajectory):
        """
        Compute KL divergence to a deterministic fixed-K stop policy.

        For the reference policy, steps before `baseline_K` almost surely continue,
        while the baseline step and later steps almost surely stop. This keeps the
        learned scheduler close to the fixed-K rollout used for the reward baseline.
        """
        if len(trajectory) == 0:
            return torch.tensor(0.0, device=self.device)

        eps = 1e-6
        kl_sum = torch.tensor(0.0, device=self.device)
        reference_stop_idx = max(int(self.baseline_K) - 1, 0)

        for step_info in trajectory:
            step_idx = int(step_info['step_idx'])
            h_k = step_info['h_k']
            h_k = torch.clamp(h_k, 1e-6, 1 - 1e-6)
            reference_prob = eps if step_idx < reference_stop_idx else 1.0 - eps
            h_ref_k = torch.full_like(h_k, reference_prob)

            # Binary KL divergence
            kl = h_k * torch.log(h_k / h_ref_k) + (1 - h_k) * torch.log((1 - h_k) / (1 - h_ref_k))
            kl_sum = kl_sum + kl.mean()

        return kl_sum / max(len(trajectory), 1)

    def _reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        reduced = dict(metrics)
        if not dist.is_initialized():
            return reduced

        mean_keys = [
            "policy_loss",
            "kl_loss",
            "total_loss",
            "reward",
            "baseline_reward",
            "advantage",
            "video_steps",
            "grad_norm",
            "lr",
        ]
        values = torch.tensor(
            [float(metrics[key]) for key in mean_keys],
            dtype=torch.float32,
            device=self.device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.AVG)
        for key, value in zip(mean_keys, values.tolist()):
            reduced[key] = value

        step_time = torch.tensor([float(metrics["step_time"])], dtype=torch.float32, device=self.device)
        dist.all_reduce(step_time, op=dist.ReduceOp.MAX)
        reduced["step_time"] = step_time.item()
        return reduced

    def _add_reward_decomposition(self, metrics: Dict[str, float]) -> Dict[str, float]:
        enriched = dict(metrics)
        reward = float(enriched["reward"])
        baseline_reward = float(enriched["baseline_reward"])
        video_steps = float(enriched["video_steps"])
        baseline_video_steps = float(enriched["baseline_video_steps"])

        enriched["compute_penalty"] = self.lambda_cost * video_steps
        enriched["baseline_compute_penalty"] = self.lambda_cost * baseline_video_steps
        enriched["compute_penalty_gap"] = (
            enriched["baseline_compute_penalty"] - enriched["compute_penalty"]
        )
        enriched["implied_action_loss"] = -reward - enriched["compute_penalty"]
        enriched["implied_baseline_action_loss"] = (
            -baseline_reward - enriched["baseline_compute_penalty"]
        )
        enriched["implied_action_loss_gap"] = (
            enriched["implied_action_loss"] - enriched["implied_baseline_action_loss"]
        )
        return enriched

    def _sync_scheduler_gradients(self) -> None:
        if not dist.is_initialized():
            return

        world_size = float(dist.get_world_size())
        for param in self.scheduler_head.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

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
            gt_action_mask=rollout_inputs['gt_action_mask'],
            clean_history_latents=rollout_inputs['clean_history_latents'],
            clean_history_actions=rollout_inputs['clean_history_actions'],
            latent_cond=rollout_inputs['latent_cond'],
            action_cond=rollout_inputs['action_cond'],
            mode='train',
        )

        # Execute baseline rollout (fixed K) - no gradients needed
        with torch.no_grad():
            _, baseline_pred_action, baseline_reward = self.rollout_runner.rollout_fixed_K(
                video_noise=rollout_inputs['video_noise'],
                action_noise=rollout_inputs['action_noise'],
                text_emb=rollout_inputs['text_emb'],
                gt_action=rollout_inputs['gt_action'],
                gt_action_mask=rollout_inputs['gt_action_mask'],
                K=self.baseline_K,
                clean_history_latents=rollout_inputs['clean_history_latents'],
                clean_history_actions=rollout_inputs['clean_history_actions'],
                latent_cond=rollout_inputs['latent_cond'],
                action_cond=rollout_inputs['action_cond'],
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

        kl_loss = self._compute_kl_regularization(trajectory)

        # Total loss
        total_loss = policy_loss + self.lambda_kl * kl_loss
        total_loss = total_loss / self.gradient_accumulation_steps

        # Backward
        total_loss.backward()

        # Optimizer step (if accumulation is done)
        should_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        grad_norm = 0.0
        lr = self.optimizer.param_groups[0]["lr"]
        if should_step:
            self._sync_scheduler_gradients()
            # Gradient clipping
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.scheduler_head.parameters(), max_norm=1.0)
            grad_norm = float(grad_norm_tensor.item())
            if torch.isfinite(torch.tensor(grad_norm)):
                self.optimizer.step()
            else:
                logger.warning("Non-finite grad_norm detected at step=%s micro_step=%s; skipping optimizer step", self.step, self.micro_step)
            self.optimizer.zero_grad(set_to_none=True)

        # Metrics
        metrics = {
            'policy_loss': policy_loss.item() * self.gradient_accumulation_steps,
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item() * self.gradient_accumulation_steps,
            'reward': reward,
            'baseline_reward': baseline_reward,
            'advantage': advantage,
            'video_steps': len(trajectory),
            'baseline_video_steps': self.baseline_K,
            'step_time': time.perf_counter() - step_t0,
            'grad_norm': grad_norm,
            'lr': lr,
            'should_step': should_step,
            'segment_length': rollout_inputs.get('segment_length'),
            'history_start_frame': rollout_inputs.get('history_start_frame'),
            'history_end_frame': rollout_inputs.get('history_end_frame'),
            'target_start_frame': rollout_inputs.get('target_start_frame'),
            'target_end_frame': rollout_inputs.get('target_end_frame'),
        }

        return self._add_reward_decomposition(metrics)

    def train(self):
        """Main training loop."""
        logger.info("Starting Hazard training...")

        pbar = tqdm(
            total=self.max_steps,
            desc="HazardTrain",
            disable=(self.config.rank != 0),
            dynamic_ncols=True,
        )
        self.optimizer.zero_grad(set_to_none=True)
        if self.monitor is not None:
            self.monitor.maybe_heartbeat(
                step=self.step,
                micro_step=self.micro_step,
                max_steps=self.max_steps,
                note="training_started",
                force=True,
            )

        while self.step < self.max_steps:
            metrics = self._train_step(self.micro_step)
            self.micro_step += 1

            if not metrics["should_step"]:
                if self.monitor is not None:
                    self.monitor.maybe_heartbeat(
                        step=self.step,
                        micro_step=self.micro_step,
                        max_steps=self.max_steps,
                        note="accumulating_gradients",
                    )
                continue

            self.step += 1
            reduced_metrics = self._add_reward_decomposition(self._reduce_metrics(metrics))

            if self.config.rank == 0:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{reduced_metrics['total_loss']:.4f}",
                    reward=f"{reduced_metrics['reward']:.4f}",
                    steps=f"{reduced_metrics['video_steps']:.2f}",
                    grad=f"{reduced_metrics['grad_norm']:.2f}",
                )

                if self.monitor is not None:
                    self.monitor.log_metrics(
                        step=self.step,
                        micro_step=self.micro_step,
                        max_steps=self.max_steps,
                        metrics=reduced_metrics,
                    )

                if self.step == 1 or self.step % self.log_interval == 0:
                    log_str = (
                        f"Step {self.step}/{self.max_steps} | "
                        f"Loss: {reduced_metrics['total_loss']:.4f} "
                        f"(policy={reduced_metrics['policy_loss']:.4f}, kl={reduced_metrics['kl_loss']:.4f}) | "
                        f"Reward: {reduced_metrics['reward']:.4f} | "
                        f"Baseline: {reduced_metrics['baseline_reward']:.4f} | "
                        f"Adv: {reduced_metrics['advantage']:.4f} | "
                        f"ActLoss: {reduced_metrics['implied_action_loss']:.4f} "
                        f"(base={reduced_metrics['implied_baseline_action_loss']:.4f}) | "
                        f"VideoSteps: {reduced_metrics['video_steps']:.2f}/{reduced_metrics['baseline_video_steps']} | "
                        f"GradNorm: {reduced_metrics['grad_norm']:.2f} | "
                        f"LR: {reduced_metrics['lr']:.2e} | "
                        f"StepTime(max): {reduced_metrics['step_time']:.2f}s"
                    )
                    logger.info(log_str)

                if self.wandb is not None:
                    self.wandb.log(
                        {
                            'train/policy_loss': reduced_metrics['policy_loss'],
                            'train/kl_loss': reduced_metrics['kl_loss'],
                            'train/total_loss': reduced_metrics['total_loss'],
                            'train/reward': reduced_metrics['reward'],
                            'train/baseline_reward': reduced_metrics['baseline_reward'],
                            'train/advantage': reduced_metrics['advantage'],
                            'train/video_steps': reduced_metrics['video_steps'],
                            'train/implied_action_loss': reduced_metrics['implied_action_loss'],
                            'train/implied_baseline_action_loss': reduced_metrics['implied_baseline_action_loss'],
                            'train/implied_action_loss_gap': reduced_metrics['implied_action_loss_gap'],
                            'train/compute_penalty': reduced_metrics['compute_penalty'],
                            'train/baseline_compute_penalty': reduced_metrics['baseline_compute_penalty'],
                            'train/compute_penalty_gap': reduced_metrics['compute_penalty_gap'],
                            'train/grad_norm': reduced_metrics['grad_norm'],
                            'train/lr': reduced_metrics['lr'],
                            'train/step_time_max': reduced_metrics['step_time'],
                            'train/micro_step': self.micro_step,
                        },
                        step=self.step,
                    )

                if self.step % self.save_interval == 0:
                    checkpoint_path = self.save_checkpoint()
                    if self.monitor is not None:
                        self.monitor.note_checkpoint(
                            step=self.step,
                            checkpoint_path=checkpoint_path,
                            final=False,
                        )

        pbar.close()
        logger.info("Training completed!")

        # Save final checkpoint
        if self.config.rank == 0:
            final_path = self.save_checkpoint(final=True)
            if self.monitor is not None:
                self.monitor.note_checkpoint(
                    step=self.step,
                    checkpoint_path=final_path,
                    final=True,
                )
                self.monitor.mark_finished(step=self.step, max_steps=self.max_steps)

    def save_checkpoint(self, final=False):
        """Save checkpoint."""
        if final:
            ckpt_path = self.save_dir / "hazard_scheduler_final.pt"
        else:
            ckpt_path = self.save_dir / f"hazard_scheduler_step_{self.step}.pt"

        scheduler_head = self._unwrap_scheduler_head(self.scheduler_head)
        checkpoint = {
            'step': self.step,
            'scheduler_head_state_dict': scheduler_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': _to_plain_python(dict(self.config)),
        }

        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")
        return ckpt_path

    def load_checkpoint(self, ckpt_path):
        """Load checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        scheduler_head = self._unwrap_scheduler_head(self.scheduler_head)
        scheduler_head.load_state_dict(
            self._normalize_scheduler_state_dict(checkpoint['scheduler_head_state_dict'])
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        logger.info(f"Checkpoint loaded from {ckpt_path}, resuming from step {self.step}")
