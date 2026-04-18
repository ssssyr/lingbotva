# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""
REINFORCE trainer for HazardScheduler.

This module implements the policy gradient training loop for learning
adaptive video denoising schedules using the REINFORCE algorithm.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ..modules.hazard_scheduler import HazardJumpSchedulerHead, HazardSchedulerHead
from .hazard_rollout_inputs import prepare_chunked_rollout_inputs
from .hazard_runtime import HazardRolloutRunner
from .hazard_reward import DualAnchorRewardEvaluator
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
    3. Execute cheap/high-quality anchor rollouts
    4. Compute path reward from semantic sequence + trend quality terms
    5. Compute advantage against an EMA reward baseline
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
            self.reward_baseline_mode = str(getattr(config, 'reward_adv_baseline', 'ema')).lower()
            self.reward_ema_decay = float(getattr(config, 'reward_ema_decay', 0.95))
        else:
            # Read from hazard config section
            self.learning_rate = float(getattr(hazard_config, 'learning_rate', 1e-4))
            self.lambda_cost = float(getattr(hazard_config, 'lambda_cost', 0.1))
            self.lambda_kl = float(getattr(hazard_config, 'lambda_kl', 0.01))
            self.baseline_K = int(getattr(hazard_config, 'baseline_K', 10))
            self.max_steps = int(getattr(config.training, 'num_steps', 10000))
            self.save_interval = int(getattr(hazard_config, 'save_interval', 1000))
            self.log_interval = int(getattr(hazard_config, 'log_interval', 10))
            self.reward_baseline_mode = str(getattr(hazard_config, 'reward_adv_baseline', 'ema')).lower()
            self.reward_ema_decay = float(getattr(hazard_config, 'reward_ema_decay', 0.95))
        self.policy_variant = str(getattr(hazard_config, "policy_variant", "stop_only_v1") if hazard_config is not None else getattr(config, "hazard_policy_variant", "stop_only_v1")).lower()
        self.optimizer_mode = str(getattr(hazard_config, "optimizer_mode", "reinforce") if hazard_config is not None else "reinforce").lower()
        self.ppo_num_epochs = int(getattr(hazard_config, "ppo_num_epochs", 4) if hazard_config is not None else 4)
        self.cliprange = float(getattr(hazard_config, "cliprange", 0.2) if hazard_config is not None else 0.2)
        self.jump_logprob_scale = float(getattr(hazard_config, "jump_logprob_scale", 1.0) if hazard_config is not None else 1.0)
        self.advantage_normalize = bool(getattr(hazard_config, "advantage_normalize", False) if hazard_config is not None else False)
        self.jump_ref_concentration = float(getattr(hazard_config, "jump_ref_concentration", 20.0) if hazard_config is not None else 20.0)

        # Read gradient_accumulation_steps from training config
        self.gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))
        if self.optimizer_mode == "ppo" and self.gradient_accumulation_steps != 1:
            logger.warning(
                "optimizer_mode=ppo currently runs one optimizer step per sampled rollout; "
                "overriding gradient_accumulation_steps from %s to 1",
                self.gradient_accumulation_steps,
            )
            self.gradient_accumulation_steps = 1

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
        self.reward_evaluator = DualAnchorRewardEvaluator(
            config=config,
            device=device,
        )
        if hazard_config is None:
            self.kl_reference_K = int(getattr(config, 'kl_reference_K', self.reward_evaluator.anchor_hi_k))
        else:
            self.kl_reference_K = int(getattr(hazard_config, 'kl_reference_K', self.reward_evaluator.anchor_hi_k))
        self.kl_reference_K = max(1, min(int(self.kl_reference_K), int(config.num_inference_steps)))

        # Training state
        self.step = 0
        self.micro_step = 0
        self.train_loader_iter = None
        self.reward_baseline_value: Optional[float] = None

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
        logger.info(f"  Reward anchors: lo={self.reward_evaluator.anchor_lo_k} hi={self.reward_evaluator.anchor_hi_k}")
        logger.info(f"  KL reference K: {self.kl_reference_K}")
        logger.info(f"  Reward baseline mode: {self.reward_baseline_mode}")
        logger.info(f"  Policy variant: {self.policy_variant}")
        logger.info(f"  Optimizer mode: {self.optimizer_mode}")
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

        For the reference policy, steps before `kl_reference_K` almost surely continue,
        while the baseline step and later steps almost surely stop. This keeps the
        learned scheduler close to a stable fixed-K rollout during early training.
        """
        if len(trajectory) == 0:
            return torch.tensor(0.0, device=self.device)

        eps = 1e-6
        kl_sum = torch.tensor(0.0, device=self.device)
        reference_stop_idx = max(int(self.kl_reference_K) - 1, 0)

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
            key for key, value in metrics.items()
            if isinstance(value, (int, float)) and key not in {"should_step", "step_time"}
        ]
        if mean_keys:
            values = torch.tensor(
                [float(metrics[key]) for key in mean_keys],
                dtype=torch.float32,
                device=self.device,
            )
            dist.all_reduce(values, op=dist.ReduceOp.AVG)
            for key, value in zip(mean_keys, values.tolist()):
                reduced[key] = value

        if "step_time" in metrics:
            step_time = torch.tensor([float(metrics["step_time"])], dtype=torch.float32, device=self.device)
            dist.all_reduce(step_time, op=dist.ReduceOp.MAX)
            reduced["step_time"] = step_time.item()
        return reduced

    def _get_advantage_baseline(self, reward: float) -> float:
        if self.reward_baseline_mode == "zero":
            return 0.0
        if self.reward_baseline_mode != "ema":
            raise ValueError(f"Unsupported reward_adv_baseline={self.reward_baseline_mode!r}")

        if self.reward_baseline_value is None:
            baseline = float(reward)
            self.reward_baseline_value = float(reward)
            return baseline

        baseline = float(self.reward_baseline_value)
        self.reward_baseline_value = (
            self.reward_ema_decay * self.reward_baseline_value
            + (1.0 - self.reward_ema_decay) * float(reward)
        )
        return baseline

    def _normalize_advantage(self, advantage: float) -> float:
        # With batch_size=1 this is a no-op, but keep the hook for future
        # multi-rollout / RLOO variants.
        return float(advantage)

    def _trajectory_old_logprob(self, trajectory) -> torch.Tensor:
        old_logprob = torch.tensor(0.0, device=self.device)
        for step_info in trajectory:
            if step_info.get("old_logprob", None) is not None:
                old_logprob = old_logprob + step_info["old_logprob"].to(device=self.device)
        return old_logprob

    def _recompute_path_logprob(self, trajectory) -> torch.Tensor:
        if not isinstance(self.scheduler_head, HazardJumpSchedulerHead):
            raise TypeError("Path logprob recomputation is only supported for HazardJumpSchedulerHead")
        logprob = torch.tensor(0.0, device=self.device)
        for step_info in trajectory:
            new_step_logprob = self.scheduler_head.recompute_step_logprob(
                video_feature=step_info["video_feature"].to(device=self.device),
                sigma_cur=float(step_info["sigma_cur"]),
                stop_action=bool(step_info["stop_action"]),
                sigma_next=step_info.get("sigma_next", None),
                forced_stop=bool(step_info.get("forced_stop", False)),
                forced_continue=bool(step_info.get("forced_continue", False)),
                jump_logprob_scale=self.jump_logprob_scale,
            )
            if new_step_logprob is not None:
                logprob = logprob + new_step_logprob
        return logprob

    def _reference_jump_beta(self, sigma_cur: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
        sigmas = self.rollout_runner.video_scheduler.sigmas.float()
        sigma_cur_tensor = torch.tensor(float(sigma_cur), dtype=torch.float32)
        idx = int(torch.argmin((sigmas - sigma_cur_tensor).abs()).item())
        idx = max(0, min(idx, len(sigmas) - 2))
        ref_ratio = torch.clamp(sigmas[idx + 1] / torch.clamp(sigmas[idx], min=1e-6), 1e-3, 1.0 - 1e-3)
        concentration = float(self.jump_ref_concentration)
        alpha_ref = ref_ratio * (concentration - 2.0) + 1.0
        beta_ref = (1.0 - ref_ratio) * (concentration - 2.0) + 1.0
        return (
            torch.tensor([alpha_ref], dtype=torch.float32, device=device),
            torch.tensor([beta_ref], dtype=torch.float32, device=device),
        )

    def _compute_kl_regularization_v2(self, trajectory) -> torch.Tensor:
        if len(trajectory) == 0:
            return torch.tensor(0.0, device=self.device)

        eps = 1e-6
        stop_kl_sum = torch.tensor(0.0, device=self.device)
        jump_kl_sum = torch.tensor(0.0, device=self.device)
        jump_terms = 0
        stop_terms = 0
        reference_stop_idx = max(int(self.kl_reference_K) - 1, 0)

        for step_info in trajectory:
            feature = step_info["video_feature"].to(device=self.device)
            sigma_cur = float(step_info["sigma_cur"])
            outputs = self.scheduler_head(feature, sigma_cur=sigma_cur)
            if not step_info.get("forced_stop", False) and not step_info.get("forced_continue", False):
                h_k = torch.clamp(outputs.h_k, 1e-6, 1.0 - 1e-6)
                reference_prob = eps if int(step_info["step_idx"]) < reference_stop_idx else 1.0 - eps
                h_ref_k = torch.full_like(h_k, reference_prob)
                stop_kl = h_k * torch.log(h_k / h_ref_k) + (1 - h_k) * torch.log((1 - h_k) / (1 - h_ref_k))
                stop_kl_sum = stop_kl_sum + stop_kl.mean()
                stop_terms += 1

            if step_info.get("sigma_next", None) is not None:
                alpha_ref, beta_ref = self._reference_jump_beta(sigma_cur, feature.device)
                jump_kl = torch.distributions.kl_divergence(
                    torch.distributions.Beta(outputs.alpha_k.squeeze(-1), outputs.beta_k.squeeze(-1)),
                    torch.distributions.Beta(alpha_ref, beta_ref),
                )
                jump_kl_sum = jump_kl_sum + jump_kl.mean()
                jump_terms += 1

        stop_kl_mean = stop_kl_sum / max(stop_terms, 1)
        jump_kl_mean = jump_kl_sum / max(jump_terms, 1)
        return stop_kl_mean + jump_kl_mean

    def _sync_scheduler_gradients(self) -> None:
        if not dist.is_initialized():
            return

        world_size = float(dist.get_world_size())
        for param in self.scheduler_head.parameters():
            if not param.requires_grad:
                continue

            # IMPORTANT: all ranks must execute the exact same collective sequence.
            # Some scheduler parameters can legitimately receive no gradient on a
            # given rank for a given rollout. If we `continue` here on only a
            # subset of ranks, NCCL will see different tensor sizes / different
            # collective order and eventually deadlock / timeout.
            if param.grad is None:
                param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)

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
        current_result = self.rollout_runner.rollout_single_sample(
            video_noise=rollout_inputs['video_noise'],
            action_noise=rollout_inputs['action_noise'],
            text_emb=rollout_inputs['text_emb'],
            clean_history_latents=rollout_inputs['clean_history_latents'],
            clean_history_actions=rollout_inputs['clean_history_actions'],
            latent_cond=rollout_inputs['latent_cond'],
            action_cond=rollout_inputs['action_cond'],
            mode='train',
        )

        trajectory = current_result.trajectory

        # Execute anchor rollouts (fixed K) - no gradients needed
        with torch.no_grad():
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
            reward_breakdown = self.reward_evaluator.evaluate(
                current=current_result,
                anchor_lo=anchor_lo_result,
                anchor_hi=anchor_hi_result,
                gt_action=rollout_inputs['gt_action'],
                gt_action_mask=rollout_inputs['gt_action_mask'],
            )

        reward = reward_breakdown.reward
        baseline_reward = self._get_advantage_baseline(reward)
        advantage = reward - baseline_reward

        if self.advantage_normalize:
            advantage = self._normalize_advantage(advantage)

        grad_norm = 0.0
        lr = self.optimizer.param_groups[0]["lr"]
        approxkl = 0.0
        clipfrac = 0.0
        ratio_mean = 1.0
        policy_loss_value = 0.0
        kl_loss_value = 0.0
        total_loss_value = 0.0

        if self.optimizer_mode == "ppo" and isinstance(self.scheduler_head, HazardJumpSchedulerHead):
            should_step = True
            old_logprob = self._trajectory_old_logprob(trajectory).detach()
            self.optimizer.zero_grad(set_to_none=True)
            policy_loss_terms = []
            kl_loss_terms = []
            approxkl_terms = []
            clipfrac_terms = []
            ratio_terms = []
            grad_norm_terms = []

            for _ in range(self.ppo_num_epochs):
                new_logprob = self._recompute_path_logprob(trajectory)
                logprobs_diff = new_logprob - old_logprob
                ratio = torch.exp(logprobs_diff)
                unclipped = -float(advantage) * ratio
                clipped = -float(advantage) * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
                policy_loss = torch.max(unclipped, clipped)
                kl_loss = self._compute_kl_regularization_v2(trajectory)
                total_loss = policy_loss + self.lambda_kl * kl_loss

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self._sync_scheduler_gradients()
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.scheduler_head.parameters(), max_norm=1.0)
                grad_norm_epoch = float(grad_norm_tensor.item())
                if torch.isfinite(torch.tensor(grad_norm_epoch)):
                    self.optimizer.step()
                else:
                    logger.warning(
                        "Non-finite grad_norm detected at step=%s micro_step=%s during PPO; skipping optimizer step",
                        self.step,
                        self.micro_step,
                    )

                policy_loss_terms.append(float(policy_loss.item()))
                kl_loss_terms.append(float(kl_loss.item()))
                approxkl_terms.append(float((0.5 * (logprobs_diff ** 2)).item()))
                clipfrac_terms.append(float((clipped > unclipped).float().item()))
                ratio_terms.append(float(ratio.item()))
                grad_norm_terms.append(grad_norm_epoch)

            policy_loss_value = sum(policy_loss_terms) / max(len(policy_loss_terms), 1)
            kl_loss_value = sum(kl_loss_terms) / max(len(kl_loss_terms), 1)
            total_loss_value = (policy_loss_value + self.lambda_kl * kl_loss_value)
            approxkl = sum(approxkl_terms) / max(len(approxkl_terms), 1)
            clipfrac = sum(clipfrac_terms) / max(len(clipfrac_terms), 1)
            ratio_mean = sum(ratio_terms) / max(len(ratio_terms), 1)
            grad_norm = sum(grad_norm_terms) / max(len(grad_norm_terms), 1)
        else:
            # Compute policy gradient loss
            # REINFORCE: L = -sum(log_prob * advantage)
            policy_loss = torch.tensor(0.0, device=self.device)
            for step_info in trajectory:
                if step_info['log_prob'] is not None:
                    policy_loss = policy_loss + (-step_info['log_prob'] * float(advantage))

            policy_loss = policy_loss / max(len(trajectory), 1)
            if isinstance(self.scheduler_head, HazardJumpSchedulerHead):
                kl_loss = self._compute_kl_regularization_v2(trajectory)
            else:
                kl_loss = self._compute_kl_regularization(trajectory)

            total_loss = policy_loss + self.lambda_kl * kl_loss
            total_loss = total_loss / self.gradient_accumulation_steps
            total_loss.backward()

            should_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
            if should_step:
                self._sync_scheduler_gradients()
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.scheduler_head.parameters(), max_norm=1.0)
                grad_norm = float(grad_norm_tensor.item())
                if torch.isfinite(torch.tensor(grad_norm)):
                    self.optimizer.step()
                else:
                    logger.warning(
                        "Non-finite grad_norm detected at step=%s micro_step=%s; skipping optimizer step",
                        self.step,
                        self.micro_step,
                    )
                self.optimizer.zero_grad(set_to_none=True)
            policy_loss_value = float(policy_loss.item() * self.gradient_accumulation_steps)
            kl_loss_value = float(kl_loss.item())
            total_loss_value = float(total_loss.item() * self.gradient_accumulation_steps)

        # Metrics
        metrics = {
            'policy_loss': policy_loss_value,
            'kl_loss': kl_loss_value,
            'total_loss': total_loss_value,
            'reward': reward,
            'baseline_reward': baseline_reward,
            'advantage': advantage,
            'quality': reward_breakdown.quality,
            'cost': reward_breakdown.cost,
            'q_seq': reward_breakdown.q_seq,
            'q_delta': reward_breakdown.q_delta,
            'l_seq_cur': reward_breakdown.l_seq_cur,
            'l_seq_lo': reward_breakdown.l_seq_lo,
            'l_seq_hi': reward_breakdown.l_seq_hi,
            'l_delta_cur': reward_breakdown.l_delta_cur,
            'l_delta_lo': reward_breakdown.l_delta_lo,
            'l_delta_hi': reward_breakdown.l_delta_hi,
            'gap_seq': reward_breakdown.gap_seq,
            'gap_delta': reward_breakdown.gap_delta,
            'g_seq': reward_breakdown.g_seq,
            'g_delta': reward_breakdown.g_delta,
            'd_seq': reward_breakdown.d_seq,
            'd_delta': reward_breakdown.d_delta,
            'video_steps': current_result.executed_video_steps,
            'equivalent_fixed_steps': float(current_result.equivalent_fixed_steps or current_result.video_steps),
            'terminal_sigma': float(current_result.terminal_sigma) if current_result.terminal_sigma is not None else -1.0,
            'jump_distance_mean': float(sum(v for v in (current_result.jump_distance_history or []) if v is not None) / max(1, len([v for v in (current_result.jump_distance_history or []) if v is not None]))) if current_result.jump_distance_history else 0.0,
            'anchor_lo_steps': anchor_lo_result.video_steps,
            'anchor_hi_steps': anchor_hi_result.video_steps,
            'step_time': time.perf_counter() - step_t0,
            'grad_norm': grad_norm,
            'lr': lr,
            'should_step': should_step,
            'ema_baseline': float(self.reward_baseline_value if self.reward_baseline_value is not None else baseline_reward),
            'approxkl': approxkl,
            'clipfrac': clipfrac,
            'ratio_mean': ratio_mean,
        }

        return metrics

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
            reduced_metrics = self._reduce_metrics(metrics)

            if self.config.rank == 0:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{reduced_metrics['total_loss']:.4f}",
                    reward=f"{reduced_metrics['reward']:.4f}",
                    quality=f"{reduced_metrics['quality']:.4f}",
                    steps=f"{reduced_metrics['video_steps']:.2f}",
                    eq=f"{reduced_metrics.get('equivalent_fixed_steps', 0.0):.2f}",
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
                        f"Reward: {reduced_metrics['reward']:.4f} "
                        f"(quality={reduced_metrics['quality']:.4f}, cost={reduced_metrics['cost']:.4f}) | "
                        f"Baseline: {reduced_metrics['baseline_reward']:.4f} "
                        f"(ema={reduced_metrics['ema_baseline']:.4f}) | "
                        f"Adv: {reduced_metrics['advantage']:.4f} | "
                        f"Q(seq={reduced_metrics['q_seq']:.4f}, delta={reduced_metrics['q_delta']:.4f}) | "
                        f"Steps: cur={reduced_metrics['video_steps']:.2f} "
                        f"eq={reduced_metrics.get('equivalent_fixed_steps', 0.0):.2f} "
                        f"lo={reduced_metrics['anchor_lo_steps']:.2f} "
                        f"hi={reduced_metrics['anchor_hi_steps']:.2f} | "
                        f"Sigma: {reduced_metrics.get('terminal_sigma', -1.0):.4f} | "
                        f"JumpDist: {reduced_metrics.get('jump_distance_mean', 0.0):.4f} | "
                        f"PPO(r={reduced_metrics.get('ratio_mean', 1.0):.4f}, kl={reduced_metrics.get('approxkl', 0.0):.4f}, clip={reduced_metrics.get('clipfrac', 0.0):.4f}) | "
                        f"GradNorm: {reduced_metrics['grad_norm']:.2f} | "
                        f"LR: {reduced_metrics['lr']:.2e} | "
                        f"StepTime(max): {reduced_metrics['step_time']:.2f}s"
                    )
                    logger.info(log_str)

                if self.wandb is not None:
                    wandb_metrics = {
                        f"train/{key}": value
                        for key, value in reduced_metrics.items()
                        if isinstance(value, (int, float)) and key != "should_step"
                    }
                    wandb_metrics['train/micro_step'] = self.micro_step
                    self.wandb.log(wandb_metrics, step=self.step)

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
            'schema_version': 'hazard_scheduler_v2' if self.policy_variant == 'stop_jump_v2' else 'hazard_scheduler_v1',
            'policy_variant': self.policy_variant,
            'optimizer_mode': self.optimizer_mode,
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
