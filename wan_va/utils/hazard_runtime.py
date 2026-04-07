# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""
Hazard-based rollout runtime for adaptive video denoising.

This module provides the runtime infrastructure for executing rollouts with
HazardScheduler, including video denoising, stop decisions, handoff to action
branch, and reward computation.
"""

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from .scheduler import FlowMatchScheduler
from .utils import data_seq_to_patch, get_mesh_id
from ..modules.hazard_scheduler import HazardScheduler, HazardSchedulerHead


class HazardRolloutRunner:
    """
    Runtime for executing Hazard-based rollouts.

    This class manages the complete rollout process:
    1. Video denoising with adaptive stopping
    2. HazardScheduler decision-making
    3. Handoff to action branch
    4. Action generation
    5. Reward computation
    """

    def __init__(
        self,
        transformer,
        scheduler_head: HazardSchedulerHead,
        config,
        device,
        dtype,
        cache_name="pos",
    ):
        """
        Args:
            transformer: WanTransformer3DModel instance
            scheduler_head: HazardSchedulerHead module
            config: Configuration object with model/training parameters
            device: torch device
            dtype: torch dtype
            cache_name: Name for transformer cache
        """
        self.transformer = transformer
        self.scheduler_head = scheduler_head
        self.config = config
        self.device = device
        self.dtype = dtype
        self.cache_name = cache_name

        # Model configuration
        self.patch_size = tuple(config.patch_size)
        self.action_dim = int(config.action_dim)
        self.action_per_frame = int(config.action_per_frame)
        self.num_video_steps = int(config.num_inference_steps)
        self.num_action_steps = int(config.action_num_inference_steps)

        # Schedulers
        self.video_scheduler = FlowMatchScheduler(
            shift=float(config.snr_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.video_scheduler.set_timesteps(self.num_video_steps)
        self.video_timesteps = self.video_scheduler.timesteps.clone()

        self.action_scheduler = FlowMatchScheduler(
            shift=float(config.action_snr_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler.set_timesteps(self.num_action_steps)
        action_timesteps = self.action_scheduler.timesteps
        self.action_timesteps_with_terminal = torch.nn.functional.pad(
            action_timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )

        # Action mask
        self.action_mask = torch.zeros([self.action_dim], dtype=torch.bool, device=self.device)
        self.action_mask[config.used_action_channel_ids] = True

        # Normalization stats for denormalized L1
        self.q01 = torch.tensor(config.norm_stat["q01"], dtype=torch.float32, device=self.device).view(
            1, -1, 1, 1, 1
        )
        self.q99 = torch.tensor(config.norm_stat["q99"], dtype=torch.float32, device=self.device).view(
            1, -1, 1, 1, 1
        )

        # Hazard scheduler configuration
        model_config = getattr(config, 'model', None)
        configured_k_max = getattr(model_config, 'hazard_K_max', None)
        if configured_k_max is None:
            configured_k_max = getattr(config, 'hazard_K_max', None)
        if configured_k_max is None:
            self.K_max = self.num_video_steps
        else:
            self.K_max = max(1, min(int(configured_k_max), self.num_video_steps))

        # Read from nested config: hazard.*
        hazard_config = getattr(config, 'hazard', None)
        if hazard_config is None:
            # Fallback to top-level attributes
            self.K_min = int(getattr(config, 'hazard_k_min', 3))
            self.eta = float(getattr(config, 'hazard_eta', 0.5))
            self.lambda_cost = float(getattr(config, 'lambda_cost', 0.1))
        else:
            self.K_min = int(getattr(hazard_config, 'K_min', 3))
            self.eta = float(getattr(hazard_config, 'eta', 0.5))
            self.lambda_cost = float(getattr(hazard_config, 'lambda_cost', 0.1))
        self.K_min = max(0, min(int(self.K_min), self.K_max - 1))

        # Set transformer to eval mode
        self.transformer.eval()

    def _init_cache(self, latents, actions):
        """Initialize transformer cache."""
        patch_f, patch_h, patch_w = self.patch_size
        _, _, f_lat, h_lat, w_lat = latents.shape
        _, _, f_act, n_act, _ = actions.shape

        latent_token_per_chunk = (f_lat * h_lat * w_lat) // (patch_f * patch_h * patch_w)
        action_token_per_chunk = f_act * n_act

        self.transformer.clear_cache(self.cache_name)
        self.transformer.create_empty_cache(
            self.cache_name,
            int(self.config.attn_window),
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=self.dtype,
            device=self.device,
            batch_size=latents.shape[0],
        )

    def _prepare_input(
        self,
        noisy_latents,
        text_emb,
        timestep,
        action_mode,
        cond,
        frame_st_id=0,
    ):
        """Prepare input dict for transformer forward pass."""
        if isinstance(timestep, torch.Tensor):
            timestep_value = float(timestep.item())
        else:
            timestep_value = float(timestep)

        if action_mode:
            grid_id = get_mesh_id(
                noisy_latents.shape[-3],
                noisy_latents.shape[-2],
                noisy_latents.shape[-1],
                1,
                1,
                frame_st_id,
                action=True,
            ).to(self.device)
        else:
            grid_id = get_mesh_id(
                noisy_latents.shape[-3] // self.patch_size[0],
                noisy_latents.shape[-2] // self.patch_size[1],
                noisy_latents.shape[-1] // self.patch_size[2],
                0,
                1,
                frame_st_id,
            ).to(self.device)

        timesteps = torch.ones([noisy_latents.shape[2]], dtype=torch.float32, device=self.device)
        timesteps = timesteps * timestep_value

        model_input = {
            "noisy_latents": noisy_latents,
            "timesteps": timesteps[None],
            "grid_id": grid_id[None],
            "text_emb": text_emb,
        }

        if cond is not None:
            model_input["noisy_latents"] = model_input["noisy_latents"].clone()
            model_input["timesteps"] = model_input["timesteps"].clone()
            model_input["noisy_latents"][:, :, 0:1] = cond[:, :, 0:1]
            model_input["timesteps"][:, 0:1] *= 0

        if action_mode:
            model_input["noisy_latents"] = model_input["noisy_latents"].clone()
            model_input["noisy_latents"][:, ~self.action_mask] *= 0

        return model_input

    def rollout_single_sample(
        self,
        video_noise: torch.Tensor,
        action_noise: torch.Tensor,
        text_emb: torch.Tensor,
        gt_action: torch.Tensor,
        latent_cond: Optional[torch.Tensor] = None,
        mode: str = 'train',
    ) -> Tuple[List[Dict], torch.Tensor, float]:
        """
        Execute a single rollout with HazardScheduler.

        Args:
            video_noise: [B, C, F, H, W] initial video noise
            action_noise: [B, A, F, N, 1] initial action noise
            text_emb: [B, L, D] text embeddings
            gt_action: [B, A, F, N, 1] ground truth actions for reward
            latent_cond: [B, C, 1, H, W] optional conditioning frame
            mode: 'train' (stochastic) or 'eval' (deterministic)

        Returns:
            trajectory: List of dicts with step info (delta_H_k, h_k, log_prob, etc.)
            final_action: [B, A, F, N, 1] predicted actions
            reward: scalar reward value
        """
        if video_noise.shape[0] != 1:
            raise ValueError(
                "HazardRolloutRunner V1 currently only supports batch_size=1 "
                f"(got {video_noise.shape[0]})."
            )

        # Initialize cache
        self._init_cache(video_noise, action_noise)

        # Create HazardScheduler
        hazard_scheduler = HazardScheduler(
            scheduler_head=self.scheduler_head,
            K_max=self.K_max,
            K_min=self.K_min,
            eta=self.eta,
            mode=mode,
        )

        # Prepare conditioning
        if latent_cond is None:
            latent_cond = torch.zeros(
                [video_noise.shape[0], video_noise.shape[1], 1, video_noise.shape[3], video_noise.shape[4]],
                device=self.device,
                dtype=self.dtype,
            )

        # Video rollout with adaptive stopping
        latents = video_noise.clone()
        trajectory = []

        for step_idx in range(self.K_max):
            t = self.video_timesteps[step_idx]

            # Prepare input
            input_dict = self._prepare_input(
                latents,
                text_emb,
                t,
                action_mode=False,
                cond=latent_cond,
            )

            # Forward pass with feature extraction
            # Use no_grad for transformer to save memory, but scheduler_head will compute with gradients
            with torch.no_grad():
                video_noise_pred, video_pooled = self.transformer(
                    input_dict,
                    update_cache=0,
                    cache_name=self.cache_name,
                    action_mode=False,
                    return_video_features=True,
                )

            # Convert to patch format
            video_noise_pred = data_seq_to_patch(
                self.patch_size,
                video_noise_pred,
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
                batch_size=latents.shape[0],
            )

            # Update latents (no gradients needed for scheduler step)
            with torch.no_grad():
                latents = self.video_scheduler.step(
                    video_noise_pred,
                    t,
                    latents,
                    return_dict=False,
                )
                latents[:, :, 0:1] = latent_cond[:, :, 0:1]

            # HazardScheduler decision (keep gradients for scheduler_head)
            # video_pooled comes from transformer with no_grad, but scheduler_head will recompute with gradients
            scheduler_result = hazard_scheduler.step(video_pooled.detach())

            # Record trajectory
            # CRITICAL: Keep log_prob with gradients for REINFORCE training
            trajectory.append({
                'step_idx': step_idx,
                'delta_H_k': scheduler_result['delta_H_k'].detach(),
                'h_k': scheduler_result['h_k'].detach(),
                'log_prob': scheduler_result['log_prob'],  # Keep gradients!
                'should_stop': scheduler_result['should_stop'],
                'H_cumulative': scheduler_result['H_cumulative'],
            })

            # Check if should stop
            if scheduler_result['should_stop']:
                break

        # Refresh video cache at stop point
        stop_step = len(trajectory)
        with torch.no_grad():
            self._refresh_video_cache_exact(latents, text_emb, latent_cond, stop_step)

            # Generate actions from cache
            final_action = self._run_action_from_cache(action_noise, text_emb)

        # Compute reward (no gradients needed)
        reward = self._compute_reward(final_action, gt_action, stop_step)

        return trajectory, final_action, reward

    def _refresh_video_cache_exact(self, latents, text_emb, latent_cond, cutoff_idx):
        """Refresh video cache at the cutoff point."""
        if cutoff_idx < self.num_video_steps:
            refresh_t = self.video_timesteps[cutoff_idx]
        else:
            refresh_t = 0.0

        input_dict = self._prepare_input(
            latents,
            text_emb,
            refresh_t,
            action_mode=False,
            cond=latent_cond,
        )
        self.transformer(
            input_dict,
            update_cache=1,
            cache_name=self.cache_name,
            action_mode=False,
        )

    def _run_action_from_cache(self, action_noise, text_emb):
        """Generate actions using cached video features."""
        actions = action_noise.clone()
        action_cond = torch.zeros(
            [
                actions.shape[0],
                self.action_dim,
                1,
                self.action_per_frame,
                1,
            ],
            device=self.device,
            dtype=self.dtype,
        )

        for i, t in enumerate(self.action_timesteps_with_terminal):
            last_step = i == len(self.action_timesteps_with_terminal) - 1
            input_dict = self._prepare_input(
                actions,
                text_emb,
                t,
                action_mode=True,
                cond=action_cond,
            )
            action_noise_pred = self.transformer(
                input_dict,
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=True,
            )

            if not last_step:
                action_noise_pred = rearrange(
                    action_noise_pred,
                    "b (f n) c -> b c f n 1",
                    f=actions.shape[2],
                )
                actions = self.action_scheduler.step(
                    action_noise_pred,
                    t,
                    actions,
                    return_dict=False,
                )

            actions[:, :, 0:1] = action_cond[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0
        return actions

    def _compute_reward(self, pred_action, gt_action, video_steps):
        """
        Compute reward: R = -L_act - lambda_cost * video_steps

        Args:
            pred_action: [B, A, F, N, 1] predicted actions
            gt_action: [B, A, F, N, 1] ground truth actions
            video_steps: number of video denoising steps used

        Returns:
            reward: scalar reward value
        """
        # Compute action loss (MSE)
        L_act = F.mse_loss(pred_action.float(), gt_action.float())

        # Compute reward
        reward = -L_act.item() - self.lambda_cost * video_steps

        return reward

    def rollout_fixed_K(
        self,
        video_noise: torch.Tensor,
        action_noise: torch.Tensor,
        text_emb: torch.Tensor,
        gt_action: torch.Tensor,
        K: int,
        latent_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Dict], torch.Tensor, float]:
        """
        Execute rollout with fixed K steps (for baseline comparison).

        Args:
            video_noise: [B, C, F, H, W] initial video noise
            action_noise: [B, A, F, N, 1] initial action noise
            text_emb: [B, L, D] text embeddings
            gt_action: [B, A, F, N, 1] ground truth actions
            K: fixed number of video steps
            latent_cond: [B, C, 1, H, W] optional conditioning frame

        Returns:
            trajectory: Empty list (no scheduler decisions)
            final_action: [B, A, F, N, 1] predicted actions
            reward: scalar reward value
        """
        if video_noise.shape[0] != 1:
            raise ValueError(
                "HazardRolloutRunner V1 currently only supports batch_size=1 "
                f"(got {video_noise.shape[0]})."
            )

        # Initialize cache
        self._init_cache(video_noise, action_noise)

        # Prepare conditioning
        if latent_cond is None:
            latent_cond = torch.zeros(
                [video_noise.shape[0], video_noise.shape[1], 1, video_noise.shape[3], video_noise.shape[4]],
                device=self.device,
                dtype=self.dtype,
            )

        # Run video denoising for K steps (no gradients needed for baseline)
        latents = video_noise.clone()

        with torch.no_grad():
            for step_idx in range(K):
                t = self.video_timesteps[step_idx]

                input_dict = self._prepare_input(
                    latents,
                    text_emb,
                    t,
                    action_mode=False,
                    cond=latent_cond,
                )

                video_noise_pred = self.transformer(
                    input_dict,
                    update_cache=0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )

                video_noise_pred = data_seq_to_patch(
                    self.patch_size,
                    video_noise_pred,
                    latents.shape[2],
                    latents.shape[3],
                    latents.shape[4],
                    batch_size=latents.shape[0],
                )

                latents = self.video_scheduler.step(
                    video_noise_pred,
                    t,
                    latents,
                    return_dict=False,
                )
                latents[:, :, 0:1] = latent_cond[:, :, 0:1]

            # Refresh cache and generate actions
            self._refresh_video_cache_exact(latents, text_emb, latent_cond, K)
            final_action = self._run_action_from_cache(action_noise, text_emb)

        # Compute reward
        reward = self._compute_reward(final_action, gt_action, K)

        return [], final_action, reward
