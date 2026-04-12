# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""
Hazard-based rollout runtime for adaptive video denoising.

This module provides the runtime infrastructure for executing rollouts with
HazardScheduler, including video denoising, stop decisions, and handoff to the
action branch.
"""

from typing import Dict, List, Tuple, Optional

import torch
from einops import rearrange

from .scheduler import FlowMatchScheduler
from .utils import data_seq_to_patch, get_mesh_id
from .hazard_reward import RolloutResult
from ..modules.hazard_scheduler import HazardScheduler, HazardSchedulerHead


class HazardRolloutRunner:
    """
    Runtime for executing Hazard-based rollouts.

    This class manages the complete rollout process:
    1. Video denoising with adaptive stopping
    2. HazardScheduler decision-making
    3. Handoff to action branch
    4. Action generation
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
        self.frame_chunk_size = int(config.frame_chunk_size)
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
        else:
            self.K_min = int(getattr(hazard_config, 'K_min', 3))
            self.eta = float(getattr(hazard_config, 'eta', 0.5))
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

    def _prefill_clean_history_cache(
        self,
        clean_history_latents: torch.Tensor,
        clean_history_actions: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> None:
        """Populate cache with one clean history chunk using update_cache=2."""
        if clean_history_latents is None or clean_history_actions is None:
            return
        if clean_history_latents.shape[2] == 0 or clean_history_actions.shape[2] == 0:
            return

        latent_input = self._prepare_input(
            clean_history_latents,
            text_emb,
            0.0,
            action_mode=False,
            cond=None,
            frame_st_id=0,
        )
        action_input = self._prepare_input(
            clean_history_actions,
            text_emb,
            0.0,
            action_mode=True,
            cond=None,
            frame_st_id=0,
        )

        with torch.no_grad():
            self.transformer(
                latent_input,
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=False,
            )
            self.transformer(
                action_input,
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=True,
            )

    def rollout_single_sample(
        self,
        video_noise: torch.Tensor,
        action_noise: torch.Tensor,
        text_emb: torch.Tensor,
        clean_history_latents: Optional[torch.Tensor] = None,
        clean_history_actions: Optional[torch.Tensor] = None,
        latent_cond: Optional[torch.Tensor] = None,
        action_cond: Optional[torch.Tensor] = None,
        mode: str = 'train',
    ) -> RolloutResult:
        """
        Execute a single rollout with HazardScheduler.

        Args:
            video_noise: [B, C, F, H, W] initial video noise
            action_noise: [B, A, F, N, 1] initial action noise
            text_emb: [B, L, D] text embeddings
            latent_cond: [B, C, 1, H, W] optional conditioning frame
            action_cond: [B, A, 1, N, 1] optional conditioning action frame
            mode: 'train' (stochastic) or 'eval' (deterministic)

        Returns:
            RolloutResult containing the trajectory, predicted actions, and
            rollout metadata.
        """
        if video_noise.shape[0] != 1:
            raise ValueError(
                "HazardRolloutRunner V1 currently only supports batch_size=1 "
                f"(got {video_noise.shape[0]})."
            )

        # Initialize cache
        self._init_cache(video_noise, action_noise)
        self._prefill_clean_history_cache(clean_history_latents, clean_history_actions, text_emb)

        # Create HazardScheduler
        hazard_scheduler = HazardScheduler(
            scheduler_head=self.scheduler_head,
            K_max=self.K_max,
            K_min=self.K_min,
            eta=self.eta,
            mode=mode,
        )

        # Prepare conditioning
        chunk_frame_st_id = int(clean_history_latents.shape[2]) if clean_history_latents is not None else 0

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
                cond=(latent_cond if clean_history_latents is None else None),
                frame_st_id=chunk_frame_st_id,
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
                if latent_cond is not None:
                    latents[:, :, 0:1] = latent_cond[:, :, 0:1]

            # HazardScheduler decision (keep gradients for scheduler_head)
            # video_pooled comes from transformer with no_grad, but scheduler_head will recompute with gradients
            scheduler_result = hazard_scheduler.step(video_pooled.detach())

            # Record trajectory
            # CRITICAL: Keep log_prob with gradients for REINFORCE training
            trajectory.append({
                'step_idx': step_idx,
                'delta_H_k': scheduler_result['delta_H_k'],
                'h_k': scheduler_result['h_k'],
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
            self._refresh_video_cache_exact(
                latents,
                text_emb,
                latent_cond,
                stop_step,
                frame_st_id=chunk_frame_st_id,
            )

            # Generate actions from cache
            final_action = self._run_action_from_cache(
                action_noise,
                text_emb,
                action_cond=action_cond,
                frame_st_id=chunk_frame_st_id,
            )

        final_hazard = trajectory[-1]['H_cumulative'] if trajectory else 0.0
        if isinstance(final_hazard, torch.Tensor):
            final_hazard = float(final_hazard.detach().float().mean().item())
        final_stop_prob = trajectory[-1]['h_k'] if trajectory else 0.0
        if isinstance(final_stop_prob, torch.Tensor):
            final_stop_prob = float(final_stop_prob.detach().float().mean().item())

        return RolloutResult(
            trajectory=trajectory,
            final_action=final_action,
            video_steps=stop_step,
            stop_step=stop_step,
            final_hazard=float(final_hazard),
            final_stop_prob=float(final_stop_prob),
        )

    def _refresh_video_cache_exact(self, latents, text_emb, latent_cond, cutoff_idx, frame_st_id=0):
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
            frame_st_id=frame_st_id,
        )
        self.transformer(
            input_dict,
            update_cache=1,
            cache_name=self.cache_name,
            action_mode=False,
        )

    def _run_action_from_cache(self, action_noise, text_emb, action_cond=None, frame_st_id=0):
        """Generate actions using cached video features."""
        actions = action_noise.clone()

        for i, t in enumerate(self.action_timesteps_with_terminal):
            last_step = i == len(self.action_timesteps_with_terminal) - 1
            input_dict = self._prepare_input(
                actions,
                text_emb,
                t,
                action_mode=True,
                cond=action_cond,
                frame_st_id=frame_st_id,
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

            if action_cond is not None:
                actions[:, :, 0:1] = action_cond[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0
        return actions

    def rollout_fixed_K(
        self,
        video_noise: torch.Tensor,
        action_noise: torch.Tensor,
        text_emb: torch.Tensor,
        K: int,
        clean_history_latents: Optional[torch.Tensor] = None,
        clean_history_actions: Optional[torch.Tensor] = None,
        latent_cond: Optional[torch.Tensor] = None,
        action_cond: Optional[torch.Tensor] = None,
    ) -> RolloutResult:
        """
        Execute rollout with fixed K steps (for baseline comparison).

        Args:
            video_noise: [B, C, F, H, W] initial video noise
            action_noise: [B, A, F, N, 1] initial action noise
            text_emb: [B, L, D] text embeddings
            K: fixed number of video steps
            latent_cond: [B, C, 1, H, W] optional conditioning frame
            action_cond: [B, A, 1, N, 1] optional conditioning action frame

        Returns:
            RolloutResult containing the fixed-K rollout outputs.
        """
        if video_noise.shape[0] != 1:
            raise ValueError(
                "HazardRolloutRunner V1 currently only supports batch_size=1 "
                f"(got {video_noise.shape[0]})."
            )
        K = max(0, min(int(K), self.num_video_steps))

        # Initialize cache
        self._init_cache(video_noise, action_noise)
        self._prefill_clean_history_cache(clean_history_latents, clean_history_actions, text_emb)

        # Prepare conditioning
        chunk_frame_st_id = int(clean_history_latents.shape[2]) if clean_history_latents is not None else 0

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
                    cond=(latent_cond if clean_history_latents is None else None),
                    frame_st_id=chunk_frame_st_id,
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
                if latent_cond is not None:
                    latents[:, :, 0:1] = latent_cond[:, :, 0:1]

            # Refresh cache and generate actions
            self._refresh_video_cache_exact(
                latents,
                text_emb,
                latent_cond,
                K,
                frame_st_id=chunk_frame_st_id,
            )
            final_action = self._run_action_from_cache(
                action_noise,
                text_emb,
                action_cond=action_cond,
                frame_st_id=chunk_frame_st_id,
            )

        return RolloutResult(
            trajectory=[],
            final_action=final_action,
            video_steps=K,
            stop_step=K,
            final_hazard=None,
            final_stop_prob=None,
        )
