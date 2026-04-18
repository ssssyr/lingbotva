# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wan_va.modules.hazard_scheduler import HazardJumpSchedulerHead
from wan_va.utils.hazard_runtime import HazardRolloutRunner


class RecordingTransformer:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def clear_cache(self, *_args, **_kwargs):
        return None

    def create_empty_cache(self, *_args, **_kwargs):
        return None

    def __call__(
        self,
        input_dict,
        update_cache,
        cache_name,
        action_mode,
        return_video_features=False,
    ):
        noisy_latents = input_dict["noisy_latents"]
        self.calls.append(
            {
                "action_mode": bool(action_mode),
                "update_cache": int(update_cache),
                "cache_name": cache_name,
                "timesteps": input_dict["timesteps"].detach().clone(),
                "noisy_latents": noisy_latents.detach().clone(),
            }
        )

        if action_mode:
            seq = torch.zeros(
                noisy_latents.shape[0],
                noisy_latents.shape[2] * noisy_latents.shape[3],
                noisy_latents.shape[1],
                dtype=noisy_latents.dtype,
                device=noisy_latents.device,
            )
        else:
            seq = torch.zeros(
                noisy_latents.shape[0],
                noisy_latents.shape[2] * noisy_latents.shape[3] * noisy_latents.shape[4],
                noisy_latents.shape[1],
                dtype=noisy_latents.dtype,
                device=noisy_latents.device,
            )

        if return_video_features:
            pooled = torch.full(
                (noisy_latents.shape[0], 4),
                0.1,
                dtype=noisy_latents.dtype,
                device=noisy_latents.device,
            )
            return seq, pooled
        return seq


def make_config(sigma_min=0.01):
    return SimpleNamespace(
        patch_size=(1, 1, 1),
        frame_chunk_size=2,
        action_dim=2,
        action_per_frame=3,
        num_inference_steps=4,
        action_num_inference_steps=2,
        snr_shift=1.0,
        action_snr_shift=1.0,
        attn_window=8,
        used_action_channel_ids=[0, 1],
        norm_stat={"q01": [0.0, 0.0], "q99": [1.0, 1.0]},
        model=SimpleNamespace(hazard_K_max=4),
        hazard=SimpleNamespace(
            K_min=1,
            eta=0.99,
            lambda_cost=0.1,
            policy_variant="stop_jump_v2",
            sigma_min=sigma_min,
            jump_logprob_scale=0.3,
        ),
    )


def make_runner(sigma_min=0.01):
    transformer = RecordingTransformer()
    config = make_config(sigma_min=sigma_min)
    head = HazardJumpSchedulerHead(feature_dim=4, hidden_dim=8)
    runner = HazardRolloutRunner(
        transformer=transformer,
        scheduler_head=head,
        config=config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return runner, transformer


def test_v2_rollout_records_sigma_and_jump_histories():
    runner, _transformer = make_runner()
    result = runner.rollout_single_sample(
        video_noise=torch.randn(1, 1, 2, 1, 1),
        action_noise=torch.randn(1, 2, 2, 3, 1),
        text_emb=torch.zeros(1, 4, 4),
        mode="eval",
    )

    assert result.executed_video_steps >= 1
    assert len(result.trajectory) >= result.executed_video_steps
    assert len(result.sigma_cur_history) == len(result.trajectory)
    assert len(result.sigma_next_history) == len(result.trajectory)
    assert len(result.jump_ratio_history) == len(result.trajectory)
    assert len(result.jump_distance_history) == len(result.trajectory)
    actual_updates = sum(1 for sigma_next in result.sigma_next_history if sigma_next is not None)
    assert result.executed_video_steps == actual_updates
    assert result.equivalent_fixed_steps >= 1.0
    assert result.equivalent_fixed_steps <= runner.num_video_steps
    assert result.terminal_sigma is not None
    assert result.final_action.shape == (1, 2, 2, 3, 1)


def test_v2_rollout_sigma_min_forces_stop_without_jump():
    runner, _transformer = make_runner(sigma_min=2.0)
    result = runner.rollout_single_sample(
        video_noise=torch.randn(1, 1, 2, 1, 1),
        action_noise=torch.randn(1, 2, 2, 3, 1),
        text_emb=torch.zeros(1, 4, 4),
        mode="eval",
    )

    assert result.executed_video_steps == 0
    assert result.trajectory[0]["forced_stop"] is True
    assert result.trajectory[0]["sigma_next"] is None
    assert result.jump_ratio_history == [None]
    assert result.jump_distance_history == [None]


def test_v2_rollout_reaches_terminal_sigma_zero_at_k_max():
    runner, _transformer = make_runner(sigma_min=0.0)
    result = runner.rollout_single_sample(
        video_noise=torch.randn(1, 1, 2, 1, 1),
        action_noise=torch.randn(1, 2, 2, 3, 1),
        text_emb=torch.zeros(1, 4, 4),
        mode="eval",
    )

    assert result.executed_video_steps == runner.K_max
    assert result.terminal_sigma == 0.0
    assert result.equivalent_fixed_steps == runner.num_video_steps
    assert result.trajectory[-1]["forced_terminal"] is True
    assert result.sigma_next_history[-1] == 0.0


def test_v2_rollout_keeps_first_chunk_conditioning():
    runner, transformer = make_runner()
    latent_cond = torch.full((1, 1, 1, 1, 1), 9.0)
    action_cond = torch.zeros((1, 2, 1, 3, 1))
    result = runner.rollout_single_sample(
        video_noise=torch.full((1, 1, 2, 1, 1), 7.0),
        action_noise=torch.full((1, 2, 2, 3, 1), 5.0),
        text_emb=torch.zeros((1, 4, 4)),
        latent_cond=latent_cond,
        action_cond=action_cond,
        mode="eval",
    )

    video_calls = [c for c in transformer.calls if not c["action_mode"] and c["update_cache"] == 0]
    action_calls = [c for c in transformer.calls if c["action_mode"] and c["update_cache"] == 0]

    assert torch.allclose(video_calls[0]["noisy_latents"][:, :, 0:1], latent_cond)
    assert torch.allclose(action_calls[0]["noisy_latents"][:, :, 0:1], action_cond)
    assert torch.allclose(result.final_action[:, :, 0:1], action_cond)
