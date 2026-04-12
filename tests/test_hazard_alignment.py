# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wan_va.modules.hazard_scheduler import HazardSchedulerHead
from wan_va.utils.hazard_rollout_inputs import prepare_chunked_rollout_inputs
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
            pooled = torch.zeros(
                noisy_latents.shape[0],
                4,
                dtype=noisy_latents.dtype,
                device=noisy_latents.device,
            )
            return seq, pooled
        return seq


def make_config():
    return SimpleNamespace(
        patch_size=(1, 1, 1),
        frame_chunk_size=2,
        action_dim=2,
        action_per_frame=3,
        num_inference_steps=2,
        action_num_inference_steps=2,
        snr_shift=1.0,
        action_snr_shift=1.0,
        attn_window=8,
        used_action_channel_ids=[0, 1],
        norm_stat={"q01": [0.0, 0.0], "q99": [1.0, 1.0]},
        model=SimpleNamespace(hazard_K_max=2),
        hazard=SimpleNamespace(K_min=1, eta=0.5, lambda_cost=0.1),
    )


def make_runner():
    transformer = RecordingTransformer()
    config = make_config()
    scheduler_head = HazardSchedulerHead(feature_dim=4, hidden_dim=4)
    runner = HazardRolloutRunner(
        transformer=transformer,
        scheduler_head=scheduler_head,
        config=config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return runner, transformer


def test_rollout_without_history_keeps_first_chunk_conditioning():
    runner, transformer = make_runner()
    video_noise = torch.full((1, 1, 2, 1, 1), 7.0)
    action_noise = torch.full((1, 2, 2, 3, 1), 5.0)
    text_emb = torch.zeros((1, 4, 4))
    gt_action = torch.zeros_like(action_noise)

    latent_cond = torch.full((1, 1, 1, 1, 1), 9.0)
    action_cond = torch.zeros((1, 2, 1, 3, 1))

    result = runner.rollout_fixed_K(
        video_noise=video_noise,
        action_noise=action_noise,
        text_emb=text_emb,
        K=1,
        latent_cond=latent_cond,
        action_cond=action_cond,
    )
    final_action = result.final_action

    video_calls = [c for c in transformer.calls if not c["action_mode"] and c["update_cache"] == 0]
    action_calls = [c for c in transformer.calls if c["action_mode"] and c["update_cache"] == 0]

    assert torch.allclose(video_calls[0]["noisy_latents"][:, :, 0:1], latent_cond)
    assert torch.allclose(action_calls[0]["noisy_latents"][:, :, 0:1], action_cond)
    assert torch.allclose(final_action[:, :, 0:1], action_cond)


def test_rollout_with_history_skips_first_frame_conditioning():
    runner, transformer = make_runner()
    video_noise = torch.full((1, 1, 2, 1, 1), 7.0)
    action_noise = torch.full((1, 2, 2, 3, 1), 5.0)
    text_emb = torch.zeros((1, 4, 4))
    gt_action = torch.zeros_like(action_noise)
    clean_history_latents = torch.ones((1, 1, 2, 1, 1))
    clean_history_actions = torch.ones((1, 2, 2, 3, 1))

    result = runner.rollout_fixed_K(
        video_noise=video_noise,
        action_noise=action_noise,
        text_emb=text_emb,
        K=1,
        clean_history_latents=clean_history_latents,
        clean_history_actions=clean_history_actions,
        latent_cond=None,
        action_cond=None,
    )
    final_action = result.final_action

    video_calls = [c for c in transformer.calls if not c["action_mode"] and c["update_cache"] == 0]
    action_calls = [c for c in transformer.calls if c["action_mode"] and c["update_cache"] == 0]

    assert torch.allclose(video_calls[0]["noisy_latents"][:, :, 0:1], video_noise[:, :, 0:1])
    assert torch.allclose(action_calls[0]["noisy_latents"][:, :, 0:1], action_noise[:, :, 0:1])
    assert torch.allclose(final_action[:, :, 0:1], action_noise[:, :, 0:1])


def test_rollout_fixed_k_returns_metadata():
    runner, _ = make_runner()
    video_noise = torch.zeros((1, 1, 2, 1, 1), dtype=torch.float32)
    action_noise = torch.zeros((1, 2, 2, 3, 1), dtype=torch.float32)
    text_emb = torch.zeros((1, 4, 4), dtype=torch.float32)

    result = runner.rollout_fixed_K(
        video_noise=video_noise,
        action_noise=action_noise,
        text_emb=text_emb,
        K=2,
    )

    assert result.video_steps == 2
    assert result.stop_step == 2
    assert result.final_action.shape == action_noise.shape


def test_prepare_rollout_inputs_for_eval_prefills_one_history_chunk():
    batch = {
        "latents": torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1, 1),
        "actions": torch.arange(24, dtype=torch.float32).view(1, 2, 4, 3, 1),
        "actions_mask": torch.ones(1, 2, 4, 3, 1, dtype=torch.bool),
        "text_emb": torch.zeros(1, 4, 4),
        "local_start_frame": torch.tensor([0]),
        "local_end_frame": torch.tensor([4]),
    }

    rollout_inputs = prepare_chunked_rollout_inputs(
        batch=batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        frame_chunk_size=2,
        sample_history=False,
    )

    assert rollout_inputs["clean_history_latents"] is not None
    assert rollout_inputs["clean_history_latents"].shape[2] == 2
    assert rollout_inputs["history_start_frame"] == 0
    assert rollout_inputs["target_start_frame"] == 2
    assert rollout_inputs["latent_cond"] is None
    assert rollout_inputs["action_cond"] is None
