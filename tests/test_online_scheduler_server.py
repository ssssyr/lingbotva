# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wan_va.wan_va_server import VA_Server


class DummyScheduler:
    def __init__(self):
        self.calls = []

    def step(self, pred, t, latents, return_dict=False):
        self.calls.append(float(t))
        return latents + 1


def make_server_stub():
    server = VA_Server.__new__(VA_Server)
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.cache_name = "pos"
    server.use_cfg = False
    server.latent_height = 1
    server.latent_width = 1
    server.action_per_frame = 2
    server.job_config = SimpleNamespace(
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        num_inference_steps=25,
        fixed_video_steps=5,
        video_exec_step=-1,
        action_dim=2,
        action_per_frame=2,
        hazard_return_metadata=True,
        save_debug_artifacts=True,
    )
    server.scheduler = DummyScheduler()
    server._sync_device = lambda: None
    server._get_latent_cond = lambda frame_st_id: None
    server._repeat_input_for_cfg = lambda input_dict: input_dict
    server._apply_video_cfg = lambda video_noise_pred, frame_chunk_size: video_noise_pred
    server.action_mask = torch.ones(2, dtype=torch.bool)

    refresh_calls = []

    def refresh(latents, refresh_t, frame_st_id=0):
        refresh_calls.append((float(refresh_t), int(frame_st_id), latents.clone()))

    server._refresh_video_cache_exact = refresh
    server._prepare_latent_input = lambda latents, _action, t1, _t2, _cond1, _cond2, frame_st_id=0: {
        "latent_res_lst": {
            "noisy_latents": latents,
            "timesteps": torch.tensor([float(t1)], dtype=torch.float32),
            "grid_id": torch.zeros(1, 1, dtype=torch.long),
            "text_emb": torch.zeros(1, 1, 4, dtype=torch.float32),
        }
    }

    def transformer(input_dict, update_cache, cache_name, action_mode, return_video_features=False):
        noisy_latents = input_dict["noisy_latents"]
        if return_video_features:
            seq = torch.zeros_like(noisy_latents)
            pooled = torch.zeros(noisy_latents.shape[0], 4, dtype=noisy_latents.dtype)
            return seq, pooled
        return torch.zeros_like(noisy_latents)

    server.transformer = transformer
    server._test_refresh_calls = refresh_calls
    return server


def test_online_scheduler_mode_prefers_explicit_mode():
    server = make_server_stub()
    server.job_config.online_scheduler_mode = "hazard"
    server.job_config.enable_hazard_scheduler_runtime = False

    assert server._get_online_scheduler_mode() == "hazard"


def test_online_scheduler_mode_falls_back_to_legacy_flag():
    server = make_server_stub()
    server.job_config.online_scheduler_mode = None
    server.job_config.enable_hazard_scheduler_runtime = True

    assert server._get_online_scheduler_mode() == "hazard"


def test_get_fixed_video_steps_respects_explicit_value():
    server = make_server_stub()
    server.job_config.fixed_video_steps = 7
    server.job_config.num_inference_steps = 25

    assert server._get_fixed_video_steps() == 7


def test_maybe_save_async_respects_debug_flag(monkeypatch):
    server = make_server_stub()
    server.job_config.save_debug_artifacts = False
    calls = []

    monkeypatch.setattr("wan_va.wan_va_server.save_async", lambda tensor, path: calls.append(path))

    server._maybe_save_async(torch.zeros(1), "/tmp/unused.pt")

    assert calls == []


def test_run_fixed_video_loop_uses_exact_fixed_k_and_refreshes_cache():
    server = make_server_stub()
    latents = torch.zeros(1, 1, 2, 1, 1)
    video_timesteps = torch.tensor([25.0, 20.0, 15.0, 10.0, 5.0], dtype=torch.float32)
    padded_video_timesteps = torch.tensor([25.0, 20.0, 15.0, 10.0, 5.0, 0.0], dtype=torch.float32)

    out_latents, meta = server._run_fixed_video_loop(
        latents,
        video_timesteps,
        padded_video_timesteps,
        frame_chunk_size=2,
        frame_st_id=0,
    )

    assert len(server.scheduler.calls) == 5
    assert meta["mode"] == "fixed"
    assert meta["fixed_video_steps"] == 5
    assert meta["video_steps_used"] == 5
    assert server._test_refresh_calls[0][0] == 0.0
    assert torch.allclose(out_latents, latents + 5)


def test_infer_returns_scheduler_metadata_when_enabled():
    server = make_server_stub()
    server.job_config.hazard_return_metadata = True
    server._infer = lambda obs, frame_st_id=0: (
        torch.zeros(2, 2, 2),
        torch.zeros(1),
        {"mode": "fixed", "video_steps_used": 5},
    )

    response = server.infer({})

    assert "action" in response
    assert response["scheduler_meta"]["video_steps_used"] == 5


def test_infer_omits_scheduler_metadata_when_disabled():
    server = make_server_stub()
    server.job_config.hazard_return_metadata = False
    server._infer = lambda obs, frame_st_id=0: (
        torch.zeros(2, 2, 2),
        torch.zeros(1),
        {"mode": "fixed", "video_steps_used": 5},
    )

    response = server.infer({})

    assert "action" in response
    assert "scheduler_meta" not in response


def test_invalid_online_scheduler_mode_raises():
    server = make_server_stub()
    server.job_config.online_scheduler_mode = "something-else"

    with pytest.raises(ValueError, match="Unsupported online_scheduler_mode"):
        server._get_online_scheduler_mode()
