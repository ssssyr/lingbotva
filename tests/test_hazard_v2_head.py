# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wan_va.modules.hazard_scheduler import HazardJumpSchedulerHead


def test_v2_head_initialization_matches_conservative_jump_prior():
    head = HazardJumpSchedulerHead(feature_dim=8, hidden_dim=16)
    feature = torch.randn(1, 8)
    outputs = head(feature, sigma_cur=1.0)

    mode = float(outputs.jump_mode_k.detach().mean().item())
    concentration = float(outputs.jump_concentration_k.detach().mean().item())
    alpha = float(outputs.alpha_k.detach().mean().item())
    beta = float(outputs.beta_k.detach().mean().item())

    assert abs(mode - 0.75) < 1e-3
    assert abs(concentration - 10.0) < 1e-3
    assert alpha > 1.0
    assert beta > 1.0


def test_v2_head_guarantees_unimodal_beta_parameters():
    head = HazardJumpSchedulerHead(feature_dim=8, hidden_dim=16)
    outputs = head(torch.randn(4, 8), sigma_cur=torch.tensor([1.0, 0.8, 0.4, 0.1]))

    assert torch.all(outputs.alpha_k > 1.0)
    assert torch.all(outputs.beta_k > 1.0)
    assert torch.all(outputs.jump_mode_k > 0.0)
    assert torch.all(outputs.jump_mode_k < 1.0)
    assert torch.all(outputs.jump_concentration_k > 2.0)


def test_v2_head_deterministic_jump_uses_mode():
    head = HazardJumpSchedulerHead(feature_dim=8, hidden_dim=16)
    outputs = head(torch.randn(2, 8), sigma_cur=torch.tensor([1.0, 0.5]))

    sampled = head.sample_jump_ratio(outputs.alpha_k, outputs.beta_k, deterministic=True)
    expected_mode = (outputs.alpha_k - 1.0) / (outputs.alpha_k + outputs.beta_k - 2.0)

    assert torch.allclose(sampled, expected_mode, atol=1e-6)


def test_v2_head_recompute_logprob_returns_none_for_forced_stop():
    head = HazardJumpSchedulerHead(feature_dim=8, hidden_dim=16)
    feature = torch.randn(1, 8)

    logprob = head.recompute_step_logprob(
        video_feature=feature,
        sigma_cur=0.005,
        stop_action=True,
        sigma_next=None,
        forced_stop=True,
        forced_continue=False,
    )
    assert logprob is None
