# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wan_va.utils.scheduler import FlowMatchScheduler


def test_custom_step_matches_closed_form_update():
    scheduler = FlowMatchScheduler(num_inference_steps=5, extra_one_step=True)
    sample = torch.randn(1, 2, 3)
    model_output = torch.randn(1, 2, 3)
    sigma_cur = 1.0
    sigma_next = 0.7

    result = scheduler.custom_step(
        model_output=model_output,
        sigma_cur=sigma_cur,
        sigma_next=sigma_next,
        sample=sample,
        return_dict=False,
    )

    expected = sample + model_output * (sigma_next - sigma_cur)
    assert torch.allclose(result, expected, atol=1e-6)


def test_custom_step_is_noop_when_sigma_does_not_change():
    scheduler = FlowMatchScheduler(num_inference_steps=5, extra_one_step=True)
    sample = torch.randn(1, 2, 3)
    model_output = torch.randn(1, 2, 3)

    result = scheduler.custom_step(
        model_output=model_output,
        sigma_cur=0.5,
        sigma_next=0.5,
        sample=sample,
        return_dict=False,
    )

    assert torch.allclose(result, sample, atol=1e-6)


def test_custom_step_advances_towards_next_fixed_sigma():
    scheduler = FlowMatchScheduler(num_inference_steps=5, extra_one_step=True)
    sample = torch.randn(1, 2, 3)
    model_output = torch.randn(1, 2, 3)
    sigma_cur = float(scheduler.sigmas[0].item())
    sigma_next = float(scheduler.sigmas[1].item())

    result = scheduler.custom_step(
        model_output=model_output,
        sigma_cur=sigma_cur,
        sigma_next=sigma_next,
        sample=sample,
        return_dict=False,
    )
    fixed_step = sample + model_output * (sigma_next - sigma_cur)

    assert torch.allclose(result, fixed_step, atol=1e-6)
