# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wan_va.utils.hazard_reward import ConvexCostSchedule, DualAnchorRewardEvaluator, RolloutResult


def make_config():
    q01 = [0.0] * 30
    q99 = [1.0] * 30
    hazard = SimpleNamespace(
        reward_anchor_lo_k=4,
        reward_anchor_hi_k=10,
        reward_w_seq=0.7,
        reward_w_delta=0.3,
        reward_lambda_neg=1.0,
        reward_lambda_c=0.0,
        reward_pos_weight=1.0,
        reward_rot_weight=0.2,
        reward_gripper_weight=0.5,
        reward_delta_pos_weight=1.0,
        reward_delta_rot_weight=0.1,
        reward_delta_gripper_weight=0.5,
        reward_gap_epsilon=1e-4,
        reward_difficulty_kappa=0.1,
        reward_time_rho=0.05,
        reward_time_eta_motion=0.5,
        reward_time_eta_gripper=1.0,
        reward_time_eta_early=0.0,
        reward_cost_c0=0.2,
        reward_cost_c1=0.8,
        reward_cost_gamma=2.0,
        reward_skip_initial_action_frames=1,
    )
    return SimpleNamespace(
        num_inference_steps=25,
        norm_stat={"q01": q01, "q99": q99},
        hazard=hazard,
    )


def to_norm(value: float) -> float:
    return value * 2.0 - 1.0


def make_robotwin_action(left_pos, right_pos, left_grip, right_grip):
    action = torch.full((1, 30, 1, len(left_pos), 1), fill_value=-1.0, dtype=torch.float32)
    left_quat = [-1.0, -1.0, -1.0, 1.0]
    right_quat = [-1.0, -1.0, -1.0, 1.0]
    for idx, pos in enumerate(left_pos):
        action[0, 0:3, 0, idx, 0] = torch.tensor([to_norm(v) for v in pos], dtype=torch.float32)
        action[0, 3:7, 0, idx, 0] = torch.tensor(left_quat, dtype=torch.float32)
        action[0, 7:10, 0, idx, 0] = torch.tensor([to_norm(v) for v in right_pos[idx]], dtype=torch.float32)
        action[0, 10:14, 0, idx, 0] = torch.tensor(right_quat, dtype=torch.float32)
        action[0, 28, 0, idx, 0] = to_norm(left_grip[idx])
        action[0, 29, 0, idx, 0] = to_norm(right_grip[idx])
    return action


def make_mask(timesteps: int):
    mask = torch.zeros((1, 30, 1, timesteps, 1), dtype=torch.bool)
    mask[:, 0:14] = True
    mask[:, 28:30] = True
    return mask


def make_rollout(action: torch.Tensor, steps: int) -> RolloutResult:
    return RolloutResult(
        trajectory=[],
        final_action=action,
        video_steps=steps,
        stop_step=steps,
        final_hazard=None,
        final_stop_prob=None,
    )


def test_dual_anchor_reward_prefers_high_quality_path():
    config = make_config()
    evaluator = DualAnchorRewardEvaluator(config=config, device=torch.device("cpu"))

    gt_action = make_robotwin_action(
        left_pos=[(0.1, 0.2, 0.3), (0.2, 0.3, 0.4), (0.4, 0.5, 0.6)],
        right_pos=[(0.6, 0.5, 0.4), (0.5, 0.4, 0.3), (0.4, 0.3, 0.2)],
        left_grip=[0.0, 0.0, 1.0],
        right_grip=[0.0, 0.0, 0.0],
    )
    lo_action = make_robotwin_action(
        left_pos=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        right_pos=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        left_grip=[0.0, 0.0, 0.0],
        right_grip=[0.0, 0.0, 0.0],
    )
    hi_action = gt_action.clone()
    mask = make_mask(timesteps=3)

    lo_result = make_rollout(lo_action, steps=4)
    hi_result = make_rollout(hi_action, steps=10)

    current_lo = make_rollout(lo_action.clone(), steps=6)
    current_hi = make_rollout(hi_action.clone(), steps=6)

    breakdown_lo = evaluator.evaluate(current_lo, lo_result, hi_result, gt_action, mask)
    breakdown_hi = evaluator.evaluate(current_hi, lo_result, hi_result, gt_action, mask)

    assert abs(breakdown_lo.g_seq) < 1e-6
    assert abs(breakdown_lo.g_delta) < 1e-6
    assert breakdown_hi.g_seq > 0.99
    assert breakdown_hi.g_delta > 0.99
    assert breakdown_hi.reward > breakdown_lo.reward


def test_convex_cost_schedule_is_monotonic_and_back_loaded():
    schedule = ConvexCostSchedule(
        max_video_steps=25,
        c0=0.2,
        c1=0.8,
        gamma=2.0,
        device=torch.device("cpu"),
    )

    cost_1 = schedule(1)
    cost_4 = schedule(4)
    cost_10 = schedule(10)
    cost_24 = schedule(24)
    cost_25 = schedule(25)

    assert 0.0 < cost_1 < cost_4 < cost_10 < cost_25 <= 1.0
    assert (cost_25 - cost_24) > (schedule(2) - schedule(1))


def test_reward_ignores_conditioned_first_action_frame():
    config = make_config()
    evaluator = DualAnchorRewardEvaluator(config=config, device=torch.device("cpu"))

    gt_action = make_robotwin_action(
        left_pos=[(0.0, 0.0, 0.0), (0.2, 0.3, 0.4), (0.3, 0.4, 0.5)],
        right_pos=[(0.0, 0.0, 0.0), (0.5, 0.4, 0.3), (0.4, 0.3, 0.2)],
        left_grip=[0.0, 0.0, 1.0],
        right_grip=[0.0, 0.0, 0.0],
    )
    current_action = gt_action.clone()
    current_action[0, 0:14, 0, 0, 0] = -1.0
    current_action[0, 28:30, 0, 0, 0] = 1.0
    mask = make_mask(timesteps=3)

    lo_result = make_rollout(gt_action.clone(), steps=4)
    hi_result = make_rollout(gt_action.clone(), steps=10)
    current_result = make_rollout(current_action, steps=6)

    breakdown = evaluator.evaluate(current_result, lo_result, hi_result, gt_action, mask)
    assert abs(breakdown.l_seq_cur) < 1e-6
    assert abs(breakdown.l_delta_cur) < 1e-6
