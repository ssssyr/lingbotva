from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class RolloutResult:
    trajectory: List[Dict]
    final_action: torch.Tensor
    video_steps: int
    stop_step: int
    final_hazard: Optional[float] = None
    final_stop_prob: Optional[float] = None
    executed_video_steps: Optional[int] = None
    equivalent_fixed_steps: Optional[float] = None
    terminal_sigma: Optional[float] = None
    sigma_cur_history: Optional[List[float]] = None
    sigma_next_history: Optional[List[Optional[float]]] = None
    jump_ratio_history: Optional[List[Optional[float]]] = None
    jump_distance_history: Optional[List[Optional[float]]] = None
    path_logprob: Optional[float] = None

    def __post_init__(self):
        if self.executed_video_steps is None:
            self.executed_video_steps = int(self.video_steps)
        if self.video_steps is None:
            self.video_steps = int(self.executed_video_steps)


@dataclass
class RewardBreakdown:
    reward: float
    quality: float
    cost: float
    q_seq: float
    q_delta: float
    l_seq_cur: float
    l_seq_lo: float
    l_seq_hi: float
    l_delta_cur: float
    l_delta_lo: float
    l_delta_hi: float
    gap_seq: float
    gap_delta: float
    g_seq: float
    g_delta: float
    d_seq: float
    d_delta: float

    def to_metrics(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ParsedRobotWinActions:
    pos_left: torch.Tensor
    quat_left: torch.Tensor
    pos_right: torch.Tensor
    quat_right: torch.Tensor
    grip_left: torch.Tensor
    grip_right: torch.Tensor
    valid_timestep_count: int


class RobotWinActionSemanticParser:
    LEFT_POS = slice(0, 3)
    LEFT_QUAT = slice(3, 7)
    RIGHT_POS = slice(7, 10)
    RIGHT_QUAT = slice(10, 14)
    LEFT_GRIP = 28
    RIGHT_GRIP = 29

    def __init__(self, config, device, skip_initial_frames: int = 1):
        q01 = torch.tensor(config.norm_stat["q01"], dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
        q99 = torch.tensor(config.norm_stat["q99"], dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
        self.q01 = q01
        self.q99 = q99
        self.skip_initial_frames = max(0, int(skip_initial_frames))

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        action = action.float()
        return (action + 1.0) * 0.5 * (self.q99 - self.q01 + 1e-6) + self.q01

    def parse(self, action: torch.Tensor, action_mask: Optional[torch.Tensor]) -> ParsedRobotWinActions:
        if action.shape[0] != 1:
            raise ValueError(f"RobotWin reward currently expects batch_size=1, got {action.shape[0]}")

        if self.skip_initial_frames > 0:
            if action.shape[2] <= self.skip_initial_frames:
                zeros_3 = torch.zeros((0, 3), dtype=torch.float32, device=action.device)
                zeros_4 = torch.zeros((0, 4), dtype=torch.float32, device=action.device)
                zeros_1 = torch.zeros((0,), dtype=torch.float32, device=action.device)
                return ParsedRobotWinActions(
                    pos_left=zeros_3,
                    quat_left=zeros_4,
                    pos_right=zeros_3,
                    quat_right=zeros_4,
                    grip_left=zeros_1,
                    grip_right=zeros_1,
                    valid_timestep_count=0,
                )
            action = action[:, :, self.skip_initial_frames:].contiguous()
            if action_mask is not None:
                action_mask = action_mask[:, :, self.skip_initial_frames:].contiguous()

        action_denorm = self.denormalize(action)
        action_seq = action_denorm[0, :, :, :, 0].permute(1, 2, 0).reshape(-1, action_denorm.shape[1])

        if action_mask is None:
            mask_seq = torch.ones_like(action_seq, dtype=torch.bool)
        else:
            mask_seq = action_mask[0, :, :, :, 0].permute(1, 2, 0).reshape(-1, action_mask.shape[1]).bool()

        valid_timestep = mask_seq.any(dim=1)
        action_seq = action_seq[valid_timestep]

        if action_seq.numel() == 0:
            zeros_3 = torch.zeros((0, 3), dtype=torch.float32, device=action.device)
            zeros_4 = torch.zeros((0, 4), dtype=torch.float32, device=action.device)
            zeros_1 = torch.zeros((0,), dtype=torch.float32, device=action.device)
            return ParsedRobotWinActions(
                pos_left=zeros_3,
                quat_left=zeros_4,
                pos_right=zeros_3,
                quat_right=zeros_4,
                grip_left=zeros_1,
                grip_right=zeros_1,
                valid_timestep_count=0,
            )

        quat_left = self._normalize_quaternion(action_seq[:, self.LEFT_QUAT])
        quat_right = self._normalize_quaternion(action_seq[:, self.RIGHT_QUAT])
        grip_left = action_seq[:, self.LEFT_GRIP].clamp(0.0, 1.0)
        grip_right = action_seq[:, self.RIGHT_GRIP].clamp(0.0, 1.0)

        return ParsedRobotWinActions(
            pos_left=action_seq[:, self.LEFT_POS],
            quat_left=quat_left,
            pos_right=action_seq[:, self.RIGHT_POS],
            quat_right=quat_right,
            grip_left=grip_left,
            grip_right=grip_right,
            valid_timestep_count=int(action_seq.shape[0]),
        )

    @staticmethod
    def _normalize_quaternion(quat: torch.Tensor) -> torch.Tensor:
        if quat.numel() == 0:
            return quat.float()
        return quat.float() / quat.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)


class NonUniformTimeWeightBuilder:
    def __init__(self, rho: float, eta_motion: float, eta_gripper: float, eta_early: float, gripper_event_threshold: float = 0.25):
        self.rho = float(rho)
        self.eta_motion = float(eta_motion)
        self.eta_gripper = float(eta_gripper)
        self.eta_early = float(eta_early)
        self.gripper_event_threshold = float(gripper_event_threshold)

    def build_sequence_weights(self, gt: ParsedRobotWinActions) -> torch.Tensor:
        timestep_count = gt.valid_timestep_count
        if timestep_count <= 0:
            return gt.pos_left.new_zeros((0,), dtype=torch.float32)

        motion_score, gripper_event = self._compute_motion_terms(gt)
        rank = torch.arange(timestep_count, device=gt.pos_left.device, dtype=torch.float32)
        raw = torch.exp(-self.rho * rank)
        raw = raw * (
            1.0
            + self.eta_motion * motion_score
            + self.eta_gripper * gripper_event
            + self.eta_early * torch.exp(-rank)
        )
        return raw / raw.sum().clamp_min(1e-6)

    def build_delta_weights(self, gt: ParsedRobotWinActions) -> torch.Tensor:
        timestep_count = gt.valid_timestep_count
        if timestep_count <= 1:
            return gt.pos_left.new_zeros((0,), dtype=torch.float32)

        motion_score, gripper_event = self._compute_motion_terms(gt)
        rank = torch.arange(timestep_count - 1, device=gt.pos_left.device, dtype=torch.float32)
        transition_motion = motion_score[1:]
        transition_event = gripper_event[1:]
        raw = torch.exp(-self.rho * rank)
        raw = raw * (
            1.0
            + self.eta_motion * transition_motion
            + self.eta_gripper * transition_event
            + self.eta_early * torch.exp(-rank)
        )
        return raw / raw.sum().clamp_min(1e-6)

    def _compute_motion_terms(self, gt: ParsedRobotWinActions) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_count = gt.valid_timestep_count
        motion_score = gt.pos_left.new_zeros((timestep_count,), dtype=torch.float32)
        gripper_event = gt.pos_left.new_zeros((timestep_count,), dtype=torch.float32)
        if timestep_count <= 1:
            return motion_score, gripper_event

        pos_motion = (
            (gt.pos_left[1:] - gt.pos_left[:-1]).abs().sum(dim=-1)
            + (gt.pos_right[1:] - gt.pos_right[:-1]).abs().sum(dim=-1)
        )
        rot_motion = (
            DualAnchorRewardEvaluator.quaternion_angle(gt.quat_left[1:], gt.quat_left[:-1])
            + DualAnchorRewardEvaluator.quaternion_angle(gt.quat_right[1:], gt.quat_right[:-1])
        )
        motion = pos_motion + 0.25 * rot_motion
        motion = motion / motion.max().clamp_min(1e-6)
        motion_score[1:] = motion

        grip_delta = torch.maximum(
            (gt.grip_left[1:] - gt.grip_left[:-1]).abs(),
            (gt.grip_right[1:] - gt.grip_right[:-1]).abs(),
        )
        gripper_event[1:] = (grip_delta > self.gripper_event_threshold).float()
        return motion_score, gripper_event


class ConvexCostSchedule:
    def __init__(self, max_video_steps: int, c0: float, c1: float, gamma: float, device):
        self.max_video_steps = max(1, int(max_video_steps))
        self.c0 = float(c0)
        self.c1 = float(c1)
        self.gamma = float(gamma)

        step_ids = torch.arange(1, self.max_video_steps + 1, dtype=torch.float32, device=device)
        step_cost = self.c0 + self.c1 * torch.pow(step_ids / float(self.max_video_steps), self.gamma)
        prefix = torch.zeros(self.max_video_steps + 1, dtype=torch.float32, device=device)
        prefix[1:] = torch.cumsum(step_cost, dim=0)
        self.prefix_cost = prefix
        self.norm = prefix[-1].clamp_min(1e-6)

    def __call__(self, video_steps: int) -> float:
        clamped = max(0, min(int(video_steps), self.max_video_steps))
        return float((self.prefix_cost[clamped] / self.norm).item())


class DualAnchorRewardEvaluator:
    def __init__(self, config, device):
        hazard_cfg = getattr(config, "hazard", None)
        if hazard_cfg is None:
            hazard_cfg = config

        self.device = device
        max_video_steps = int(config.num_inference_steps)
        self.anchor_lo_k = max(1, min(int(getattr(hazard_cfg, "reward_anchor_lo_k", 4)), max_video_steps))
        self.anchor_hi_k = max(
            self.anchor_lo_k,
            min(int(getattr(hazard_cfg, "reward_anchor_hi_k", max_video_steps)), max_video_steps),
        )
        self.w_seq = float(getattr(hazard_cfg, "reward_w_seq", 0.7))
        self.w_delta = float(getattr(hazard_cfg, "reward_w_delta", 0.3))
        self.lambda_neg = float(getattr(hazard_cfg, "reward_lambda_neg", 1.0))
        self.lambda_c = float(getattr(hazard_cfg, "reward_lambda_c", getattr(hazard_cfg, "lambda_cost", 0.4)))

        self.pos_weight = float(getattr(hazard_cfg, "reward_pos_weight", 1.0))
        self.rot_weight = float(getattr(hazard_cfg, "reward_rot_weight", 0.6))
        self.gripper_weight = float(getattr(hazard_cfg, "reward_gripper_weight", 0.8))
        self.delta_pos_weight = float(getattr(hazard_cfg, "reward_delta_pos_weight", 1.0))
        self.delta_rot_weight = float(getattr(hazard_cfg, "reward_delta_rot_weight", 0.2))
        self.delta_gripper_weight = float(getattr(hazard_cfg, "reward_delta_gripper_weight", 0.6))

        self.gap_epsilon = float(getattr(hazard_cfg, "reward_gap_epsilon", 1e-4))
        self.difficulty_kappa = float(getattr(hazard_cfg, "reward_difficulty_kappa", 0.1))
        self.skip_initial_action_frames = int(getattr(hazard_cfg, "reward_skip_initial_action_frames", 1))

        self.action_parser = RobotWinActionSemanticParser(
            config=config,
            device=device,
            skip_initial_frames=self.skip_initial_action_frames,
        )
        self.weight_builder = NonUniformTimeWeightBuilder(
            rho=float(getattr(hazard_cfg, "reward_time_rho", 0.08)),
            eta_motion=float(getattr(hazard_cfg, "reward_time_eta_motion", 0.6)),
            eta_gripper=float(getattr(hazard_cfg, "reward_time_eta_gripper", 1.2)),
            eta_early=float(getattr(hazard_cfg, "reward_time_eta_early", 0.0)),
        )
        self.cost_schedule = ConvexCostSchedule(
            max_video_steps=max_video_steps,
            c0=float(getattr(hazard_cfg, "reward_cost_c0", 0.2)),
            c1=float(getattr(hazard_cfg, "reward_cost_c1", 0.8)),
            gamma=float(getattr(hazard_cfg, "reward_cost_gamma", 2.0)),
            device=device,
        )

    @torch.no_grad()
    def evaluate(
        self,
        current: RolloutResult,
        anchor_lo: RolloutResult,
        anchor_hi: RolloutResult,
        gt_action: torch.Tensor,
        gt_action_mask: Optional[torch.Tensor] = None,
    ) -> RewardBreakdown:
        gt = self.action_parser.parse(gt_action, gt_action_mask)
        cur = self.action_parser.parse(current.final_action, gt_action_mask)
        lo = self.action_parser.parse(anchor_lo.final_action, gt_action_mask)
        hi = self.action_parser.parse(anchor_hi.final_action, gt_action_mask)

        l_seq_cur = self._sequence_loss(cur, gt)
        l_seq_lo = self._sequence_loss(lo, gt)
        l_seq_hi = self._sequence_loss(hi, gt)

        l_delta_cur = self._delta_loss(cur, gt)
        l_delta_lo = self._delta_loss(lo, gt)
        l_delta_hi = self._delta_loss(hi, gt)

        q_seq, gap_seq, g_seq, d_seq = self._quality_term(
            current_loss=l_seq_cur,
            low_anchor_loss=l_seq_lo,
            high_anchor_loss=l_seq_hi,
            weight=self.w_seq,
        )
        q_delta, gap_delta, g_delta, d_delta = self._quality_term(
            current_loss=l_delta_cur,
            low_anchor_loss=l_delta_lo,
            high_anchor_loss=l_delta_hi,
            weight=self.w_delta,
        )

        quality = q_seq + q_delta
        cost = self.lambda_c * self.cost_schedule(current.video_steps)
        reward = quality - cost

        return RewardBreakdown(
            reward=float(reward),
            quality=float(quality),
            cost=float(cost),
            q_seq=float(q_seq),
            q_delta=float(q_delta),
            l_seq_cur=float(l_seq_cur),
            l_seq_lo=float(l_seq_lo),
            l_seq_hi=float(l_seq_hi),
            l_delta_cur=float(l_delta_cur),
            l_delta_lo=float(l_delta_lo),
            l_delta_hi=float(l_delta_hi),
            gap_seq=float(gap_seq),
            gap_delta=float(gap_delta),
            g_seq=float(g_seq),
            g_delta=float(g_delta),
            d_seq=float(d_seq),
            d_delta=float(d_delta),
        )

    @staticmethod
    def quaternion_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        if q1.numel() == 0 or q2.numel() == 0:
            return q1.new_zeros((0,), dtype=torch.float32)
        q1 = q1.float() / q1.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q2 = q2.float() / q2.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0 - 1e-7)
        return 2.0 * torch.arccos(dot)

    @staticmethod
    def quaternion_inverse(quat: torch.Tensor) -> torch.Tensor:
        inv = quat.clone()
        inv[..., :3] = -inv[..., :3]
        return inv / quat.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        x1, y1, z1, w1 = q1.unbind(dim=-1)
        x2, y2, z2, w2 = q2.unbind(dim=-1)
        return torch.stack(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            dim=-1,
        )

    def _sequence_loss(self, pred: ParsedRobotWinActions, gt: ParsedRobotWinActions) -> float:
        if gt.valid_timestep_count == 0:
            return 0.0

        weights = self.weight_builder.build_sequence_weights(gt)
        pos_err = (
            F.smooth_l1_loss(pred.pos_left, gt.pos_left, reduction="none").sum(dim=-1)
            + F.smooth_l1_loss(pred.pos_right, gt.pos_right, reduction="none").sum(dim=-1)
        )
        rot_err = self.quaternion_angle(pred.quat_left, gt.quat_left) + self.quaternion_angle(pred.quat_right, gt.quat_right)
        grip_err = self._binary_cross_entropy(pred.grip_left, gt.grip_left) + self._binary_cross_entropy(pred.grip_right, gt.grip_right)

        per_timestep = self.pos_weight * pos_err + self.rot_weight * rot_err + self.gripper_weight * grip_err
        return float((weights * per_timestep).sum().item())

    def _delta_loss(self, pred: ParsedRobotWinActions, gt: ParsedRobotWinActions) -> float:
        if gt.valid_timestep_count <= 1:
            return 0.0

        weights = self.weight_builder.build_delta_weights(gt)

        pos_delta_pred_left = pred.pos_left[1:] - pred.pos_left[:-1]
        pos_delta_gt_left = gt.pos_left[1:] - gt.pos_left[:-1]
        pos_delta_pred_right = pred.pos_right[1:] - pred.pos_right[:-1]
        pos_delta_gt_right = gt.pos_right[1:] - gt.pos_right[:-1]
        pos_delta_err = (
            F.smooth_l1_loss(pos_delta_pred_left, pos_delta_gt_left, reduction="none").sum(dim=-1)
            + F.smooth_l1_loss(pos_delta_pred_right, pos_delta_gt_right, reduction="none").sum(dim=-1)
        )

        quat_delta_pred_left = self.quaternion_multiply(pred.quat_left[1:], self.quaternion_inverse(pred.quat_left[:-1]))
        quat_delta_gt_left = self.quaternion_multiply(gt.quat_left[1:], self.quaternion_inverse(gt.quat_left[:-1]))
        quat_delta_pred_right = self.quaternion_multiply(pred.quat_right[1:], self.quaternion_inverse(pred.quat_right[:-1]))
        quat_delta_gt_right = self.quaternion_multiply(gt.quat_right[1:], self.quaternion_inverse(gt.quat_right[:-1]))
        rot_delta_err = self.quaternion_angle(quat_delta_pred_left, quat_delta_gt_left) + self.quaternion_angle(quat_delta_pred_right, quat_delta_gt_right)

        grip_delta_pred_left = pred.grip_left[1:] - pred.grip_left[:-1]
        grip_delta_gt_left = gt.grip_left[1:] - gt.grip_left[:-1]
        grip_delta_pred_right = pred.grip_right[1:] - pred.grip_right[:-1]
        grip_delta_gt_right = gt.grip_right[1:] - gt.grip_right[:-1]
        grip_delta_err = (
            F.smooth_l1_loss(grip_delta_pred_left, grip_delta_gt_left, reduction="none")
            + F.smooth_l1_loss(grip_delta_pred_right, grip_delta_gt_right, reduction="none")
        )

        per_transition = (
            self.delta_pos_weight * pos_delta_err
            + self.delta_rot_weight * rot_delta_err
            + self.delta_gripper_weight * grip_delta_err
        )
        return float((weights * per_transition).sum().item())

    def _quality_term(self, current_loss: float, low_anchor_loss: float, high_anchor_loss: float, weight: float) -> tuple[float, float, float, float]:
        improvement = max(low_anchor_loss - high_anchor_loss, 0.0)
        gap = max(low_anchor_loss - high_anchor_loss, self.gap_epsilon)
        gain = float(max(min((low_anchor_loss - current_loss) / gap, 1.0), -1.0))
        difficulty = improvement / (improvement + self.difficulty_kappa)
        quality = weight * (difficulty * max(gain, 0.0) - self.lambda_neg * max(-gain, 0.0))
        return quality, gap, gain, difficulty

    @staticmethod
    def _binary_cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-5, 1.0 - 1e-5)
        target = target.clamp(0.0, 1.0)
        return -(target * pred.log() + (1.0 - target) * (1.0 - pred).log())
