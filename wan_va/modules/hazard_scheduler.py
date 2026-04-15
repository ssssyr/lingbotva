# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class HazardSchedulerHead(nn.Module):
    """
    Hazard Scheduler Head for learning when to stop video denoising.

    Outputs a non-negative Hazard increment delta_H_k at each video step k.
    The cumulative Hazard H_k = sum_{i=1}^{k} delta_H_i determines the stopping probability.

    Key properties:
    - delta_H_k >= 0 (enforced by softplus activation)
    - H_k is monotonically increasing
    - Stopping probability h_k = 1 - exp(-delta_H_k) is in [0, 1]
    """

    def __init__(
        self,
        feature_dim: int = None,
        hidden_dim: int = None,
        output_dim: int = 1,
        use_context: bool = False,
        context_dim: int = 0,
    ):
        """
        Args:
            feature_dim: Dimension of pooled video hidden features.
                When omitted, falls back to `hidden_dim` for backward compatibility.
            hidden_dim: Hidden width of the scheduler MLP.
                When omitted, defaults to `feature_dim`.
            output_dim: Output dimension (default 1 for scalar Hazard)
            use_context: Whether to use additional context features
            context_dim: Dimension of context features (e.g., task embedding)
        """
        super().__init__()

        if feature_dim is None and hidden_dim is None:
            raise ValueError("Either feature_dim or hidden_dim must be provided")
        if feature_dim is None:
            feature_dim = hidden_dim
        if hidden_dim is None:
            hidden_dim = feature_dim

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = output_dim
        self.use_context = use_context
        self.context_dim = context_dim
        bottleneck_dim = max(self.hidden_dim // 2, 1)

        # Input: video_feature + step_id + optional context
        input_dim = self.feature_dim + 1  # +1 for normalized step_id
        if use_context:
            input_dim += context_dim

        # MLP for Hazard prediction
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, output_dim),
        )

        # Initialize output layer to near zero
        # This makes initial delta_H_k small, avoiding premature stopping
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -2.0)  # softplus(-2) ≈ 0.127

    def forward(
        self,
        video_feature: torch.Tensor,
        step_id: torch.Tensor,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute Hazard increment.

        Args:
            video_feature: [B, D] pooled video hidden feature
            step_id: [B] or [B, 1], normalized to [0, 1], represents k/K_max
            context: [B, C] optional context features (e.g., task embedding)

        Returns:
            delta_H_k: [B, 1] non-negative Hazard increment
        """
        param_dtype = self.mlp[0].weight.dtype
        param_device = self.mlp[0].weight.device
        video_feature = video_feature.to(device=param_device, dtype=param_dtype)

        # Ensure step_id is 2D
        if step_id.dim() == 1:
            step_id = step_id.unsqueeze(-1)
        step_id = step_id.to(device=param_device, dtype=param_dtype)

        # Concatenate inputs
        x = torch.cat([video_feature, step_id], dim=-1)

        if self.use_context:
            if context is None:
                raise ValueError("context is required when use_context=True")
            context = context.to(device=param_device, dtype=param_dtype)
            x = torch.cat([x, context], dim=-1)

        # Compute logit
        logit = self.mlp(x)

        # Apply softplus to ensure non-negativity
        delta_H_k = F.softplus(logit.float())
        delta_H_k = torch.nan_to_num(delta_H_k, nan=0.0, posinf=20.0, neginf=0.0)

        return delta_H_k

    def compute_stop_probability(self, delta_H_k: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional stopping probability from Hazard increment.

        h_k = 1 - exp(-delta_H_k)

        Args:
            delta_H_k: [B, 1] Hazard increment

        Returns:
            h_k: [B, 1] stopping probability in [0, 1]
        """
        h_k = 1.0 - torch.exp(-delta_H_k)
        h_k = torch.nan_to_num(h_k, nan=0.5, posinf=1.0, neginf=0.0)
        h_k = torch.clamp(h_k, 1e-6, 1.0 - 1e-6)
        return h_k

    def compute_cumulative_hazard(self, delta_H_list: list) -> torch.Tensor:
        """
        Compute cumulative Hazard from a list of increments.

        H_k = sum_{i=1}^{k} delta_H_i

        Args:
            delta_H_list: List of [B, 1] tensors

        Returns:
            H_k: [B, 1] cumulative Hazard
        """
        H_k = torch.stack(delta_H_list, dim=0).sum(dim=0)
        return H_k

    def compute_survival_probability(self, H_k: torch.Tensor) -> torch.Tensor:
        """
        Compute survival probability (probability of not stopping by step k).

        F_k = 1 - exp(-H_k)
        S_k = exp(-H_k) = 1 - F_k

        Args:
            H_k: [B, 1] cumulative Hazard

        Returns:
            S_k: [B, 1] survival probability
        """
        S_k = torch.exp(-H_k)
        return S_k


class HazardScheduler:
    """
    Hazard-based scheduler for adaptive video denoising stopping.

    This class manages the stopping decision logic during inference or training.
    """

    def __init__(
        self,
        scheduler_head: HazardSchedulerHead,
        K_max: int = 25,
        K_min: int = 3,
        eta: float = 0.5,
        mode: str = 'train',
    ):
        """
        Args:
            scheduler_head: HazardSchedulerHead module
            K_max: Maximum number of video steps
            K_min: Minimum number of video steps before stopping is allowed
            eta: Stopping threshold for deterministic inference (F_k >= eta)
            mode: 'train' (stochastic sampling) or 'eval' (deterministic)
        """
        self.scheduler_head = scheduler_head
        self.K_max = K_max
        self.K_min = K_min
        self.eta = eta
        self.mode = mode

        # State tracking
        self.reset()

    def reset(self):
        """Reset scheduler state for a new rollout."""
        self.step_count = 0
        self.H_cumulative = 0.0
        self.delta_H_history = []
        self.h_k_history = []
        self.stopped = False
        self.stop_step = None
        # CRITICAL: Clear tensor state to avoid cross-sample contamination
        if hasattr(self, 'H_cumulative_tensor'):
            delattr(self, 'H_cumulative_tensor')

    def step(
        self,
        video_feature: torch.Tensor,
        context: torch.Tensor = None,
    ) -> dict:
        """
        Perform one step of the scheduler.

        Args:
            video_feature: [B, D] pooled video hidden feature
            context: [B, C] optional context features

        Returns:
            result: Dict containing:
                - delta_H_k: Hazard increment
                - h_k: stopping probability
                - should_stop: whether to stop (bool)
                - log_prob: log probability of the action taken (for REINFORCE)
        """
        if self.stopped:
            raise RuntimeError("Scheduler has already stopped. Call reset() first.")
        if video_feature.shape[0] != 1:
            raise ValueError(
                "HazardScheduler V1 currently only supports batch_size=1 "
                f"(got {video_feature.shape[0]})."
            )

        # Compute normalized step_id
        step_id = torch.tensor(
            [self.step_count / self.K_max],
            dtype=video_feature.dtype,
            device=video_feature.device,
        ).expand(video_feature.shape[0], 1)

        # Compute Hazard increment
        delta_H_k = self.scheduler_head(video_feature, step_id, context)

        # Compute stopping probability
        h_k = self.scheduler_head.compute_stop_probability(delta_H_k)

        # Update cumulative Hazard (keep as tensor, not scalar)
        # For batch processing, we track per-sample cumulative Hazard
        if not hasattr(self, 'H_cumulative_tensor'):
            self.H_cumulative_tensor = torch.zeros(
                delta_H_k.shape[0],
                dtype=delta_H_k.dtype,
                device=delta_H_k.device
            )
        self.H_cumulative_tensor = self.H_cumulative_tensor + delta_H_k.squeeze(-1)
        self.H_cumulative = self.H_cumulative_tensor.mean().item()  # For logging only
        self.delta_H_history.append(delta_H_k)
        self.h_k_history.append(h_k)

        # Decide whether to stop
        if self.step_count < self.K_min:
            # Force continue before K_min
            should_stop = False
            log_prob = torch.log(1.0 - h_k + 1e-8).mean()  # Average over batch
        elif self.step_count >= self.K_max - 1:
            # Force stop at K_max
            should_stop = True
            log_prob = torch.log(h_k + 1e-8).mean()  # Average over batch
        else:
            if self.mode == 'train':
                # Stochastic sampling - use batch mean for single decision
                h_k_mean = torch.nan_to_num(h_k.mean(), nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6)
                h_k_mean = torch.clamp(h_k_mean, 1e-6, 1.0 - 1e-6)
                should_stop = torch.bernoulli(h_k_mean).bool().item()
                if should_stop:
                    log_prob = torch.log(h_k + 1e-8).mean()
                else:
                    log_prob = torch.log(1.0 - h_k + 1e-8).mean()
            else:
                # Deterministic inference
                F_k = 1.0 - torch.exp(-self.H_cumulative_tensor.mean())
                should_stop = (F_k >= self.eta).item()
                log_prob = None  # Not needed in eval mode

        # Update state
        self.step_count += 1
        if should_stop:
            self.stopped = True
            self.stop_step = self.step_count

        result = {
            'delta_H_k': delta_H_k,
            'h_k': h_k,
            'should_stop': should_stop,
            'log_prob': log_prob,
            'step_count': self.step_count,
            'H_cumulative': self.H_cumulative,
        }

        return result

    def get_trajectory(self) -> dict:
        """
        Get the full trajectory after stopping.

        Returns:
            trajectory: Dict containing:
                - delta_H_history: List of Hazard increments
                - h_k_history: List of stopping probabilities
                - stop_step: Step at which stopping occurred
                - H_cumulative: Final cumulative Hazard
        """
        return {
            'delta_H_history': self.delta_H_history,
            'h_k_history': self.h_k_history,
            'stop_step': self.stop_step,
            'H_cumulative': self.H_cumulative,
        }


def _inverse_softplus(x: float) -> float:
    x_tensor = torch.tensor(float(x), dtype=torch.float32)
    return float(torch.log(torch.exp(x_tensor) - 1.0).item())


@dataclass
class HazardJumpHeadOutput:
    delta_H_k: torch.Tensor
    h_k: torch.Tensor
    jump_mode_k: torch.Tensor
    jump_concentration_k: torch.Tensor
    alpha_k: torch.Tensor
    beta_k: torch.Tensor


class HazardJumpSchedulerHead(nn.Module):
    """
    V2 scheduler head that jointly predicts:
    - Hazard increment delta_H_k
    - Jump distribution parameters via mode + concentration

    The jump distribution is parameterized so that alpha > 1 and beta > 1,
    ensuring a well-defined mode for deterministic deployment.
    """

    def __init__(
        self,
        feature_dim: int = None,
        hidden_dim: int = None,
        use_context: bool = False,
        context_dim: int = 0,
        min_mode_eps: float = 1e-3,
        init_jump_mode: float = 0.75,
        init_jump_concentration: float = 10.0,
    ):
        super().__init__()

        if feature_dim is None and hidden_dim is None:
            raise ValueError("Either feature_dim or hidden_dim must be provided")
        if feature_dim is None:
            feature_dim = hidden_dim
        if hidden_dim is None:
            hidden_dim = feature_dim

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_context = bool(use_context)
        self.context_dim = int(context_dim)
        self.min_mode_eps = float(min_mode_eps)
        bottleneck_dim = max(self.hidden_dim // 2, 1)

        input_dim = self.feature_dim + 1
        if self.use_context:
            input_dim += self.context_dim

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim),
        )
        self.hazard_head = nn.Linear(bottleneck_dim, 1)
        self.jump_head = nn.Linear(bottleneck_dim, 2)

        # Conservative initialization:
        # - small hazard increment
        # - small denoising step (retain ~75% sigma each step)
        nn.init.zeros_(self.hazard_head.weight)
        nn.init.constant_(self.hazard_head.bias, -2.0)

        nn.init.zeros_(self.jump_head.weight)
        init_mode = min(max(float(init_jump_mode), self.min_mode_eps), 1.0 - self.min_mode_eps)
        init_concentration = max(float(init_jump_concentration), 2.0 + self.min_mode_eps)
        mode_scale = (init_mode - self.min_mode_eps) / (1.0 - 2.0 * self.min_mode_eps)
        mode_scale = min(max(mode_scale, self.min_mode_eps), 1.0 - self.min_mode_eps)
        mode_bias = float(torch.logit(torch.tensor(mode_scale, dtype=torch.float32)).item())
        concentration_bias = _inverse_softplus(init_concentration - 2.0 - self.min_mode_eps)
        nn.init.constant_(self.jump_head.bias[0], mode_bias)
        nn.init.constant_(self.jump_head.bias[1], concentration_bias)

    def _prepare_inputs(self, video_feature: torch.Tensor, sigma_cur: torch.Tensor, context: torch.Tensor = None):
        param_dtype = self.shared_net[0].weight.dtype
        param_device = self.shared_net[0].weight.device
        video_feature = video_feature.to(device=param_device, dtype=param_dtype)

        if not torch.is_tensor(sigma_cur):
            sigma_cur = torch.tensor([sigma_cur], dtype=param_dtype, device=param_device)
        sigma_cur = sigma_cur.to(device=param_device, dtype=param_dtype)
        if sigma_cur.dim() == 0:
            sigma_cur = sigma_cur.unsqueeze(0)
        if sigma_cur.dim() == 1:
            sigma_cur = sigma_cur.unsqueeze(-1)

        x = torch.cat([video_feature, sigma_cur], dim=-1)
        if self.use_context:
            if context is None:
                raise ValueError("context is required when use_context=True")
            context = context.to(device=param_device, dtype=param_dtype)
            x = torch.cat([x, context], dim=-1)
        return x

    def forward(
        self,
        video_feature: torch.Tensor,
        sigma_cur: torch.Tensor,
        context: torch.Tensor = None,
    ) -> HazardJumpHeadOutput:
        x = self._prepare_inputs(video_feature, sigma_cur, context=context)
        shared = self.shared_net(x)

        hazard_logit = self.hazard_head(shared)
        delta_H_k = F.softplus(hazard_logit.float())
        delta_H_k = torch.nan_to_num(delta_H_k, nan=0.0, posinf=20.0, neginf=0.0)

        jump_raw = self.jump_head(shared).float()
        jump_mode_k = torch.sigmoid(jump_raw[:, 0:1])
        jump_mode_k = jump_mode_k * (1.0 - 2.0 * self.min_mode_eps) + self.min_mode_eps
        jump_mode_k = torch.clamp(jump_mode_k, self.min_mode_eps, 1.0 - self.min_mode_eps)

        jump_concentration_k = F.softplus(jump_raw[:, 1:2]) + 2.0 + self.min_mode_eps
        jump_concentration_k = torch.nan_to_num(
            jump_concentration_k,
            nan=2.0 + self.min_mode_eps,
            posinf=1e3,
            neginf=2.0 + self.min_mode_eps,
        )

        alpha_k = jump_mode_k * (jump_concentration_k - 2.0) + 1.0
        beta_k = (1.0 - jump_mode_k) * (jump_concentration_k - 2.0) + 1.0

        return HazardJumpHeadOutput(
            delta_H_k=delta_H_k,
            h_k=self.compute_stop_probability(delta_H_k),
            jump_mode_k=jump_mode_k,
            jump_concentration_k=jump_concentration_k,
            alpha_k=alpha_k,
            beta_k=beta_k,
        )

    @staticmethod
    def compute_stop_probability(delta_H_k: torch.Tensor) -> torch.Tensor:
        h_k = 1.0 - torch.exp(-delta_H_k)
        h_k = torch.nan_to_num(h_k, nan=0.5, posinf=1.0, neginf=0.0)
        return torch.clamp(h_k, 1e-6, 1.0 - 1e-6)

    @staticmethod
    def beta_distribution(alpha_k: torch.Tensor, beta_k: torch.Tensor) -> torch.distributions.Beta:
        alpha_k = torch.clamp(alpha_k.float(), min=1.0 + 1e-6)
        beta_k = torch.clamp(beta_k.float(), min=1.0 + 1e-6)
        return torch.distributions.Beta(alpha_k, beta_k)

    def sample_jump_ratio(
        self,
        alpha_k: torch.Tensor,
        beta_k: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if deterministic:
            ratio = (alpha_k - 1.0) / torch.clamp(alpha_k + beta_k - 2.0, min=1e-6)
        else:
            ratio = self.beta_distribution(alpha_k, beta_k).sample()
        ratio = torch.clamp(ratio, self.min_mode_eps, 1.0 - self.min_mode_eps)
        return ratio

    def compute_action_logprob(
        self,
        h_k: torch.Tensor,
        stop_action: bool,
        jump_ratio: torch.Tensor = None,
        alpha_k: torch.Tensor = None,
        beta_k: torch.Tensor = None,
        forced_stop: bool = False,
        forced_continue: bool = False,
        jump_logprob_scale: float = 1.0,
    ) -> torch.Tensor:
        if forced_stop:
            return None

        if stop_action:
            if forced_continue:
                return None
            return torch.log(h_k + 1e-8).mean()

        if jump_ratio is None or alpha_k is None or beta_k is None:
            raise ValueError("jump_ratio, alpha_k and beta_k are required for continue actions")

        jump_logprob = self.beta_distribution(alpha_k, beta_k).log_prob(jump_ratio).mean()
        jump_logprob = jump_logprob * float(jump_logprob_scale)
        if forced_continue:
            return jump_logprob
        return torch.log(1.0 - h_k + 1e-8).mean() + jump_logprob

    def recompute_step_logprob(
        self,
        video_feature: torch.Tensor,
        sigma_cur: float,
        stop_action: bool,
        sigma_next: float = None,
        forced_stop: bool = False,
        forced_continue: bool = False,
        context: torch.Tensor = None,
        jump_logprob_scale: float = 1.0,
    ) -> torch.Tensor:
        outputs = self(video_feature, sigma_cur=sigma_cur, context=context)
        jump_ratio = None
        if not stop_action and sigma_next is not None:
            sigma_cur_value = max(float(sigma_cur), self.min_mode_eps)
            jump_ratio = max(min(float(sigma_next) / sigma_cur_value, 1.0 - self.min_mode_eps), self.min_mode_eps)
            jump_ratio = outputs.alpha_k.new_tensor([[jump_ratio]], dtype=outputs.alpha_k.dtype)
        return self.compute_action_logprob(
            h_k=outputs.h_k,
            stop_action=bool(stop_action),
            jump_ratio=jump_ratio,
            alpha_k=outputs.alpha_k,
            beta_k=outputs.beta_k,
            forced_stop=bool(forced_stop),
            forced_continue=bool(forced_continue),
            jump_logprob_scale=jump_logprob_scale,
        )


class HazardJumpScheduler:
    """
    Scheduler runtime for the stop + jump V2 policy.

    Each decision step consumes one video forward pass and then:
    - force stops if sigma is already tiny
    - samples or decides stop/continue via hazard
    - if continue, predicts a jump ratio and updates sigma externally
    """

    def __init__(
        self,
        scheduler_head: HazardJumpSchedulerHead,
        K_max: int = 25,
        K_min: int = 3,
        eta: float = 0.5,
        sigma_min: float = 0.01,
        jump_logprob_scale: float = 1.0,
        mode: str = "train",
    ):
        self.scheduler_head = scheduler_head
        self.K_max = int(K_max)
        self.K_min = int(K_min)
        self.eta = float(eta)
        self.sigma_min = float(sigma_min)
        self.jump_logprob_scale = float(jump_logprob_scale)
        self.mode = str(mode)
        self.reset()

    def reset(self):
        self.step_count = 0
        self.H_cumulative = 0.0
        self.delta_H_history = []
        self.h_k_history = []
        self.jump_ratio_history = []
        self.stopped = False
        self.stop_step = None
        if hasattr(self, "H_cumulative_tensor"):
            delattr(self, "H_cumulative_tensor")

    def step(self, video_feature: torch.Tensor, sigma_cur: float, context: torch.Tensor = None) -> dict:
        if self.stopped:
            raise RuntimeError("Scheduler has already stopped. Call reset() first.")
        if video_feature.shape[0] != 1:
            raise ValueError(
                "HazardJumpScheduler V2 currently only supports batch_size=1 "
                f"(got {video_feature.shape[0]})."
            )

        outputs = self.scheduler_head(video_feature, sigma_cur=sigma_cur, context=context)
        delta_H_k = outputs.delta_H_k
        h_k = outputs.h_k

        if not hasattr(self, "H_cumulative_tensor"):
            self.H_cumulative_tensor = torch.zeros(
                delta_H_k.shape[0],
                dtype=delta_H_k.dtype,
                device=delta_H_k.device,
            )
        self.H_cumulative_tensor = self.H_cumulative_tensor + delta_H_k.squeeze(-1)
        self.H_cumulative = self.H_cumulative_tensor.mean().item()
        F_k = 1.0 - torch.exp(-self.H_cumulative_tensor.mean())

        forced_stop = bool(float(sigma_cur) < self.sigma_min or self.step_count >= self.K_max - 1)
        forced_continue = bool(self.step_count < self.K_min and not forced_stop)

        if forced_stop:
            should_stop = True
        elif forced_continue:
            should_stop = False
        elif self.mode == "train":
            h_k_mean = torch.clamp(torch.nan_to_num(h_k.mean(), nan=0.5), 1e-6, 1.0 - 1e-6)
            should_stop = bool(torch.bernoulli(h_k_mean).item())
        else:
            should_stop = bool((F_k >= self.eta).item())

        jump_ratio = None
        jump_distance = None
        sigma_next = None
        log_prob = None

        if should_stop:
            log_prob = self.scheduler_head.compute_action_logprob(
                h_k=h_k,
                stop_action=True,
                forced_stop=forced_stop,
                forced_continue=forced_continue,
                jump_logprob_scale=self.jump_logprob_scale,
            )
        else:
            deterministic_jump = self.mode != "train"
            jump_ratio = self.scheduler_head.sample_jump_ratio(
                outputs.alpha_k,
                outputs.beta_k,
                deterministic=deterministic_jump,
            )
            jump_distance = 1.0 - jump_ratio
            sigma_next = float((jump_ratio.squeeze() * float(sigma_cur)).item())
            sigma_next = max(0.0, min(float(sigma_next), float(sigma_cur) * (1.0 - self.scheduler_head.min_mode_eps)))
            jump_ratio = jump_ratio.squeeze(-1)
            jump_distance = jump_distance.squeeze(-1)
            self.jump_ratio_history.append(jump_ratio.detach())
            log_prob = self.scheduler_head.compute_action_logprob(
                h_k=h_k,
                stop_action=False,
                jump_ratio=jump_ratio.unsqueeze(-1),
                alpha_k=outputs.alpha_k,
                beta_k=outputs.beta_k,
                forced_stop=forced_stop,
                forced_continue=forced_continue,
                jump_logprob_scale=self.jump_logprob_scale,
            )

        self.delta_H_history.append(delta_H_k)
        self.h_k_history.append(h_k)
        self.step_count += 1
        if should_stop:
            self.stopped = True
            self.stop_step = self.step_count

        return {
            "delta_H_k": delta_H_k,
            "h_k": h_k,
            "F_k": float(F_k.item()),
            "should_stop": bool(should_stop),
            "forced_stop": forced_stop,
            "forced_continue": forced_continue,
            "log_prob": log_prob,
            "step_count": self.step_count,
            "H_cumulative": self.H_cumulative,
            "sigma_cur": float(sigma_cur),
            "sigma_next": sigma_next,
            "jump_ratio": None if jump_ratio is None else jump_ratio,
            "jump_distance": None if jump_distance is None else jump_distance,
            "alpha_k": outputs.alpha_k,
            "beta_k": outputs.beta_k,
            "jump_mode_k": outputs.jump_mode_k,
            "jump_concentration_k": outputs.jump_concentration_k,
        }
