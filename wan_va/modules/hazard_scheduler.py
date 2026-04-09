# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

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
