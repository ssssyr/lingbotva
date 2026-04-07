# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wan_va.modules.hazard_scheduler import HazardSchedulerHead, HazardScheduler


class TestHazardSchedulerHead:
    """Test suite for HazardSchedulerHead module."""

    def test_initialization(self):
        """Test that the module initializes correctly."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)

        assert scheduler_head.hidden_dim == 768
        assert scheduler_head.output_dim == 1
        assert not scheduler_head.use_context

    def test_forward_pass(self):
        """Test forward pass with valid inputs."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)

        batch_size = 4
        video_feature = torch.randn(batch_size, 768)
        step_id = torch.tensor([0.0, 0.25, 0.5, 0.75])

        delta_H_k = scheduler_head(video_feature, step_id)

        # Check output shape
        assert delta_H_k.shape == (batch_size, 1)

        # Check non-negativity
        assert (delta_H_k >= 0).all(), "delta_H_k must be non-negative"

    def test_step_id_broadcasting(self):
        """Test that step_id is correctly broadcasted."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)

        video_feature = torch.randn(4, 768)

        # Test 1D step_id
        step_id_1d = torch.tensor([0.0, 0.25, 0.5, 0.75])
        delta_H_k_1d = scheduler_head(video_feature, step_id_1d)

        # Test 2D step_id
        step_id_2d = step_id_1d.unsqueeze(-1)
        delta_H_k_2d = scheduler_head(video_feature, step_id_2d)

        # Should produce same results
        assert torch.allclose(delta_H_k_1d, delta_H_k_2d, atol=1e-6)

    def test_hazard_monotonicity(self):
        """Test that cumulative Hazard is monotonically increasing."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)

        batch_size = 4
        video_feature = torch.randn(batch_size, 768)

        H_cumulative = torch.zeros(batch_size, 1)

        for step_idx in range(10):
            step_id = torch.tensor([step_idx / 10.0]).expand(batch_size, 1)
            delta_H_k = scheduler_head(video_feature, step_id)

            # Check non-negativity
            assert (delta_H_k >= 0).all()

            # Update cumulative Hazard
            H_prev = H_cumulative.clone()
            H_cumulative += delta_H_k

            # Check monotonicity
            assert (H_cumulative >= H_prev).all(), "Cumulative Hazard must be monotonically increasing"

    def test_stop_probability_range(self):
        """Test that stopping probability is in [0, 1]."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)

        # Test various delta_H_k values
        delta_H_k = torch.tensor([[0.0], [0.1], [0.5], [1.0], [2.0], [5.0]])
        h_k = scheduler_head.compute_stop_probability(delta_H_k)

        # Check range
        assert (h_k >= 0).all() and (h_k <= 1).all(), "h_k must be in [0, 1]"

        # Check monotonicity: larger delta_H_k -> larger h_k
        for i in range(len(h_k) - 1):
            assert h_k[i] <= h_k[i + 1], "h_k should increase with delta_H_k"

    def test_survival_probability(self):
        """Test survival probability computation."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)

        H_k = torch.tensor([[0.0], [0.5], [1.0], [2.0], [5.0]])
        S_k = scheduler_head.compute_survival_probability(H_k)

        # Check range
        assert (S_k >= 0).all() and (S_k <= 1).all(), "S_k must be in [0, 1]"

        # Check monotonicity: larger H_k -> smaller S_k
        for i in range(len(S_k) - 1):
            assert S_k[i] >= S_k[i + 1], "S_k should decrease with H_k"

        # Check relationship: S_k = exp(-H_k)
        expected_S_k = torch.exp(-H_k)
        assert torch.allclose(S_k, expected_S_k, atol=1e-6)

    def test_with_context(self):
        """Test forward pass with context features."""
        scheduler_head = HazardSchedulerHead(
            hidden_dim=768,
            use_context=True,
            context_dim=128,
        )

        batch_size = 4
        video_feature = torch.randn(batch_size, 768)
        step_id = torch.tensor([0.0, 0.25, 0.5, 0.75])
        context = torch.randn(batch_size, 128)

        delta_H_k = scheduler_head(video_feature, step_id, context)

        # Check output shape
        assert delta_H_k.shape == (batch_size, 1)

        # Check non-negativity
        assert (delta_H_k >= 0).all()

    def test_context_required_error(self):
        """Test that error is raised when context is required but not provided."""
        scheduler_head = HazardSchedulerHead(
            hidden_dim=768,
            use_context=True,
            context_dim=128,
        )

        video_feature = torch.randn(4, 768)
        step_id = torch.tensor([0.0, 0.25, 0.5, 0.75])

        with pytest.raises(ValueError, match="context is required"):
            scheduler_head(video_feature, step_id, context=None)

    def test_initial_hazard_small(self):
        """Test that initial Hazard is small (avoiding premature stopping)."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)

        video_feature = torch.randn(4, 768)
        step_id = torch.zeros(4, 1)  # First step

        delta_H_k = scheduler_head(video_feature, step_id)

        # Initial delta_H_k should be small (< 0.5)
        assert (delta_H_k < 0.5).all(), "Initial delta_H_k should be small to avoid premature stopping"


class TestHazardScheduler:
    """Test suite for HazardScheduler class."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)
        scheduler = HazardScheduler(
            scheduler_head=scheduler_head,
            K_max=25,
            K_min=3,
            eta=0.5,
            mode='train',
        )

        assert scheduler.K_max == 25
        assert scheduler.K_min == 3
        assert scheduler.eta == 0.5
        assert scheduler.mode == 'train'
        assert scheduler.step_count == 0
        assert not scheduler.stopped

    def test_reset(self):
        """Test that reset clears scheduler state."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)
        scheduler = HazardScheduler(scheduler_head, K_max=25)

        # Simulate some steps
        video_feature = torch.randn(1, 768)
        scheduler.step(video_feature)
        scheduler.step(video_feature)

        assert scheduler.step_count == 2

        # Reset
        scheduler.reset()

        assert scheduler.step_count == 0
        assert scheduler.H_cumulative == 0.0
        assert len(scheduler.delta_H_history) == 0
        assert not scheduler.stopped

    def test_step_output_format(self):
        """Test that step() returns correct format."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)
        scheduler = HazardScheduler(scheduler_head, K_max=25, mode='train')

        video_feature = torch.randn(1, 768)
        result = scheduler.step(video_feature)

        # Check keys
        assert 'delta_H_k' in result
        assert 'h_k' in result
        assert 'should_stop' in result
        assert 'log_prob' in result
        assert 'step_count' in result
        assert 'H_cumulative' in result

        # Check types
        assert isinstance(result['delta_H_k'], torch.Tensor)
        assert isinstance(result['h_k'], torch.Tensor)
        assert isinstance(result['should_stop'], bool)
        assert result['log_prob'] is None or isinstance(result['log_prob'], torch.Tensor)

    def test_k_min_enforcement(self):
        """Test that stopping is not allowed before K_min."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)
        scheduler = HazardScheduler(scheduler_head, K_max=25, K_min=3, mode='train')

        video_feature = torch.randn(1, 768)

        # First K_min steps should never stop
        for _ in range(3):
            result = scheduler.step(video_feature)
            assert not result['should_stop'], "Should not stop before K_min"

    def test_k_max_enforcement(self):
        """Test that stopping is forced at K_max."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)
        scheduler = HazardScheduler(scheduler_head, K_max=5, K_min=1, mode='train')

        video_feature = torch.randn(1, 768)

        # Run until K_max
        for _ in range(4):
            result = scheduler.step(video_feature)

        # Last step should force stop
        result = scheduler.step(video_feature)
        assert result['should_stop'], "Should force stop at K_max"
        assert scheduler.stopped

    def test_deterministic_eval_mode(self):
        """Test deterministic stopping in eval mode."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)
        scheduler = HazardScheduler(
            scheduler_head,
            K_max=25,
            K_min=3,
            eta=0.5,
            mode='eval',
        )

        video_feature = torch.randn(1, 768)

        # Run multiple steps
        stopped = False
        for _ in range(25):
            result = scheduler.step(video_feature)
            if result['should_stop']:
                stopped = True
                break

        assert stopped, "Should eventually stop in eval mode"
        assert result['log_prob'] is None, "log_prob not needed in eval mode"

    def test_stochastic_train_mode(self):
        """Test stochastic sampling in train mode."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)
        scheduler = HazardScheduler(scheduler_head, K_max=25, K_min=3, mode='train')

        video_feature = torch.randn(1, 768)

        # Run multiple steps
        stopped = False
        for _ in range(25):
            result = scheduler.step(video_feature)
            if result['should_stop']:
                stopped = True
                assert result['log_prob'] is not None, "log_prob required in train mode"
                break

        assert stopped, "Should eventually stop in train mode"

    def test_trajectory_recording(self):
        """Test that trajectory is correctly recorded."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)
        scheduler = HazardScheduler(scheduler_head, K_max=10, K_min=1, mode='train')

        video_feature = torch.randn(1, 768)

        # Run until stop
        for _ in range(10):
            result = scheduler.step(video_feature)
            if result['should_stop']:
                break

        trajectory = scheduler.get_trajectory()

        # Check trajectory format
        assert 'delta_H_history' in trajectory
        assert 'h_k_history' in trajectory
        assert 'stop_step' in trajectory
        assert 'H_cumulative' in trajectory

        # Check lengths match
        assert len(trajectory['delta_H_history']) == trajectory['stop_step']
        assert len(trajectory['h_k_history']) == trajectory['stop_step']

    def test_error_after_stop(self):
        """Test that error is raised if step() is called after stopping."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)
        scheduler = HazardScheduler(scheduler_head, K_max=5, K_min=1, mode='train')

        video_feature = torch.randn(1, 768)

        # Run until stop
        for _ in range(5):
            result = scheduler.step(video_feature)
            if result['should_stop']:
                break

        # Try to step again
        with pytest.raises(RuntimeError, match="already stopped"):
            scheduler.step(video_feature)

    def test_cumulative_hazard_increases(self):
        """Test that cumulative Hazard increases over time."""
        scheduler_head = HazardSchedulerHead(hidden_dim=768)
        scheduler = HazardScheduler(scheduler_head, K_max=10, K_min=1, mode='eval')

        video_feature = torch.randn(1, 768)

        prev_H = 0.0
        for _ in range(5):
            result = scheduler.step(video_feature)
            current_H = result['H_cumulative']

            assert current_H >= prev_H, "Cumulative Hazard must increase"
            prev_H = current_H


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
