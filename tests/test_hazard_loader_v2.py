# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wan_va.modules.hazard_scheduler import HazardJumpSchedulerHead
from wan_va.utils.hazard_loader import load_hazard_scheduler_head


class DummyTransformer:
    def __init__(self, hidden_size=32):
        self.config = SimpleNamespace(hidden_size=hidden_size)


def make_config():
    return SimpleNamespace(
        model=SimpleNamespace(hazard_hidden_dim=16),
        hazard=SimpleNamespace(
            policy_variant="stop_jump_v2",
            heuristic_jump_mode=0.75,
            heuristic_jump_concentration=10.0,
        ),
    )


def test_loader_restores_v2_checkpoint():
    transformer = DummyTransformer(hidden_size=32)
    config = make_config()
    head = HazardJumpSchedulerHead(
        feature_dim=32,
        hidden_dim=16,
        init_jump_mode=0.75,
        init_jump_concentration=10.0,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "hazard_v2.pt"
        torch.save(
            {
                "step": 3,
                "schema_version": "hazard_scheduler_v2",
                "policy_variant": "stop_jump_v2",
                "scheduler_head_state_dict": head.state_dict(),
            },
            ckpt_path,
        )

        restored_head, checkpoint = load_hazard_scheduler_head(
            transformer=transformer,
            config=config,
            checkpoint_path=str(ckpt_path),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert type(restored_head).__name__ == "HazardJumpSchedulerHead"
        assert checkpoint["policy_variant"] == "stop_jump_v2"

        original_state = head.state_dict()
        restored_state = restored_head.state_dict()
        for key in original_state.keys():
            assert torch.allclose(original_state[key], restored_state[key])
