# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import os

import torch
try:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
except ImportError:
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    if os.environ.get("LINGBOT_VA_ENABLE_AC", "1") != "1":
        return
    for layer_id, transformer_block in enumerate(model.blocks):
        transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.blocks[layer_id] = transformer_block


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}
    sharding_layout = os.environ.get("LINGBOT_VA_FSDP_LAYOUT", "block")

    for block in model.blocks:
        if sharding_layout == "nested":
            fully_shard(block.attn1, **fsdp_config)
            fully_shard(block.attn2, **fsdp_config)
            fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
