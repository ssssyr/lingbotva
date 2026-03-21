# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """
    TODO
    """
    if eval_mode:
        model.eval().requires_grad_(False)

    if dist.is_initialized():
        dist_strategy = os.environ.get("LINGBOT_VA_DIST_STRATEGY", "fsdp")
        if dist_strategy == "ddp":
            find_unused_parameters = (
                os.environ.get("LINGBOT_VA_DDP_FIND_UNUSED", "0") == "1"
            )
            gradient_as_bucket_view = (
                os.environ.get("LINGBOT_VA_DDP_GRAD_BUCKET_VIEW", "1") == "1"
            )
            static_graph = (
                os.environ.get("LINGBOT_VA_DDP_STATIC_GRAPH", "0") == "1"
                and not find_unused_parameters
            )
            model.to(param_dtype)
            model.to(device)
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=gradient_as_bucket_view,
                static_graph=static_graph,
            )
        else:
            model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size <= 1:
        return
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            rank=rank,
                            world_size=world_size,
                            device_id=device)

def dist_mean(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
