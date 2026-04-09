from typing import Dict

import torch


def prepare_chunked_rollout_inputs(
    batch,
    device,
    dtype,
    frame_chunk_size: int,
    sample_history: bool = True,
) -> Dict[str, torch.Tensor]:
    """Slice one history chunk plus one target chunk for hazard rollouts."""
    latents = batch["latents"].to(device, dtype=dtype)
    actions = batch["actions"].to(device, dtype=dtype)
    actions_mask = batch["actions_mask"].to(device=device)
    text_emb = batch["text_emb"].to(device, dtype=dtype)
    total_latent_frames = int(latents.shape[2])

    if total_latent_frames > 2 * frame_chunk_size:
        max_history_start = total_latent_frames - 2 * frame_chunk_size
        if sample_history:
            history_start = int(torch.randint(0, max_history_start + 1, (1,)).item())
        else:
            history_start = 0
    else:
        history_start = 0

    history_end = min(history_start + frame_chunk_size, total_latent_frames - frame_chunk_size)
    target_start = history_end
    target_end = min(target_start + frame_chunk_size, total_latent_frames)
    if target_end <= target_start:
        target_start = max(0, total_latent_frames - frame_chunk_size)
        target_end = total_latent_frames
        history_start = max(0, target_start - frame_chunk_size)
        history_end = target_start

    clean_history_latents = latents[:, :, history_start:history_end].contiguous()
    clean_history_actions = actions[:, :, history_start:history_end].contiguous()
    target_latents = latents[:, :, target_start:target_end].contiguous()
    target_actions = actions[:, :, target_start:target_end].contiguous()
    target_actions_mask = actions_mask[:, :, target_start:target_end].contiguous()

    if clean_history_latents.shape[2] == 0:
        clean_history_latents = None
        clean_history_actions = None
        latent_cond = target_latents[:, :, 0:1].clone()
        action_cond = torch.zeros_like(target_actions[:, :, 0:1])
    else:
        latent_cond = None
        action_cond = None

    return {
        "video_noise": torch.randn_like(target_latents),
        "action_noise": torch.randn_like(target_actions),
        "text_emb": text_emb,
        "gt_action": target_actions,
        "gt_action_mask": target_actions_mask,
        "clean_history_latents": clean_history_latents,
        "clean_history_actions": clean_history_actions,
        "latent_cond": latent_cond,
        "action_cond": action_cond,
        "segment_length": int(
            (batch["local_end_frame"] - batch["local_start_frame"]).flatten()[0].item()
        ) if "local_end_frame" in batch and "local_start_frame" in batch else None,
        "history_start_frame": history_start,
        "history_end_frame": history_end,
        "target_start_frame": target_start,
        "target_end_frame": target_end,
    }
