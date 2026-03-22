# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

from typing import Any, Dict, List

import torch
from einops import rearrange
from tqdm import tqdm

from .scheduler import FlowMatchScheduler
from .utils import data_seq_to_patch, get_mesh_id


class CutoffScanner(object):

    def __init__(
        self,
        transformer,
        config,
        device,
        dtype,
        cache_name="pos",
    ):
        self.transformer = transformer
        self.config = config
        self.device = device
        self.dtype = dtype
        self.cache_name = cache_name

        self.patch_size = tuple(config.patch_size)
        self.action_dim = int(config.action_dim)
        self.action_per_frame = int(config.action_per_frame)
        self.num_video_steps = int(config.num_inference_steps)
        self.num_action_steps = int(config.action_num_inference_steps)

        self.video_scheduler = FlowMatchScheduler(
            shift=float(config.snr_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.video_scheduler.set_timesteps(self.num_video_steps)
        self.video_timesteps = self.video_scheduler.timesteps.clone()

        self.action_scheduler = FlowMatchScheduler(
            shift=float(config.action_snr_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler.set_timesteps(self.num_action_steps)
        action_timesteps = self.action_scheduler.timesteps
        self.action_timesteps_with_terminal = torch.nn.functional.pad(
            action_timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )

        self.action_mask = torch.zeros([self.action_dim], dtype=torch.bool, device=self.device)
        self.action_mask[config.used_action_channel_ids] = True

        self.q01 = torch.tensor(config.norm_stat["q01"], dtype=torch.float32, device=self.device).view(
            1, -1, 1, 1, 1
        )
        self.q99 = torch.tensor(config.norm_stat["q99"], dtype=torch.float32, device=self.device).view(
            1, -1, 1, 1, 1
        )

        self.transformer.eval()

    def _init_cache(self, latents, actions):
        patch_f, patch_h, patch_w = self.patch_size
        _, _, f_lat, h_lat, w_lat = latents.shape
        _, _, f_act, n_act, _ = actions.shape

        latent_token_per_chunk = (f_lat * h_lat * w_lat) // (patch_f * patch_h * patch_w)
        action_token_per_chunk = f_act * n_act

        self.transformer.clear_cache(self.cache_name)
        self.transformer.create_empty_cache(
            self.cache_name,
            int(self.config.attn_window),
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=self.dtype,
            device=self.device,
            batch_size=latents.shape[0],
        )

    def _prepare_input(
        self,
        noisy_latents,
        text_emb,
        timestep,
        action_mode,
        cond,
        frame_st_id=0,
    ):
        if isinstance(timestep, torch.Tensor):
            timestep_value = float(timestep.item())
        else:
            timestep_value = float(timestep)

        if action_mode:
            grid_id = get_mesh_id(
                noisy_latents.shape[-3],
                noisy_latents.shape[-2],
                noisy_latents.shape[-1],
                1,
                1,
                frame_st_id,
                action=True,
            ).to(self.device)
        else:
            grid_id = get_mesh_id(
                noisy_latents.shape[-3] // self.patch_size[0],
                noisy_latents.shape[-2] // self.patch_size[1],
                noisy_latents.shape[-1] // self.patch_size[2],
                0,
                1,
                frame_st_id,
            ).to(self.device)

        timesteps = torch.ones([noisy_latents.shape[2]], dtype=torch.float32, device=self.device)
        timesteps = timesteps * timestep_value

        model_input = {
            "noisy_latents": noisy_latents,
            "timesteps": timesteps[None],
            "grid_id": grid_id[None],
            "text_emb": text_emb,
        }

        if cond is not None:
            model_input["noisy_latents"] = model_input["noisy_latents"].clone()
            model_input["timesteps"] = model_input["timesteps"].clone()
            model_input["noisy_latents"][:, :, 0:1] = cond[:, :, 0:1]
            model_input["timesteps"][:, 0:1] *= 0

        if action_mode:
            model_input["noisy_latents"] = model_input["noisy_latents"].clone()
            model_input["noisy_latents"][:, ~self.action_mask] *= 0

        return model_input

    def _run_video_to_cutoff(self, video_noise, text_emb, latent_cond, cutoff_idx):
        latents = video_noise.clone()

        for step_idx in range(cutoff_idx):
            t = self.video_timesteps[step_idx]
            input_dict = self._prepare_input(
                latents,
                text_emb,
                t,
                action_mode=False,
                cond=latent_cond,
            )
            video_noise_pred = self.transformer(
                input_dict,
                update_cache=0,
                cache_name=self.cache_name,
                action_mode=False,
            )
            video_noise_pred = data_seq_to_patch(
                self.patch_size,
                video_noise_pred,
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
                batch_size=latents.shape[0],
            )
            latents = self.video_scheduler.step(
                video_noise_pred,
                t,
                latents,
                return_dict=False,
            )
            latents[:, :, 0:1] = latent_cond[:, :, 0:1]

        return latents

    def _refresh_video_cache_exact(self, latents, text_emb, latent_cond, cutoff_idx):
        if cutoff_idx < self.num_video_steps:
            refresh_t = self.video_timesteps[cutoff_idx]
        else:
            refresh_t = 0.0

        input_dict = self._prepare_input(
            latents,
            text_emb,
            refresh_t,
            action_mode=False,
            cond=latent_cond,
        )
        self.transformer(
            input_dict,
            update_cache=1,
            cache_name=self.cache_name,
            action_mode=False,
        )

    def _run_action_from_cache(self, action_noise, text_emb):
        actions = action_noise.clone()
        action_cond = torch.zeros(
            [
                actions.shape[0],
                self.action_dim,
                1,
                self.action_per_frame,
                1,
            ],
            device=self.device,
            dtype=self.dtype,
        )

        for i, t in enumerate(self.action_timesteps_with_terminal):
            last_step = i == len(self.action_timesteps_with_terminal) - 1
            input_dict = self._prepare_input(
                actions,
                text_emb,
                t,
                action_mode=True,
                cond=action_cond,
            )
            action_noise_pred = self.transformer(
                input_dict,
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=True,
            )

            if not last_step:
                action_noise_pred = rearrange(
                    action_noise_pred,
                    "b (f n) c -> b c f n 1",
                    f=actions.shape[2],
                )
                actions = self.action_scheduler.step(
                    action_noise_pred,
                    t,
                    actions,
                    return_dict=False,
                )

            actions[:, :, 0:1] = action_cond[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0
        return actions

    def predict_actions_for_cutoff(
        self,
        text_emb,
        latent_cond,
        video_noise,
        action_noise,
        cutoff_idx,
    ):
        if cutoff_idx < 1 or cutoff_idx > self.num_video_steps:
            raise ValueError(
                "cutoff_idx must be in [1, {}], got {}.".format(self.num_video_steps, cutoff_idx)
            )

        self._init_cache(video_noise, action_noise)
        latents = self._run_video_to_cutoff(video_noise, text_emb, latent_cond, cutoff_idx)
        self._refresh_video_cache_exact(latents, text_emb, latent_cond, cutoff_idx)
        return self._run_action_from_cache(action_noise, text_emb)

    def compute_action_metrics(self, pred_actions, target_actions, target_mask):
        mask = target_mask.float()
        denom = mask.sum().clamp_min(1.0)

        mse = ((pred_actions.float() - target_actions.float()) ** 2 * mask).sum() / denom

        # Frame 0 is explicitly conditioned to zeros during action generation,
        # so the first meaningful comparison should use the first later frame
        # that still has valid supervision.
        first_pred_numer = pred_actions.new_tensor(0.0, dtype=torch.float32)
        first_pred_denom = pred_actions.new_tensor(0.0, dtype=torch.float32)
        frame_valid = mask.sum(dim=(1, 3, 4)) > 0
        for batch_idx in range(mask.shape[0]):
            valid_frames = torch.nonzero(frame_valid[batch_idx], as_tuple=False).flatten()
            valid_frames = valid_frames[valid_frames > 0]
            if valid_frames.numel() == 0:
                continue
            first_frame_idx = int(valid_frames[0].item())
            frame_mask = mask[batch_idx : batch_idx + 1, :, first_frame_idx:first_frame_idx + 1]
            frame_sq_err = (
                pred_actions[batch_idx : batch_idx + 1, :, first_frame_idx:first_frame_idx + 1].float()
                - target_actions[batch_idx : batch_idx + 1, :, first_frame_idx:first_frame_idx + 1].float()
            ) ** 2
            first_pred_numer = first_pred_numer + (frame_sq_err * frame_mask).sum()
            first_pred_denom = first_pred_denom + frame_mask.sum()
        if float(first_pred_denom.item()) > 0:
            first_pred_mse = first_pred_numer / first_pred_denom
            first_pred_mse_value = float(first_pred_mse.item())
        else:
            first_pred_mse_value = float('nan')

        pred_denorm = (pred_actions.float() + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01
        target_denorm = (target_actions.float() + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01
        denorm_l1 = (torch.abs(pred_denorm - target_denorm) * mask).sum() / denom

        return {
            "masked_action_mse": float(mse.item()),
            "first_pred_action_mse": first_pred_mse_value,
            "denorm_action_l1": float(denorm_l1.item()),
        }


def summarize_cutoff_metrics(metric_table):
    rows = []
    for cutoff_idx in sorted(metric_table.keys()):
        metrics = metric_table[cutoff_idx]
        row = {
            "cutoff_idx": int(cutoff_idx),
            "num_states": int(len(metrics["masked_action_mse"])),
        }
        for key, values in metrics.items():
            if len(values) == 0:
                row["{}_mean".format(key)] = float("nan")
                row["{}_std".format(key)] = float("nan")
                continue
            tensor_vals = torch.tensor(values, dtype=torch.float32)
            row["{}_mean".format(key)] = float(tensor_vals.mean().item())
            row["{}_std".format(key)] = float(tensor_vals.std(unbiased=False).item())
        rows.append(row)
    return rows


def scan_cutoff_curve(
    scanner,
    dataloader,
    cutoff_indices,
    max_samples,
    seed,
    show_progress=True,
):
    metric_table = {
        int(idx): {
            "masked_action_mse": [],
            "first_pred_action_mse": [],
            "denorm_action_l1": [],
        }
        for idx in cutoff_indices
    }

    per_state_records = []
    scanned = 0

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="cutoff-scan", leave=False)

    for batch in iterator:
        if max_samples > 0 and scanned >= max_samples:
            break

        latents = batch["latents"].to(device=scanner.device, dtype=scanner.dtype)
        actions = batch["actions"].to(device=scanner.device, dtype=scanner.dtype)
        actions_mask = batch["actions_mask"].to(device=scanner.device)
        text_emb = batch["text_emb"].to(device=scanner.device, dtype=scanner.dtype)

        global_index = _batch_to_int(batch.get("global_index"), default=scanned)
        sample_seed = int(seed + global_index * 1000003 + 17)

        video_noise = _seeded_noise_like(latents, sample_seed, scanner.device, scanner.dtype)
        action_noise = _seeded_noise_like(actions, sample_seed + 1, scanner.device, scanner.dtype)

        latent_cond = latents[:, :, 0:1].to(scanner.dtype)

        state_record = {
            "global_index": int(global_index),
            "sample_index": _batch_to_int(batch.get("sample_index"), default=-1),
            "source_dataset_id": _batch_to_int(batch.get("source_dataset_id"), default=-1),
            "episode_index": _batch_to_int(batch.get("episode_index"), default=-1),
            "local_start_frame": _batch_to_int(batch.get("local_start_frame"), default=-1),
            "local_end_frame": _batch_to_int(batch.get("local_end_frame"), default=-1),
            "dataset_repo_id": _batch_to_str(batch.get("dataset_repo_id"), default=""),
            "state_uid": _batch_to_str(batch.get("state_uid"), default="global_{}".format(global_index)),
            "metrics_by_cutoff": {},
        }

        for cutoff_idx in cutoff_indices:
            pred_actions = scanner.predict_actions_for_cutoff(
                text_emb=text_emb,
                latent_cond=latent_cond,
                video_noise=video_noise,
                action_noise=action_noise,
                cutoff_idx=cutoff_idx,
            )
            metrics = scanner.compute_action_metrics(pred_actions, actions, actions_mask)
            for key, value in metrics.items():
                metric_table[cutoff_idx][key].append(float(value))
            state_record["metrics_by_cutoff"][str(cutoff_idx)] = metrics

        per_state_records.append(state_record)
        scanned += 1

    return {
        "metric_table": metric_table,
        "per_state_records": per_state_records,
        "num_states": scanned,
    }


def _seeded_noise_like(x, seed, device, dtype):
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=cpu_generator, dtype=torch.float32)
    return noise.to(device=device, dtype=dtype)


def _batch_to_int(value, default):
    if value is None:
        return int(default)
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return int(default)
        return _batch_to_int(value[0], default=default)
    return int(value)


def _batch_to_str(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return _batch_to_str(value[0], default=default)
    return str(value)
