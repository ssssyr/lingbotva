#!/usr/bin/env python3
"""Extract Wan VAE latents and empty text embedding for a local MT50 LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from wan_va.modules.utils import load_text_encoder, load_tokenizer, load_vae


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_DATASET_ROOT = Path("metaworld/data/mt50_lerobot")
DEFAULT_DTYPE = "bfloat16"


@dataclass
class SegmentJob:
    episode_index: int
    episode_chunk: int
    camera_key: str
    video_path: Path
    start_frame: int
    end_frame: int
    action_text: str
    ori_fps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Wan VAE latents and empty_emb.pt for a LeRobot-formatted MT50 dataset."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path containing Wan model subfolders: vae/, tokenizer/, text_encoder/.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated natural-language task names from meta/tasks.jsonl, or 'all'.",
    )
    parser.add_argument("--exclude-tasks", type=str, default="")
    parser.add_argument(
        "--camera-keys",
        type=str,
        default="",
        help="Comma-separated LeRobot video keys to encode. Defaults to all video keys in meta/info.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("bfloat16", "float16", "float32"),
        default=DEFAULT_DTYPE,
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Optional downsample fps. Must evenly divide the dataset fps.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Max tokenizer sequence length for Wan text encoder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode existing latent files and overwrite empty_emb.pt if it already exists.",
    )
    parser.add_argument(
        "--skip-empty-emb",
        action="store_true",
        help="Do not write empty_emb.pt.",
    )
    return parser.parse_args()


def parse_task_selection(task_arg: str, all_task_names: list[str]) -> list[str]:
    if task_arg == "all":
        return list(all_task_names)
    selected = [task.strip() for task in task_arg.split(",") if task.strip()]
    unknown = sorted(set(selected) - set(all_task_names))
    if unknown:
        raise ValueError(f"Unknown tasks: {', '.join(unknown)}")
    return selected


def apply_task_exclusions(task_names: list[str], exclude_arg: str) -> list[str]:
    excluded = [task.strip() for task in exclude_arg.split(",") if task.strip()]
    if not excluded:
        return task_names
    unknown = sorted(set(excluded) - set(task_names))
    if unknown:
        raise ValueError(f"Unknown excluded tasks: {', '.join(unknown)}")
    excluded_set = set(excluded)
    return [task for task in task_names if task not in excluded_set]


def parse_camera_keys(camera_keys_arg: str, default_keys: list[str]) -> list[str]:
    if not camera_keys_arg.strip():
        return list(default_keys)
    camera_keys = [key.strip() for key in camera_keys_arg.split(",") if key.strip()]
    unknown = sorted(set(camera_keys) - set(default_keys))
    if unknown:
        raise ValueError(f"Unknown camera keys: {', '.join(unknown)}")
    return camera_keys


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        resolved = model_path.resolve()
    else:
        env_value = os.environ.get("LINGBOT_VA_MODEL_PATH")
        if not env_value:
            raise ValueError(
                "--model-path is required when LINGBOT_VA_MODEL_PATH is not set."
            )
        resolved = Path(env_value).expanduser().resolve()
    for subdir in ("vae", "tokenizer", "text_encoder"):
        if not (resolved / subdir).exists():
            raise FileNotFoundError(f"Missing model subdirectory: {resolved / subdir}")
    return resolved


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    return json.loads(info_path.read_text())


def load_tasks(task_path: Path) -> dict[int, str]:
    tasks = {}
    with task_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    return tasks


def iter_episode_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def read_video_segment(video_path: Path, start_frame: int, end_frame: int, stride: int) -> torch.Tensor:
    reader = imageio.get_reader(str(video_path))
    frames = []
    try:
        for frame_idx, frame in enumerate(reader):
            if frame_idx < start_frame:
                continue
            if frame_idx >= end_frame:
                break
            if (frame_idx - start_frame) % stride != 0:
                continue
            frames.append(torch.from_numpy(frame))
    finally:
        reader.close()
    if not frames:
        raise ValueError(
            f"No frames sampled from {video_path} with range [{start_frame}, {end_frame}) and stride={stride}"
        )
    video = torch.stack(frames, dim=0).permute(3, 0, 1, 2).float() / 255.0
    return video * 2.0 - 1.0


def normalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    latents_std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    return (latents.float() - latents_mean) * (1.0 / latents_std)


def encode_text(
    tokenizer,
    text_encoder,
    prompt: str,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> torch.Tensor:
    cleaned = prompt_clean(prompt)
    text_inputs = tokenizer(
        [cleaned],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask
    encoder_device = next(text_encoder.parameters()).device
    with torch.no_grad():
        prompt_embeds = text_encoder(
            input_ids.to(encoder_device),
            attention_mask.to(encoder_device),
        ).last_hidden_state
    return prompt_embeds[0].to(dtype=dtype).cpu()


def build_segment_jobs(
    dataset_root: Path,
    info: dict[str, Any],
    selected_tasks: list[str],
    camera_keys: list[str],
) -> tuple[list[SegmentJob], int]:
    episodes = iter_episode_rows(dataset_root / "meta" / "episodes.jsonl")
    tasks_by_index = load_tasks(dataset_root / "meta" / "tasks.jsonl")
    fps = int(info["fps"])
    jobs: list[SegmentJob] = []

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        episode_chunk = episode_index // int(info["chunks_size"])
        task_name = episode["tasks"][0]
        if task_name not in selected_tasks:
            continue
        for action_cfg in episode["action_config"]:
            start_frame = int(action_cfg["start_frame"])
            end_frame = int(action_cfg["end_frame"])
            action_text = action_cfg.get("action_text") or task_name
            for camera_key in camera_keys:
                video_rel = info["video_path"].format(
                    episode_chunk=episode_chunk,
                    video_key=camera_key,
                    episode_index=episode_index,
                )
                video_path = dataset_root / video_rel
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing video file: {video_path}")
                jobs.append(
                    SegmentJob(
                        episode_index=episode_index,
                        episode_chunk=episode_chunk,
                        camera_key=camera_key,
                        video_path=video_path,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        action_text=action_text,
                        ori_fps=fps,
                    )
                )
    return jobs, fps


def latent_output_path(dataset_root: Path, job: SegmentJob) -> Path:
    return (
        dataset_root
        / "latents"
        / f"chunk-{job.episode_chunk:03d}"
        / job.camera_key
        / f"episode_{job.episode_index:06d}_{job.start_frame}_{job.end_frame}.pth"
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    model_root = resolve_model_path(args.model_path)
    info = load_info(dataset_root)
    all_tasks = [row["task"] for row in iter_episode_rows(dataset_root / "meta" / "tasks.jsonl")]
    selected_tasks = parse_task_selection(args.tasks, all_tasks)
    selected_tasks = apply_task_exclusions(selected_tasks, args.exclude_tasks)
    camera_keys = parse_camera_keys(args.camera_keys, list(info["features"].keys()))
    camera_keys = [key for key in camera_keys if info["features"].get(key, {}).get("dtype") == "video"]
    if not camera_keys:
        raise ValueError("No video camera keys selected for latent extraction.")

    jobs, ori_fps = build_segment_jobs(dataset_root, info, selected_tasks, camera_keys)
    if args.target_fps is None:
        target_fps = ori_fps
    else:
        target_fps = int(args.target_fps)
        if target_fps <= 0:
            raise ValueError("--target-fps must be positive.")
        if ori_fps % target_fps != 0:
            raise ValueError(f"target_fps={target_fps} must evenly divide dataset fps={ori_fps}.")
    frame_stride = ori_fps // target_fps

    device = torch.device(args.device)
    torch_dtype = get_torch_dtype(args.dtype)

    vae = load_vae(model_root / "vae", torch_dtype=torch_dtype, torch_device=device)
    vae.eval()
    tokenizer = load_tokenizer(model_root / "tokenizer")
    text_encoder = load_text_encoder(
        model_root / "text_encoder",
        torch_dtype=torch.float32,
        torch_device="cpu",
    )
    text_encoder.eval()

    if not args.skip_empty_emb:
        empty_path = dataset_root / "empty_emb.pt"
        if args.overwrite or not empty_path.exists():
            empty_emb = encode_text(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt="",
                dtype=torch_dtype,
                max_sequence_length=args.max_sequence_length,
            )
            torch.save(empty_emb.cpu(), empty_path)

    text_cache: dict[str, torch.Tensor] = {}
    encoded_segments = 0
    skipped_segments = 0

    with torch.no_grad():
        progress = tqdm(jobs, desc="Extract latents")
        for job in progress:
            output_path = latent_output_path(dataset_root, job)
            if output_path.exists() and not args.overwrite:
                skipped_segments += 1
                continue

            video = read_video_segment(
                video_path=job.video_path,
                start_frame=job.start_frame,
                end_frame=job.end_frame,
                stride=frame_stride,
            ).unsqueeze(0)
            if video.shape[-2] != info["features"][job.camera_key]["shape"][0] or video.shape[-1] != info["features"][job.camera_key]["shape"][1]:
                raise ValueError(
                    f"Unexpected frame size in {job.video_path}: {tuple(video.shape[-2:])}"
                )

            video = video.to(device=device, dtype=torch_dtype)
            mu = vae.encode(video).latent_dist.mean
            mu_norm = normalize_latents(mu, vae)

            latent = rearrange(mu_norm[0].detach().cpu().to(torch.bfloat16), "c f h w -> (f h w) c").contiguous()
            latent_num_frames = int(mu_norm.shape[2])
            latent_height = int(mu_norm.shape[3])
            latent_width = int(mu_norm.shape[4])

            action_text = job.action_text.strip()
            if action_text not in text_cache:
                text_cache[action_text] = encode_text(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=action_text,
                    dtype=torch_dtype,
                    max_sequence_length=args.max_sequence_length,
                )

            frame_ids = list(range(job.start_frame, job.end_frame, frame_stride))
            payload = {
                "latent": latent,
                "latent_num_frames": latent_num_frames,
                "latent_height": latent_height,
                "latent_width": latent_width,
                "video_num_frames": len(frame_ids),
                "video_height": int(video.shape[-2]),
                "video_width": int(video.shape[-1]),
                "text_emb": text_cache[action_text].clone().to(torch.bfloat16),
                "text": action_text,
                "frame_ids": frame_ids,
                "start_frame": job.start_frame,
                "end_frame": job.end_frame,
                "fps": target_fps,
                "ori_fps": job.ori_fps,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, output_path)
            encoded_segments += 1

    summary = {
        "dataset_root": str(dataset_root),
        "model_root": str(model_root),
        "selected_tasks": selected_tasks,
        "camera_keys": camera_keys,
        "jobs_total": len(jobs),
        "encoded_segments": encoded_segments,
        "skipped_segments": skipped_segments,
        "target_fps": target_fps,
        "ori_fps": ori_fps,
        "frame_stride": frame_stride,
        "empty_emb_written": (not args.skip_empty_emb),
        "dtype": args.dtype,
        "device": str(device),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
