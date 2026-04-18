#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute UR10 action stats with optional relative tcp.xyz anchoring."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of the LeRobot dataset containing meta/ and data/.",
    )
    parser.add_argument(
        "--representation",
        choices=("absolute", "relative_chunk_anchor"),
        default="relative_chunk_anchor",
        help="Action representation used before computing quantiles.",
    )
    parser.add_argument(
        "--config-action-dim",
        type=int,
        default=30,
        help="Expanded action dimension used by LingBot-VA configs.",
    )
    parser.add_argument(
        "--config-gripper-channel",
        type=int,
        default=28,
        help="Channel index used for gripper.pos in the LingBot-VA config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"info.json not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_episodes(dataset_root: Path) -> list[dict]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_path}")
    with episodes_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_named_indices(action_names: list[str]) -> tuple[int, int, int, int]:
    required = ("tcp.x", "tcp.y", "tcp.z", "gripper.pos")
    indices = []
    for name in required:
        if name not in action_names:
            raise ValueError(
                f"Action feature {name!r} not found. Available names: {action_names}"
            )
        indices.append(action_names.index(name))
    return tuple(indices)


def load_episode_actions(parquet_path: Path) -> np.ndarray:
    table = pq.ParquetFile(parquet_path).read(columns=["action"])
    return np.asarray(table["action"].to_pylist(), dtype=np.float32)


def build_segment_actions(
    dataset_root: Path,
    episodes: list[dict],
    xyz_indices: tuple[int, int, int],
    representation: str,
) -> np.ndarray:
    data_root = dataset_root / "data"
    cache: dict[int, np.ndarray] = {}
    segments: list[np.ndarray] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if episode_index not in cache:
            parquet_path = next(data_root.rglob(f"episode_{episode_index:06d}.parquet"))
            cache[episode_index] = load_episode_actions(parquet_path)

        episode_actions = cache[episode_index]
        for action_cfg in episode.get("action_config", []):
            start = int(action_cfg["start_frame"])
            end = int(action_cfg["end_frame"])
            segment = episode_actions[start:end].copy()
            if len(segment) == 0:
                continue
            if representation == "relative_chunk_anchor":
                segment[:, xyz_indices] -= segment[:1, xyz_indices]
            segments.append(segment)

    if not segments:
        raise ValueError("No action segments found in episodes.jsonl")
    return np.concatenate(segments, axis=0)


def compute_stats(action: np.ndarray) -> dict[str, list[float]]:
    quantiles = {
        "min": 0.0,
        "q01": 0.01,
        "q10": 0.10,
        "q50": 0.50,
        "q90": 0.90,
        "q99": 0.99,
        "max": 1.0,
    }
    stats = {
        "mean": action.mean(axis=0).tolist(),
        "std": action.std(axis=0).tolist(),
        "count": [int(action.shape[0])],
    }
    for name, q in quantiles.items():
        stats[name] = np.quantile(action, q, axis=0).tolist()
    return stats


def build_config_norm_stat(
    stats: dict[str, list[float]],
    config_action_dim: int,
    config_gripper_channel: int,
    x_idx: int,
    y_idx: int,
    z_idx: int,
    gripper_idx: int,
) -> dict[str, list[float]]:
    q01 = [0.0] * config_action_dim
    q99 = [0.0] * config_action_dim
    q01[0] = float(stats["q01"][x_idx])
    q01[1] = float(stats["q01"][y_idx])
    q01[2] = float(stats["q01"][z_idx])
    q01[config_gripper_channel] = float(stats["q01"][gripper_idx])
    q99[0] = float(stats["q99"][x_idx])
    q99[1] = float(stats["q99"][y_idx])
    q99[2] = float(stats["q99"][z_idx])
    q99[config_gripper_channel] = float(stats["q99"][gripper_idx])
    return {"q01": q01, "q99": q99}


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    info = load_info(dataset_root)
    episodes = load_episodes(dataset_root)

    action_names = info["features"]["action"]["names"]
    x_idx, y_idx, z_idx, gripper_idx = find_named_indices(action_names)
    action = build_segment_actions(
        dataset_root=dataset_root,
        episodes=episodes,
        xyz_indices=(x_idx, y_idx, z_idx),
        representation=args.representation,
    )
    stats = compute_stats(action)
    config_norm_stat = build_config_norm_stat(
        stats=stats,
        config_action_dim=args.config_action_dim,
        config_gripper_channel=args.config_gripper_channel,
        x_idx=x_idx,
        y_idx=y_idx,
        z_idx=z_idx,
        gripper_idx=gripper_idx,
    )

    result = {
        "dataset_root": str(dataset_root),
        "representation": args.representation,
        "action_names": action_names,
        "stats": stats,
        "config_norm_stat": config_norm_stat,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
