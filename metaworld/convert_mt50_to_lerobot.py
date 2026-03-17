#!/usr/bin/env python3
"""Convert cleaned MetaWorld MT50 rollouts into a local LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "pyarrow is required for MT50 -> LeRobot conversion. "
        "Install the missing LeRobot dataset dependencies first, e.g. "
        "`pip install 'datasets>=2.19.0,<=3.6.0' 'jsonlines>=4.0.0' pyarrow`."
    ) from exc


CODEBASE_VERSION = "v2.1"
DEFAULT_INPUT_ROOT = Path("metaworld/data/mt50_raw")
DEFAULT_OUTPUT_ROOT = Path("metaworld/data/mt50_lerobot")
DEFAULT_CAMERA_KEYS = ("observation.images.main", "observation.images.wrist")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_ACTION_NAMES = ("dx", "dy", "dz", "gripper")
DEFAULT_ROBOT_TYPE = "metaworld-mt50-single-arm"


@dataclass(frozen=True)
class EpisodeSource:
    task_name: str
    task_text: str
    episode_dir: Path
    arrays_path: Path
    video_paths: dict[str, Path]
    num_frames: int
    width: int
    height: int
    fps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert cleaned MetaWorld MT50 raw episodes into LeRobot v2.1 format."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--exclude-tasks", type=str, default="")
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=",".join(DEFAULT_CAMERA_KEYS),
        help="Comma-separated camera keys to expose as LeRobot videos.",
    )
    parser.add_argument(
        "--task-text-source",
        type=str,
        choices=("instruction", "env_name"),
        default="instruction",
        help="Which raw metadata field to use as LeRobot task/action text.",
    )
    parser.add_argument(
        "--video-mode",
        type=str,
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize source mp4 files into the LeRobot videos/ tree.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override dataset fps. Defaults to the shared raw MT50 video_fps value.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=DEFAULT_ROBOT_TYPE,
        help="robot_type field written into meta/info.json.",
    )
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="Include episodes marked as unsuccessful in episode_meta.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing output directory before converting.",
    )
    return parser.parse_args()


def parse_task_selection(task_arg: str, all_task_names: list[str]) -> list[str]:
    if task_arg == "all":
        return list(all_task_names)
    selected = [task.strip() for task in task_arg.split(",") if task.strip()]
    unknown = sorted(set(selected) - set(all_task_names))
    if unknown:
        raise ValueError(f"Unknown MT50 task names: {', '.join(unknown)}")
    return selected


def apply_task_exclusions(task_names: list[str], exclude_arg: str) -> list[str]:
    excluded = [task.strip() for task in exclude_arg.split(",") if task.strip()]
    if not excluded:
        return task_names
    unknown = sorted(set(excluded) - set(task_names))
    if unknown:
        raise ValueError(f"Unknown excluded task names: {', '.join(unknown)}")
    excluded_set = set(excluded)
    return [task for task in task_names if task not in excluded_set]


def parse_camera_keys(camera_keys_arg: str) -> list[str]:
    camera_keys = [key.strip() for key in camera_keys_arg.split(",") if key.strip()]
    if not camera_keys:
        raise ValueError("At least one camera key is required.")
    return camera_keys


def humanize_task_name(task_name: str) -> str:
    return task_name.removesuffix("-v3").replace("-", " ")


def serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serialize_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [serialize_value(inner) for inner in value]
    return value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialize_value(row), ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialize_value(payload), indent=2, ensure_ascii=False))


def numeric_feature_stats(array: np.ndarray) -> dict[str, np.ndarray]:
    keepdims = array.ndim == 1
    return {
        "min": np.min(array, axis=0, keepdims=keepdims),
        "max": np.max(array, axis=0, keepdims=keepdims),
        "mean": np.mean(array, axis=0, keepdims=keepdims),
        "std": np.std(array, axis=0, keepdims=keepdims),
        "count": np.array([len(array)], dtype=np.int64),
    }


def build_features(
    state_dim: int,
    action_dim: int,
    camera_keys: list[str],
    width: int,
    height: int,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"obs_{i:03d}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": list(DEFAULT_ACTION_NAMES[:action_dim])
            if action_dim <= len(DEFAULT_ACTION_NAMES)
            else [f"action_{i:03d}" for i in range(action_dim)],
        },
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    for camera_key in camera_keys:
        features[camera_key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def create_empty_dataset_info(
    fps: int,
    features: dict[str, dict[str, Any]],
    robot_type: str,
) -> dict[str, Any]:
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def materialize_video(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_episode_parquet(
    path: Path,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    episode_indices: np.ndarray,
    global_indices: np.ndarray,
    task_indices: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(global_indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
            "observation.state": pa.array(
                states.tolist(),
                type=pa.list_(pa.float32(), list_size=states.shape[1]),
            ),
            "action": pa.array(
                actions.tolist(),
                type=pa.list_(pa.float32(), list_size=actions.shape[1]),
            ),
        }
    )
    pq.write_table(table, path)


def collect_episode_sources(
    input_root: Path,
    task_names: list[str],
    camera_keys: list[str],
    task_text_source: str,
    allow_failed: bool,
) -> tuple[list[EpisodeSource], int, int, int]:
    episodes: list[EpisodeSource] = []
    inferred_fps: int | None = None
    inferred_width: int | None = None
    inferred_height: int | None = None

    for task_name in task_names:
        task_dir = input_root / task_name
        for meta_path in sorted(task_dir.glob("episode_*/episode_meta.json")):
            payload = json.loads(meta_path.read_text())
            summary = payload["summary"]
            if not summary.get("success", False) and not allow_failed:
                continue

            arrays_path = meta_path.parent / "trajectory.npz"
            if not arrays_path.exists():
                raise FileNotFoundError(f"Missing trajectory file: {arrays_path}")

            arrays = np.load(arrays_path)
            num_actions = int(arrays["actions"].shape[0])
            if num_actions <= 0:
                continue

            summary_camera_keys = list(summary["camera_keys"])
            missing_keys = [key for key in camera_keys if key not in summary_camera_keys]
            if missing_keys:
                raise ValueError(
                    f"Episode {meta_path.parent} does not contain requested camera keys: {missing_keys}"
                )

            fps = int(summary["video_fps"])
            width = int(summary["width"])
            height = int(summary["height"])
            if inferred_fps is None:
                inferred_fps = fps
                inferred_width = width
                inferred_height = height
            else:
                if fps != inferred_fps:
                    raise ValueError(f"Inconsistent fps: {fps} vs {inferred_fps}")
                if width != inferred_width or height != inferred_height:
                    raise ValueError(
                        f"Inconsistent frame size in {meta_path.parent}: {(width, height)} vs {(inferred_width, inferred_height)}"
                    )

            task_text = summary["instruction"] if task_text_source == "instruction" else summary["env_name"]
            task_text = task_text.strip() or humanize_task_name(summary["env_name"])

            video_paths = {}
            for camera_key in camera_keys:
                video_path = Path(summary["video_paths"][camera_key])
                if not video_path.exists():
                    video_path = meta_path.parent / f"{camera_key}.mp4"
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing source video for {camera_key}: {video_path}")
                video_paths[camera_key] = video_path.resolve()

            episodes.append(
                EpisodeSource(
                    task_name=task_name,
                    task_text=task_text,
                    episode_dir=meta_path.parent.resolve(),
                    arrays_path=arrays_path.resolve(),
                    video_paths=video_paths,
                    num_frames=num_actions,
                    width=width,
                    height=height,
                    fps=fps,
                )
            )

    if not episodes:
        raise ValueError(f"No convertible episodes found under {input_root}")

    assert inferred_fps is not None
    assert inferred_width is not None
    assert inferred_height is not None
    return episodes, inferred_fps, inferred_width, inferred_height


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    camera_keys = parse_camera_keys(args.camera_keys)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    all_task_names = sorted(
        path.name
        for path in input_root.iterdir()
        if path.is_dir() and any(path.glob("episode_*/episode_meta.json"))
    )
    task_names = parse_task_selection(args.tasks, all_task_names)
    task_names = apply_task_exclusions(task_names, args.exclude_tasks)

    if output_root.exists():
        if args.force:
            shutil.rmtree(output_root)
        elif any(output_root.iterdir()):
            raise FileExistsError(
                f"Output root already exists and is not empty: {output_root}. "
                "Use --force to recreate it."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    episodes, inferred_fps, width, height = collect_episode_sources(
        input_root=input_root,
        task_names=task_names,
        camera_keys=camera_keys,
        task_text_source=args.task_text_source,
        allow_failed=args.allow_failed,
    )

    dataset_fps = args.fps if args.fps is not None else inferred_fps

    first_arrays = np.load(episodes[0].arrays_path)
    first_actions = np.asarray(first_arrays["actions"], dtype=np.float32)
    first_observations = np.asarray(first_arrays["observations"], dtype=np.float32)
    state_dim = int(first_observations.shape[1])
    action_dim = int(first_actions.shape[1])

    features = build_features(
        state_dim=state_dim,
        action_dim=action_dim,
        camera_keys=camera_keys,
        width=width,
        height=height,
    )
    info = create_empty_dataset_info(dataset_fps, features, args.robot_type)

    task_to_index: dict[str, int] = {}
    for episode in episodes:
        if episode.task_text not in task_to_index:
            task_to_index[episode.task_text] = len(task_to_index)

    meta_dir = output_root / "meta"
    data_dir = output_root / "data"
    videos_dir = output_root / "videos"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for episode_index, episode in enumerate(episodes):
        arrays = np.load(episode.arrays_path)
        actions = np.asarray(arrays["actions"], dtype=np.float32)
        observations = np.asarray(arrays["observations"], dtype=np.float32)
        if observations.shape[0] == actions.shape[0] + 1:
            states = observations[:-1]
        elif observations.shape[0] == actions.shape[0]:
            states = observations
        else:
            raise ValueError(
                f"Episode {episode.episode_dir} has unsupported observation/action lengths: "
                f"{observations.shape[0]} observations vs {actions.shape[0]} actions"
            )

        if states.shape[1] != state_dim or actions.shape[1] != action_dim:
            raise ValueError(
                f"Episode {episode.episode_dir} changed tensor dimensions: "
                f"state {states.shape[1]} vs {state_dim}, action {actions.shape[1]} vs {action_dim}"
            )

        chunk_index = episode_index // DEFAULT_CHUNK_SIZE
        frame_indices = np.arange(len(actions), dtype=np.int64)
        timestamps = frame_indices.astype(np.float32) / float(dataset_fps)
        episode_indices = np.full((len(actions),), episode_index, dtype=np.int64)
        global_indices = np.arange(total_frames, total_frames + len(actions), dtype=np.int64)
        task_indices = np.full((len(actions),), task_to_index[episode.task_text], dtype=np.int64)

        parquet_path = data_dir / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"
        write_episode_parquet(
            path=parquet_path,
            timestamps=timestamps,
            frame_indices=frame_indices,
            episode_indices=episode_indices,
            global_indices=global_indices,
            task_indices=task_indices,
            states=states,
            actions=actions,
        )

        for camera_key, src_video_path in episode.video_paths.items():
            video_path = (
                videos_dir
                / f"chunk-{chunk_index:03d}"
                / camera_key
                / f"episode_{episode_index:06d}.mp4"
            )
            materialize_video(src_video_path, video_path, args.video_mode)

        episode_row = {
            "episode_index": episode_index,
            "tasks": [episode.task_text],
            "length": int(len(actions)),
            "action_config": [
                {
                    "start_frame": 0,
                    "end_frame": int(len(actions)),
                    "action_text": episode.task_text,
                }
            ],
            "source_env_name": episode.task_name,
            "source_episode_dir": str(episode.episode_dir),
        }
        append_jsonl(meta_dir / "episodes.jsonl", episode_row)

        stats_row = {
            "episode_index": episode_index,
            "stats": {
                "timestamp": numeric_feature_stats(timestamps),
                "frame_index": numeric_feature_stats(frame_indices),
                "episode_index": numeric_feature_stats(episode_indices),
                "index": numeric_feature_stats(global_indices),
                "task_index": numeric_feature_stats(task_indices),
                "observation.state": numeric_feature_stats(states),
                "action": numeric_feature_stats(actions),
            },
        }
        append_jsonl(meta_dir / "episodes_stats.jsonl", stats_row)
        total_frames += len(actions)

    for task_text, task_index in sorted(task_to_index.items(), key=lambda item: item[1]):
        append_jsonl(meta_dir / "tasks.jsonl", {"task_index": task_index, "task": task_text})

    info["total_episodes"] = len(episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(task_to_index)
    info["total_videos"] = len(episodes) * len(camera_keys)
    info["total_chunks"] = math.ceil(len(episodes) / DEFAULT_CHUNK_SIZE)
    info["splits"] = {"train": f"0:{len(episodes)}"}
    write_json(meta_dir / "info.json", info)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "tasks": task_names,
        "camera_keys": camera_keys,
        "episode_count": len(episodes),
        "total_frames": total_frames,
        "fps": dataset_fps,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "video_mode": args.video_mode,
        "robot_type": args.robot_type,
        "task_to_index": task_to_index,
    }
    write_json(meta_dir / "conversion_summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
