#!/usr/bin/env python3
"""Collect scripted demonstrations for MetaWorld MT50."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")

import metaworld
from metaworld.policies import ENV_POLICY_MAP


DEFAULT_OUTPUT_ROOT = Path("metaworld/data/mt50_raw")
DEFAULT_CAMERA_NAMES = ("corner", "gripperPOV")
DEFAULT_CAMERA_KEYS = ("observation.images.main", "observation.images.wrist")


@dataclass
class EpisodeSummary:
    env_name: str
    episode_index: int
    goal_source_seed: int
    goal_source_index: int
    instruction: str
    success: bool
    success_step: int | None
    num_actions: int
    num_observations: int
    total_reward: float
    camera_names: list[str]
    camera_keys: list[str]
    width: int
    height: int
    video_fps: int
    benchmark_seed: int
    episode_seed: int
    video_paths: dict[str, str]
    npz_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect scripted MT50 rollouts and save raw videos plus arrays."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will contain the collected raw dataset.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated MT50 task names, or 'all'.",
    )
    parser.add_argument(
        "--exclude-tasks",
        type=str,
        default="",
        help="Comma-separated task names to skip after task selection.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=50,
        help="How many benchmark goals to collect for each task.",
    )
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        default=0,
        help="Seed used when MetaWorld samples the 50 goals for each MT50 task.",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        default=",".join(DEFAULT_CAMERA_NAMES),
        help="Comma-separated MetaWorld camera names to render for each episode.",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=",".join(DEFAULT_CAMERA_KEYS),
        help="Comma-separated logical camera keys written into metadata and file names.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Rendered frame width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Rendered frame height.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="FPS used when writing the demonstration video.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum actions per episode. Defaults to the environment's max_path_length.",
    )
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="Keep failed scripted rollouts instead of dropping them.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episodes whose metadata file already exists.",
    )
    parser.add_argument(
        "--max-goal-sets",
        type=int,
        default=10,
        help="How many MT50 goal sets to try per task while filling successful demos.",
    )
    parser.add_argument(
        "--max-attempts-per-task",
        type=int,
        default=None,
        help="Optional hard cap on rollout attempts for each task.",
    )
    parser.add_argument(
        "--reject-registry",
        type=Path,
        default=None,
        help="Optional JSONL file containing rejected rand_vec records to skip during recollection.",
    )
    return parser.parse_args()


def humanize_instruction(env_name: str) -> str:
    base_name = env_name.removesuffix("-v3")
    return " ".join(base_name.split("-"))


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


def parse_camera_lists(camera_names_arg: str, camera_keys_arg: str) -> tuple[list[str], list[str]]:
    camera_names = [name.strip() for name in camera_names_arg.split(",") if name.strip()]
    camera_keys = [key.strip() for key in camera_keys_arg.split(",") if key.strip()]
    if not camera_names:
        raise ValueError("At least one camera name is required.")
    if len(camera_names) != len(camera_keys):
        raise ValueError(
            "camera_names and camera_keys must have the same length. "
            f"Got {len(camera_names)} names and {len(camera_keys)} keys."
        )
    return camera_names, camera_keys


def group_train_tasks(benchmark: metaworld.Benchmark) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for task in benchmark.train_tasks:
        grouped[task.env_name].append(task)
    return grouped


def extract_task_payload(task: Any) -> dict[str, Any]:
    payload = pickle.loads(task.data)
    return {
        "env_name": task.env_name,
        "rand_vec": np.asarray(payload["rand_vec"], dtype=np.float32).tolist(),
        "partially_observable": bool(payload.get("partially_observable", False)),
    }


def render_frame(env: Any, camera_name: str) -> np.ndarray:
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"Camera '{camera_name}' was not found in the MetaWorld scene.")

    original_camera_id = env.mujoco_renderer.camera_id
    try:
        env.mujoco_renderer.camera_id = camera_id
        # MuJoCo offscreen rendering returns images with OpenGL's bottom-left origin.
        return np.flipud(np.asarray(env.render(), dtype=np.uint8))
    finally:
        env.mujoco_renderer.camera_id = original_camera_id


def rollout_episode(
    env: Any,
    policy: Any,
    task: Any,
    max_steps: int,
    episode_seed: int,
    camera_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]], bool, int | None, float]:
    env.set_task(task)
    obs, _ = env.reset(seed=episode_seed)

    observations = [np.asarray(obs, dtype=np.float32)]
    frames_by_camera = {
        camera_name: [render_frame(env, camera_name)] for camera_name in camera_names
    }
    actions = []
    rewards = []
    terminations = []
    truncations = []
    successes = []
    grasp_successes = []
    total_reward = 0.0
    success_step = None

    for step_idx in range(max_steps):
        action = np.asarray(policy.get_action(obs), dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action.astype(np.float32))
        rewards.append(np.float32(reward))
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))
        successes.append(np.float32(info.get("success", 0.0)))
        grasp_successes.append(np.float32(info.get("grasp_success", np.nan)))
        total_reward += float(reward)

        obs = next_obs
        observations.append(np.asarray(obs, dtype=np.float32))
        for camera_name in camera_names:
            frames_by_camera[camera_name].append(render_frame(env, camera_name))

        if info.get("success", 0.0) == 1.0 and success_step is None:
            success_step = step_idx

        if terminated or truncated or success_step is not None:
            break

    arrays = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminations, dtype=bool),
        "truncated": np.asarray(truncations, dtype=bool),
        "success": np.asarray(successes, dtype=np.float32),
        "grasp_success": np.asarray(grasp_successes, dtype=np.float32),
        "frame_step_indices": np.arange(len(observations), dtype=np.int32),
    }
    return arrays, frames_by_camera, success_step is not None, success_step, total_reward


def save_video(video_path: Path, frames: list[np.ndarray], fps: int) -> None:
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def save_episode(
    episode_dir: Path,
    arrays: dict[str, np.ndarray],
    frames_by_camera: dict[str, list[np.ndarray]],
    summary: EpisodeSummary,
    task_payload: dict[str, Any],
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)

    npz_path = episode_dir / "trajectory.npz"
    meta_path = episode_dir / "episode_meta.json"
    task_path = episode_dir / "task_payload.pkl"

    for camera_key, frames in frames_by_camera.items():
        save_video(episode_dir / f"{camera_key}.mp4", frames, summary.video_fps)
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "summary": asdict(summary),
        "task_payload": task_payload,
    }
    meta["summary"]["video_paths"] = {
        camera_key: str(episode_dir / f"{camera_key}.mp4")
        for camera_key in summary.camera_keys
    }
    meta["summary"]["npz_path"] = str(npz_path)

    meta_path.write_text(json.dumps(meta, indent=2))
    task_path.write_bytes(pickle.dumps(task_payload))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_resume_state(task_dir: Path) -> tuple[int, set[tuple[float, ...]], set[int]]:
    saved_count = 0
    seen_rand_vecs: set[tuple[float, ...]] = set()
    existing_episode_indices: set[int] = set()

    for meta_path in sorted(task_dir.glob("episode_*/episode_meta.json")):
        payload = json.loads(meta_path.read_text())
        rand_vec = payload.get("task_payload", {}).get("rand_vec")
        if rand_vec is not None:
            seen_rand_vecs.add(tuple(round(float(value), 6) for value in rand_vec))
        episode_dir = meta_path.parent.name
        try:
            existing_episode_indices.add(int(episode_dir.removeprefix("episode_")))
        except ValueError:
            pass
        saved_count += 1

    return saved_count, seen_rand_vecs, existing_episode_indices


def load_reject_state(reject_registry: Path | None) -> dict[str, set[tuple[float, ...]]]:
    rejected: dict[str, set[tuple[float, ...]]] = defaultdict(set)
    if reject_registry is None or not reject_registry.exists():
        return rejected

    for line in reject_registry.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        env_name = record.get("env_name")
        rand_vec = record.get("rand_vec")
        if env_name is None or rand_vec is None:
            continue
        rejected[env_name].add(tuple(round(float(value), 6) for value in rand_vec))
    return rejected


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    camera_names, camera_keys = parse_camera_lists(args.camera_names, args.camera_keys)
    reject_registry = args.reject_registry.resolve() if args.reject_registry else None
    rejected_rand_vecs = load_reject_state(reject_registry)

    benchmark = metaworld.MT50(seed=args.benchmark_seed)
    all_task_names = list(benchmark.train_classes.keys())
    selected_tasks = parse_task_selection(args.tasks, all_task_names)
    selected_tasks = apply_task_exclusions(selected_tasks, args.exclude_tasks)

    config_payload = {
        "benchmark": "MT50",
        "benchmark_seed": args.benchmark_seed,
        "episodes_per_task": args.episodes_per_task,
        "tasks": selected_tasks,
        "exclude_tasks": [task.strip() for task in args.exclude_tasks.split(",") if task.strip()],
        "camera_names": camera_names,
        "camera_keys": camera_keys,
        "width": args.width,
        "height": args.height,
        "video_fps": args.video_fps,
        "max_steps": args.max_steps,
        "save_failures": args.save_failures,
        "resume": args.resume,
        "max_goal_sets": args.max_goal_sets,
        "max_attempts_per_task": args.max_attempts_per_task,
        "reject_registry": str(reject_registry) if reject_registry is not None else None,
    }
    write_json(output_root / "collection_config.json", config_payload)

    manifest_path = output_root / "manifest.jsonl"
    task_summaries: dict[str, dict[str, int]] = {
        task_name: {
            "requested": args.episodes_per_task,
            "saved": 0,
            "skipped": 0,
            "failed": 0,
            "attempted": 0,
            "goal_sets_used": 0,
        }
        for task_name in selected_tasks
    }

    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        for task_name in selected_tasks:
            env_cls = benchmark.train_classes[task_name]
            env = env_cls(
                render_mode="rgb_array",
                camera_name=camera_names[0],
                width=args.width,
                height=args.height,
            )
            policy = ENV_POLICY_MAP[task_name]()
            max_steps = args.max_steps or env.max_path_length
            task_dir = output_root / task_name
            if args.resume:
                saved_count, seen_rand_vecs, existing_episode_indices = load_resume_state(task_dir)
                task_summaries[task_name]["saved"] = saved_count
                task_summaries[task_name]["skipped"] = saved_count
            else:
                saved_count = 0
                seen_rand_vecs = set()
                existing_episode_indices = set()
            seen_rand_vecs.update(rejected_rand_vecs.get(task_name, set()))

            progress = tqdm(total=args.episodes_per_task, desc=f"Collect {task_name}", leave=False)
            progress.update(saved_count)
            progress.set_postfix(
                saved=saved_count,
                attempted=task_summaries[task_name]["attempted"],
                failed=task_summaries[task_name]["failed"],
                goal_set=task_summaries[task_name]["goal_sets_used"],
            )

            goal_set_offset = 0
            while saved_count < args.episodes_per_task and goal_set_offset < args.max_goal_sets:
                goal_seed = args.benchmark_seed + goal_set_offset
                current_benchmark = metaworld.MT50(seed=goal_seed)
                available_goals = group_train_tasks(current_benchmark)[task_name]
                task_summaries[task_name]["goal_sets_used"] = goal_set_offset + 1
                progress.set_postfix(
                    saved=saved_count,
                    attempted=task_summaries[task_name]["attempted"],
                    failed=task_summaries[task_name]["failed"],
                    goal_set=task_summaries[task_name]["goal_sets_used"],
                )

                for goal_index, task in enumerate(available_goals):
                    if (
                        args.max_attempts_per_task is not None
                        and task_summaries[task_name]["attempted"] >= args.max_attempts_per_task
                    ):
                        break
                    task_payload = extract_task_payload(task)
                    rand_key = tuple(round(float(value), 6) for value in task_payload["rand_vec"])
                    if rand_key in seen_rand_vecs:
                        continue
                    seen_rand_vecs.add(rand_key)

                    episode_index = 0
                    while episode_index in existing_episode_indices:
                        episode_index += 1
                    episode_dir = task_dir / f"episode_{episode_index:03d}"
                    meta_path = episode_dir / "episode_meta.json"

                    episode_seed = goal_seed * 100_000 + goal_index
                    arrays, frames_by_camera, success, success_step, total_reward = rollout_episode(
                        env=env,
                        policy=policy,
                        task=task,
                        max_steps=max_steps,
                        episode_seed=episode_seed,
                        camera_names=camera_names,
                    )
                    task_summaries[task_name]["attempted"] += 1
                    progress.set_postfix(
                        saved=saved_count,
                        attempted=task_summaries[task_name]["attempted"],
                        failed=task_summaries[task_name]["failed"],
                        goal_set=task_summaries[task_name]["goal_sets_used"],
                    )

                    if not success and not args.save_failures:
                        task_summaries[task_name]["failed"] += 1
                        progress.set_postfix(
                            saved=saved_count,
                            attempted=task_summaries[task_name]["attempted"],
                            failed=task_summaries[task_name]["failed"],
                            goal_set=task_summaries[task_name]["goal_sets_used"],
                        )
                        continue

                    summary = EpisodeSummary(
                        env_name=task_name,
                        episode_index=episode_index,
                        goal_source_seed=goal_seed,
                        goal_source_index=goal_index,
                        instruction=humanize_instruction(task_name),
                        success=success,
                        success_step=success_step,
                        num_actions=int(arrays["actions"].shape[0]),
                        num_observations=int(arrays["observations"].shape[0]),
                        total_reward=total_reward,
                        camera_names=camera_names,
                        camera_keys=camera_keys,
                        width=args.width,
                        height=args.height,
                        video_fps=args.video_fps,
                        benchmark_seed=goal_seed,
                        episode_seed=episode_seed,
                        video_paths={
                            camera_key: str(episode_dir / f"{camera_key}.mp4")
                            for camera_key in camera_keys
                        },
                        npz_path=str(episode_dir / "trajectory.npz"),
                    )
                    save_episode(
                        episode_dir=episode_dir,
                        arrays=arrays,
                        frames_by_camera={
                            camera_key: frames_by_camera[camera_name]
                            for camera_name, camera_key in zip(camera_names, camera_keys)
                        },
                        summary=summary,
                        task_payload=task_payload,
                    )
                    manifest_file.write(json.dumps(asdict(summary)) + "\n")
                    manifest_file.flush()
                    saved_count += 1
                    existing_episode_indices.add(episode_index)
                    task_summaries[task_name]["saved"] = saved_count
                    progress.update(1)
                    progress.set_postfix(
                        saved=saved_count,
                        attempted=task_summaries[task_name]["attempted"],
                        failed=task_summaries[task_name]["failed"],
                        goal_set=task_summaries[task_name]["goal_sets_used"],
                    )

                    if saved_count >= args.episodes_per_task:
                        break

                if (
                    args.max_attempts_per_task is not None
                    and task_summaries[task_name]["attempted"] >= args.max_attempts_per_task
                ):
                    break

                goal_set_offset += 1

            progress.close()

            env.close()

    total_saved = sum(item["saved"] for item in task_summaries.values())
    total_failed = sum(item["failed"] for item in task_summaries.values())
    total_skipped = sum(item["skipped"] for item in task_summaries.values())
    summary_payload = {
        "benchmark": "MT50",
        "benchmark_seed": args.benchmark_seed,
        "total_saved": total_saved,
        "total_failed": total_failed,
        "total_skipped": total_skipped,
        "tasks": task_summaries,
    }
    write_json(output_root / "collection_summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
