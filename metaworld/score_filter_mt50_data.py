#!/usr/bin/env python3
"""Score and optionally prune low-quality MetaWorld MT50 trajectories."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT_ROOT = Path("metaworld/data/mt50_raw")


@dataclass
class EpisodeMetrics:
    env_name: str
    episode_index: int
    episode_dir: str
    num_actions: int
    total_reward: float
    hand_path_length: float
    hand_workspace_span: float
    hand_path_span_ratio: float
    hand_direct_distance: float
    hand_path_direct_ratio: float
    direction_reversal_rate: float
    idle_step_ratio: float
    gripper_flip_rate: float
    quality_score: float = 0.0
    quality_reasons: list[str] = field(default_factory=list)
    flagged: bool = False
    pruned: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score MetaWorld raw trajectories and optionally prune low-quality outliers."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing raw MT50 episodes.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated task names to score, or 'all'.",
    )
    parser.add_argument(
        "--exclude-tasks",
        type=str,
        default="",
        help="Comma-separated task names to exclude.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for quality reports. Defaults to <input-root>/quality_reports.",
    )
    parser.add_argument(
        "--reject-registry",
        type=Path,
        default=None,
        help="JSONL file to append rejected rand_vec records to. Defaults to <input-root>/rejected/reject_registry.jsonl.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=1.2,
        help="Combined quality score threshold above which an episode is considered low-quality.",
    )
    parser.add_argument(
        "--max-reject-ratio",
        type=float,
        default=0.2,
        help="Upper bound on the fraction of episodes pruned per task in one pass.",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=20,
        help="Minimum number of episodes required for per-task robust filtering.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete flagged episode directories and rewrite manifest.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score and report without deleting anything, even if --prune is set.",
    )
    parser.add_argument(
        "--print-top-k",
        type=int,
        default=5,
        help="How many worst episodes per task to print in the JSON summary.",
    )
    return parser.parse_args()


def parse_task_selection(task_arg: str, all_task_names: list[str]) -> list[str]:
    if task_arg == "all":
        return list(all_task_names)

    selected = [task.strip() for task in task_arg.split(",") if task.strip()]
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


def load_episode_metrics(episode_dir: Path) -> EpisodeMetrics:
    meta = json.loads((episode_dir / "episode_meta.json").read_text())
    summary = meta["summary"]
    arrays = np.load(episode_dir / "trajectory.npz")

    observations = arrays["observations"].astype(np.float32)
    actions = arrays["actions"].astype(np.float32)
    hand_positions = observations[:-1, :3] if len(observations) > 1 else observations[:, :3]
    hand_deltas = np.diff(hand_positions, axis=0) if len(hand_positions) > 1 else np.zeros((0, 3), dtype=np.float32)
    step_distances = np.linalg.norm(hand_deltas, axis=1) if len(hand_deltas) else np.zeros((0,), dtype=np.float32)

    hand_path_length = float(step_distances.sum())
    if len(hand_positions):
        workspace_span = float(np.linalg.norm(hand_positions.max(axis=0) - hand_positions.min(axis=0)))
        direct_distance = float(np.linalg.norm(hand_positions[-1] - hand_positions[0]))
    else:
        workspace_span = 0.0
        direct_distance = 0.0

    path_span_ratio = hand_path_length / max(workspace_span, 1e-3)
    path_direct_ratio = hand_path_length / max(direct_distance, 1e-3)

    direction_reversal_rate = compute_direction_reversal_rate(hand_deltas)
    idle_step_ratio = float(np.mean(step_distances < 5e-4)) if len(step_distances) else 0.0
    gripper_flip_rate = compute_gripper_flip_rate(actions[:, 3]) if len(actions) else 0.0

    return EpisodeMetrics(
        env_name=summary["env_name"],
        episode_index=int(summary["episode_index"]),
        episode_dir=str(episode_dir.resolve()),
        num_actions=int(summary["num_actions"]),
        total_reward=float(summary["total_reward"]),
        hand_path_length=hand_path_length,
        hand_workspace_span=workspace_span,
        hand_path_span_ratio=float(path_span_ratio),
        hand_direct_distance=direct_distance,
        hand_path_direct_ratio=float(path_direct_ratio),
        direction_reversal_rate=float(direction_reversal_rate),
        idle_step_ratio=float(idle_step_ratio),
        gripper_flip_rate=float(gripper_flip_rate),
    )


def compute_direction_reversal_rate(hand_deltas: np.ndarray) -> float:
    if len(hand_deltas) < 2:
        return 0.0

    step_distances = np.linalg.norm(hand_deltas, axis=1)
    significant = hand_deltas[step_distances > 5e-4]
    if len(significant) < 2:
        return 0.0

    norms = np.linalg.norm(significant, axis=1, keepdims=True)
    unit = significant / np.clip(norms, 1e-6, None)
    cosines = np.sum(unit[1:] * unit[:-1], axis=1)
    return float(np.mean(cosines < -0.1))


def compute_gripper_flip_rate(gripper_actions: np.ndarray) -> float:
    if len(gripper_actions) < 2:
        return 0.0

    diffs = np.diff(gripper_actions)
    significant = diffs[np.abs(diffs) > 0.1]
    if len(significant) < 2:
        return 0.0

    signs = np.sign(significant)
    return float(np.mean(signs[1:] * signs[:-1] < 0))


def robust_upper_stats(values: list[float], scale_floor: float) -> tuple[float, float]:
    if not values:
        return 0.0, scale_floor

    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    q75, q25 = np.quantile(arr, [0.75, 0.25])
    iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    scale = max(1.4826 * mad, iqr_scale, scale_floor)
    return median, scale


def score_task_episodes(
    task_name: str,
    episodes: list[EpisodeMetrics],
    score_threshold: float,
    max_reject_ratio: float,
) -> dict[str, Any]:
    metric_cfg = {
        "num_actions": {"scale_floor": 10.0, "weight": 0.4, "reason_threshold": 2.0},
        "hand_path_span_ratio": {"scale_floor": 0.25, "weight": 0.3, "reason_threshold": 2.0},
        "direction_reversal_rate": {"scale_floor": 0.05, "weight": 0.2, "reason_threshold": 2.0},
        "idle_step_ratio": {"scale_floor": 0.03, "weight": 0.1, "reason_threshold": 2.0},
    }

    robust_stats = {
        metric_name: robust_upper_stats(
            [float(getattr(ep, metric_name)) for ep in episodes],
            cfg["scale_floor"],
        )
        for metric_name, cfg in metric_cfg.items()
    }

    flagged: list[EpisodeMetrics] = []
    for episode in episodes:
        z_scores: dict[str, float] = {}
        reasons: list[str] = []
        score = 0.0
        for metric_name, cfg in metric_cfg.items():
            median, scale = robust_stats[metric_name]
            metric_value = float(getattr(episode, metric_name))
            z_value = max((metric_value - median) / max(scale, 1e-6), 0.0)
            z_scores[metric_name] = z_value
            score += cfg["weight"] * z_value
            if z_value >= cfg["reason_threshold"]:
                reasons.append(metric_name)

        episode.quality_score = float(score)
        episode.quality_reasons = reasons
        episode.flagged = bool(
            score >= score_threshold
            and (
                z_scores["hand_path_span_ratio"] >= 1.5
                or (
                    z_scores["num_actions"] >= 2.5
                    and max(
                        z_scores["hand_path_span_ratio"],
                        z_scores["direction_reversal_rate"],
                        z_scores["idle_step_ratio"],
                    )
                    >= 1.0
                )
                or (
                    z_scores["direction_reversal_rate"] >= 1.5
                    and z_scores["idle_step_ratio"] >= 0.5
                )
            )
        )
        if episode.flagged:
            flagged.append(episode)

    flagged.sort(key=lambda item: item.quality_score, reverse=True)
    max_reject_count = max(1, math.floor(len(episodes) * max_reject_ratio)) if flagged else 0
    pruned_candidates = flagged[:max_reject_count]
    for episode in pruned_candidates:
        episode.pruned = True

    return {
        "task_name": task_name,
        "episode_count": len(episodes),
        "flagged_count": len(flagged),
        "prune_count": len(pruned_candidates),
        "robust_stats": {
            metric_name: {"median": stats[0], "scale": stats[1]}
            for metric_name, stats in robust_stats.items()
        },
    }


def append_reject_registry(
    registry_path: Path,
    rejected_episodes: list[EpisodeMetrics],
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    if registry_path.exists():
        for line in registry_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            env_name = record.get("env_name")
            rand_vec = record.get("rand_vec")
            if env_name is None or rand_vec is None:
                continue
            existing_keys.add((env_name, tuple(round(float(value), 6) for value in rand_vec)))

    timestamp = datetime.now(timezone.utc).isoformat()
    with registry_path.open("a", encoding="utf-8") as handle:
        for episode in rejected_episodes:
            meta = json.loads((Path(episode.episode_dir) / "episode_meta.json").read_text())
            rand_vec = meta["task_payload"]["rand_vec"]
            key = (episode.env_name, tuple(round(float(value), 6) for value in rand_vec))
            if key in existing_keys:
                continue
            existing_keys.add(key)
            handle.write(
                json.dumps(
                    {
                        "env_name": episode.env_name,
                        "episode_index": episode.episode_index,
                        "episode_dir": episode.episode_dir,
                        "rand_vec": rand_vec,
                        "quality_score": episode.quality_score,
                        "quality_reasons": episode.quality_reasons,
                        "recorded_at": timestamp,
                    }
                )
                + "\n"
            )


def rewrite_manifest(input_root: Path) -> None:
    manifest_path = input_root / "manifest.jsonl"
    kept_lines = []
    for meta_path in sorted(input_root.glob("*/episode_*/episode_meta.json")):
        payload = json.loads(meta_path.read_text())
        kept_lines.append(json.dumps(payload["summary"]))

    if kept_lines:
        manifest_path.write_text("\n".join(kept_lines) + "\n")
    else:
        manifest_path.write_text("")


def prune_episodes(input_root: Path, rejected_episodes: list[EpisodeMetrics]) -> None:
    for episode in rejected_episodes:
        shutil.rmtree(Path(episode.episode_dir), ignore_errors=True)
    rewrite_manifest(input_root)


def write_reports(
    report_dir: Path,
    summaries: list[dict[str, Any]],
    all_episodes: list[EpisodeMetrics],
    print_top_k: int,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = report_dir / "quality_summary.json"
    episode_path = report_dir / "episode_metrics.jsonl"

    summary_payload = {
        "task_summaries": summaries,
        "printed_top_k": print_top_k,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    with episode_path.open("w", encoding="utf-8") as handle:
        for episode in sorted(all_episodes, key=lambda item: (item.env_name, item.episode_index)):
            handle.write(json.dumps(asdict(episode)) + "\n")


def build_task_list(input_root: Path, task_arg: str, exclude_arg: str) -> list[str]:
    all_task_names = sorted(
        path.name
        for path in input_root.iterdir()
        if path.is_dir() and any(path.glob("episode_*/episode_meta.json"))
    )
    selected = parse_task_selection(task_arg, all_task_names)
    return apply_task_exclusions(selected, exclude_arg)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else (input_root / "quality_reports").resolve()
    )
    reject_registry = (
        args.reject_registry.resolve()
        if args.reject_registry is not None
        else (input_root / "rejected" / "reject_registry.jsonl").resolve()
    )

    task_names = build_task_list(input_root, args.tasks, args.exclude_tasks)
    task_summaries: list[dict[str, Any]] = []
    all_episodes: list[EpisodeMetrics] = []
    rejected_episodes: list[EpisodeMetrics] = []

    for task_name in task_names:
        task_dir = input_root / task_name
        episode_dirs = sorted(path.parent for path in task_dir.glob("episode_*/episode_meta.json"))
        episodes = [load_episode_metrics(episode_dir) for episode_dir in episode_dirs]
        all_episodes.extend(episodes)

        if len(episodes) < args.min_episodes:
            task_summary = {
                "task_name": task_name,
                "episode_count": len(episodes),
                "flagged_count": 0,
                "prune_count": 0,
                "skipped": True,
                "skip_reason": f"episode_count < min_episodes ({args.min_episodes})",
            }
            task_summaries.append(task_summary)
            continue

        task_summary = score_task_episodes(
            task_name=task_name,
            episodes=episodes,
            score_threshold=args.score_threshold,
            max_reject_ratio=args.max_reject_ratio,
        )
        task_summary["skipped"] = False
        task_summary["worst_episodes"] = [
            {
                "episode_index": episode.episode_index,
                "quality_score": round(episode.quality_score, 4),
                "quality_reasons": episode.quality_reasons,
                "num_actions": episode.num_actions,
                "hand_path_span_ratio": round(episode.hand_path_span_ratio, 4),
                "direction_reversal_rate": round(episode.direction_reversal_rate, 4),
                "idle_step_ratio": round(episode.idle_step_ratio, 4),
            }
            for episode in sorted(episodes, key=lambda item: item.quality_score, reverse=True)[
                : args.print_top_k
            ]
        ]
        task_summaries.append(task_summary)
        rejected_episodes.extend([episode for episode in episodes if episode.pruned])

    write_reports(report_dir, task_summaries, all_episodes, args.print_top_k)

    if args.prune and not args.dry_run and rejected_episodes:
        append_reject_registry(reject_registry, rejected_episodes)
        prune_episodes(input_root, rejected_episodes)

    summary = {
        "input_root": str(input_root),
        "report_dir": str(report_dir),
        "reject_registry": str(reject_registry),
        "task_count": len(task_names),
        "episode_count": len(all_episodes),
        "prune_requested": bool(args.prune),
        "dry_run": bool(args.dry_run),
        "rejected_episode_count": len(rejected_episodes),
        "tasks": task_summaries,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
