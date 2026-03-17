#!/usr/bin/env python3
"""Collect MT50 data, score quality, prune bad episodes, and refill automatically."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("metaworld/data/mt50_raw")
DEFAULT_COLLECT_SCRIPT = Path("metaworld/collect_mt50_data.py")
DEFAULT_QUALITY_SCRIPT = Path("metaworld/score_filter_mt50_data.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run collection -> quality filter -> refill loops for MetaWorld MT50."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--exclude-tasks", type=str, default="")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--collect-script", type=Path, default=DEFAULT_COLLECT_SCRIPT)
    parser.add_argument("--quality-script", type=Path, default=DEFAULT_QUALITY_SCRIPT)
    parser.add_argument("--benchmark-seed", type=int, default=0)
    parser.add_argument("--camera-names", type=str, default="corner,gripperPOV")
    parser.add_argument(
        "--camera-keys",
        type=str,
        default="observation.images.main,observation.images.wrist",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--collect-max-goal-sets", type=int, default=10)
    parser.add_argument("--collect-max-attempts-per-task", type=int, default=150)
    parser.add_argument("--quality-min-episodes", type=int, default=20)
    parser.add_argument("--quality-score-threshold", type=float, default=1.2)
    parser.add_argument("--quality-max-reject-ratio", type=float, default=0.2)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--reject-registry", type=Path, default=None)
    parser.add_argument("--dry-run-quality", action="store_true")
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


def all_mt50_task_names() -> list[str]:
    return [
        "assembly-v3",
        "basketball-v3",
        "bin-picking-v3",
        "box-close-v3",
        "button-press-topdown-v3",
        "button-press-topdown-wall-v3",
        "button-press-v3",
        "button-press-wall-v3",
        "coffee-button-v3",
        "coffee-pull-v3",
        "coffee-push-v3",
        "dial-turn-v3",
        "disassemble-v3",
        "door-close-v3",
        "door-lock-v3",
        "door-open-v3",
        "door-unlock-v3",
        "drawer-close-v3",
        "drawer-open-v3",
        "faucet-close-v3",
        "faucet-open-v3",
        "hammer-v3",
        "hand-insert-v3",
        "handle-press-side-v3",
        "handle-press-v3",
        "handle-pull-side-v3",
        "handle-pull-v3",
        "lever-pull-v3",
        "peg-insert-side-v3",
        "peg-unplug-side-v3",
        "pick-out-of-hole-v3",
        "pick-place-v3",
        "pick-place-wall-v3",
        "plate-slide-back-side-v3",
        "plate-slide-back-v3",
        "plate-slide-side-v3",
        "plate-slide-v3",
        "push-back-v3",
        "push-v3",
        "push-wall-v3",
        "reach-v3",
        "reach-wall-v3",
        "shelf-place-v3",
        "soccer-v3",
        "stick-pull-v3",
        "stick-push-v3",
        "sweep-into-v3",
        "sweep-v3",
        "window-close-v3",
        "window-open-v3",
    ]


def run_command(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def count_task_episodes(output_root: Path, task_name: str) -> int:
    return len(list((output_root / task_name).glob("episode_*/trajectory.npz")))


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    collect_script = args.collect_script.resolve()
    quality_script = args.quality_script.resolve()
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else (output_root / "quality_reports").resolve()
    )
    reject_registry = (
        args.reject_registry.resolve()
        if args.reject_registry is not None
        else (output_root / "rejected" / "reject_registry.jsonl").resolve()
    )

    task_names = parse_task_selection(args.tasks, all_mt50_task_names())
    task_names = apply_task_exclusions(task_names, args.exclude_tasks)

    pipeline_summary: dict[str, dict[str, int | bool]] = {}
    for task_name in task_names:
        round_index = 0
        pipeline_summary[task_name] = {
            "target": args.episodes_per_task,
            "final_count": 0,
            "completed": False,
            "rounds_used": 0,
        }

        while round_index < args.max_rounds:
            round_index += 1
            before_count = count_task_episodes(output_root, task_name)

            if before_count < args.episodes_per_task:
                collect_cmd = [
                    args.python_bin,
                    str(collect_script),
                    "--tasks",
                    task_name,
                    "--episodes-per-task",
                    str(args.episodes_per_task),
                    "--output-root",
                    str(output_root),
                    "--benchmark-seed",
                    str(args.benchmark_seed),
                    "--camera-names",
                    args.camera_names,
                    "--camera-keys",
                    args.camera_keys,
                    "--width",
                    str(args.width),
                    "--height",
                    str(args.height),
                    "--video-fps",
                    str(args.video_fps),
                    "--max-goal-sets",
                    str(args.collect_max_goal_sets),
                    "--max-attempts-per-task",
                    str(args.collect_max_attempts_per_task),
                    "--reject-registry",
                    str(reject_registry),
                    "--resume",
                ]
                if args.max_steps is not None:
                    collect_cmd.extend(["--max-steps", str(args.max_steps)])
                run_command(collect_cmd)

            quality_cmd = [
                args.python_bin,
                str(quality_script),
                "--input-root",
                str(output_root),
                "--tasks",
                task_name,
                "--report-dir",
                str(report_dir / task_name),
                "--reject-registry",
                str(reject_registry),
                "--score-threshold",
                str(args.quality_score_threshold),
                "--max-reject-ratio",
                str(args.quality_max_reject_ratio),
                "--min-episodes",
                str(args.quality_min_episodes),
                "--prune",
            ]
            if args.dry_run_quality:
                quality_cmd.append("--dry-run")
            run_command(quality_cmd)

            after_count = count_task_episodes(output_root, task_name)
            pipeline_summary[task_name]["final_count"] = after_count
            pipeline_summary[task_name]["rounds_used"] = round_index

            if after_count >= args.episodes_per_task:
                pipeline_summary[task_name]["completed"] = True
                break

            if after_count <= before_count and before_count < args.episodes_per_task:
                # No net progress in this round; continue to the next round but
                # make it visible in the summary.
                pass

    summary = {
        "output_root": str(output_root),
        "report_dir": str(report_dir),
        "reject_registry": str(reject_registry),
        "episodes_per_task": args.episodes_per_task,
        "max_rounds": args.max_rounds,
        "tasks": pipeline_summary,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
