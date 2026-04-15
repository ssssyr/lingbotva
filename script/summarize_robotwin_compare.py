#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


def load_task_metrics(root: Path, seed: int) -> dict[str, dict]:
    metrics_root = root / f"stseed-{10000 * (1 + seed)}" / "metrics"
    results = {}
    for task in TASKS:
        res_path = metrics_root / task / "res.json"
        if not res_path.exists():
            raise FileNotFoundError(f"Missing metrics for task {task}: {res_path}")
        with open(res_path, "r", encoding="utf-8") as f:
            results[task] = json.load(f)
    return results


def mean(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def summarize(condition_name: str, task_metrics: dict[str, dict]) -> dict:
    return {
        "condition": condition_name,
        "task_count": len(task_metrics),
        "mean_success_rate": mean([m.get("succ_rate") for m in task_metrics.values()]),
        "mean_scheduler_video_steps": mean([m.get("scheduler_video_steps_mean") for m in task_metrics.values()]),
        "mean_scheduler_total_runtime_ms": mean([m.get("scheduler_total_runtime_ms_mean") for m in task_metrics.values()]),
        "mean_scheduler_video_runtime_ms": mean([m.get("scheduler_video_runtime_ms_mean") for m in task_metrics.values()]),
        "mean_scheduler_action_runtime_ms": mean([m.get("scheduler_action_runtime_ms_mean") for m in task_metrics.values()]),
    }


def build_comparison(fixed_metrics: dict[str, dict], hazard_metrics: dict[str, dict]) -> dict:
    per_task = {}
    for task in TASKS:
        fixed = fixed_metrics[task]
        hazard = hazard_metrics[task]
        per_task[task] = {
            "fixed_succ_rate": fixed.get("succ_rate"),
            "hazard_succ_rate": hazard.get("succ_rate"),
            "succ_rate_delta": (
                None
                if fixed.get("succ_rate") is None or hazard.get("succ_rate") is None
                else float(hazard["succ_rate"]) - float(fixed["succ_rate"])
            ),
            "fixed_video_steps": fixed.get("scheduler_video_steps_mean"),
            "hazard_video_steps": hazard.get("scheduler_video_steps_mean"),
            "video_steps_delta": (
                None
                if fixed.get("scheduler_video_steps_mean") is None or hazard.get("scheduler_video_steps_mean") is None
                else float(hazard["scheduler_video_steps_mean"]) - float(fixed["scheduler_video_steps_mean"])
            ),
            "fixed_total_runtime_ms": fixed.get("scheduler_total_runtime_ms_mean"),
            "hazard_total_runtime_ms": hazard.get("scheduler_total_runtime_ms_mean"),
            "total_runtime_ms_delta": (
                None
                if fixed.get("scheduler_total_runtime_ms_mean") is None or hazard.get("scheduler_total_runtime_ms_mean") is None
                else float(hazard["scheduler_total_runtime_ms_mean"]) - float(fixed["scheduler_total_runtime_ms_mean"])
            ),
        }
    return per_task


def print_summary(summary: dict, per_task: dict):
    print("== Overall ==")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print()
    print("== Per Task ==")
    header = (
        "task",
        "fixed_succ",
        "hazard_succ",
        "fixed_steps",
        "hazard_steps",
        "fixed_ms",
        "hazard_ms",
    )
    print("\t".join(header))
    for task in TASKS:
        row = per_task[task]
        print(
            "\t".join(
                [
                    task,
                    str(row["fixed_succ_rate"]),
                    str(row["hazard_succ_rate"]),
                    str(row["fixed_video_steps"]),
                    str(row["hazard_video_steps"]),
                    str(row["fixed_total_runtime_ms"]),
                    str(row["hazard_total_runtime_ms"]),
                ]
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=str, required=True)
    parser.add_argument("--fixed-save-root", type=str, required=True)
    parser.add_argument("--hazard-save-root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.robotwin_root)
    fixed_root = root / args.fixed_save_root
    hazard_root = root / args.hazard_save_root

    fixed_metrics = load_task_metrics(fixed_root, args.seed)
    hazard_metrics = load_task_metrics(hazard_root, args.seed)

    fixed_summary = summarize("fixed_k25", fixed_metrics)
    hazard_summary = summarize("hazard_step9000", hazard_metrics)
    per_task = build_comparison(fixed_metrics, hazard_metrics)

    overall = {
        "fixed": fixed_summary,
        "hazard": hazard_summary,
        "hazard_minus_fixed_success_rate": (
            None
            if fixed_summary["mean_success_rate"] is None or hazard_summary["mean_success_rate"] is None
            else hazard_summary["mean_success_rate"] - fixed_summary["mean_success_rate"]
        ),
        "hazard_minus_fixed_video_steps": (
            None
            if fixed_summary["mean_scheduler_video_steps"] is None or hazard_summary["mean_scheduler_video_steps"] is None
            else hazard_summary["mean_scheduler_video_steps"] - fixed_summary["mean_scheduler_video_steps"]
        ),
        "hazard_minus_fixed_total_runtime_ms": (
            None
            if fixed_summary["mean_scheduler_total_runtime_ms"] is None or hazard_summary["mean_scheduler_total_runtime_ms"] is None
            else hazard_summary["mean_scheduler_total_runtime_ms"] - fixed_summary["mean_scheduler_total_runtime_ms"]
        ),
    }

    output = {
        "overall": overall,
        "per_task": per_task,
    }

    summary_path = root / args.hazard_save_root / "compare_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print_summary(overall, per_task)
    print()
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
