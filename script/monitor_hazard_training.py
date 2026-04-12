#!/usr/bin/env python3

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show live status for a Hazard training run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory. If omitted, resolves the latest run under --base-root.",
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("/data/syr/train_out/robotwin_hazard_local_8gpu"),
        help="Experiment root containing latest_run.txt and runs/.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Polling interval in seconds when --follow is enabled.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Continuously print status updates.",
    )
    return parser.parse_args()


def resolve_run_dir(args) -> Path:
    if args.run_dir is not None:
        return args.run_dir

    latest_txt = args.base_root / "latest_run.txt"
    if latest_txt.exists():
        return Path(latest_txt.read_text(encoding="utf-8").strip())

    runs_dir = args.base_root / "runs"
    candidates = sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {runs_dir}")
    return candidates[-1]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_duration(seconds):
    if seconds is None:
        return "?"
    return str(timedelta(seconds=int(seconds)))


def enrich_reward_decomposition(metrics):
    metrics = dict(metrics or {})
    if "quality" in metrics or "q_seq" in metrics or "q_delta" in metrics:
        return metrics

    required = ("reward", "baseline_reward", "video_steps", "baseline_video_steps")
    if any(metrics.get(key) is None for key in required):
        return metrics

    implied_action_loss = metrics.get("implied_action_loss")
    implied_baseline_action_loss = metrics.get("implied_baseline_action_loss")
    compute_penalty = metrics.get("compute_penalty")
    baseline_compute_penalty = metrics.get("baseline_compute_penalty")
    if all(value is not None for value in (
        implied_action_loss,
        implied_baseline_action_loss,
        compute_penalty,
        baseline_compute_penalty,
    )):
        return metrics

    reward = float(metrics["reward"])
    baseline_reward = float(metrics["baseline_reward"])
    video_steps = float(metrics["video_steps"])
    baseline_video_steps = float(metrics["baseline_video_steps"])

    # Current hazard trainer uses lambda_cost=0.1 from the active profile.
    lambda_cost = 0.1
    compute_penalty = lambda_cost * video_steps
    baseline_compute_penalty = lambda_cost * baseline_video_steps
    implied_action_loss = -reward - compute_penalty
    implied_baseline_action_loss = -baseline_reward - baseline_compute_penalty

    metrics["compute_penalty"] = compute_penalty
    metrics["baseline_compute_penalty"] = baseline_compute_penalty
    metrics["compute_penalty_gap"] = baseline_compute_penalty - compute_penalty
    metrics["implied_action_loss"] = implied_action_loss
    metrics["implied_baseline_action_loss"] = implied_baseline_action_loss
    metrics["implied_action_loss_gap"] = (
        implied_action_loss - implied_baseline_action_loss
    )
    return metrics


def render_status(run_dir: Path):
    status = load_json(run_dir / "status.json")
    if status is None:
        return f"run_dir={run_dir}\nstatus.json not found yet"

    metrics = enrich_reward_decomposition(status.get("latest_metrics") or {})
    lines = [
        f"run_dir={run_dir}",
        f"state={status.get('state')} run_name={status.get('run_name')}",
        f"step={status.get('step', '?')}/{status.get('max_steps', '?')} progress={status.get('progress', 0.0):.2%}",
        f"micro_step={status.get('micro_step', '?')} heartbeat={status.get('last_heartbeat_at', '?')}",
        f"elapsed={fmt_duration(metrics.get('elapsed_seconds'))} eta={fmt_duration(metrics.get('eta_seconds'))}",
        (
            "loss="
            f"{metrics.get('total_loss', '?')} "
            f"(policy={metrics.get('policy_loss', '?')}, kl={metrics.get('kl_loss', '?')})"
        ),
        (
            "reward="
            f"{metrics.get('reward', '?')} "
            f"baseline={metrics.get('baseline_reward', '?')} "
            f"advantage={metrics.get('advantage', '?')}"
        ),
        (
            "quality="
            f"{metrics.get('quality', '?')} "
            f"cost={metrics.get('cost', '?')} "
            f"ema={metrics.get('ema_baseline', '?')}"
        ),
        (
            "video_steps="
            f"{metrics.get('video_steps', '?')} "
            f"lo={metrics.get('anchor_lo_steps', '?')} "
            f"hi={metrics.get('anchor_hi_steps', '?')} "
            f"grad_norm={metrics.get('grad_norm', '?')} "
            f"lr={metrics.get('lr', '?')}"
        ),
        (
            "q_terms="
            f"seq={metrics.get('q_seq', '?')} "
            f"delta={metrics.get('q_delta', '?')} "
            f"g_seq={metrics.get('g_seq', '?')} "
            f"g_delta={metrics.get('g_delta', '?')}"
        ),
        (
            "loss_terms="
            f"l_seq(cur/lo/hi)="
            f"{metrics.get('l_seq_cur', '?')}/"
            f"{metrics.get('l_seq_lo', '?')}/"
            f"{metrics.get('l_seq_hi', '?')} "
            f"l_delta(cur/lo/hi)="
            f"{metrics.get('l_delta_cur', '?')}/"
            f"{metrics.get('l_delta_lo', '?')}/"
            f"{metrics.get('l_delta_hi', '?')}"
        ),
        f"latest_checkpoint={status.get('latest_checkpoint')}",
        f"rank0_log={run_dir / 'logs' / 'train_rank0.log'}",
    ]
    if status.get("error"):
        lines.append(f"error={status['error']}")
    return "\n".join(lines)


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args)

    if not args.follow:
        print(render_status(run_dir), flush=True)
        return

    while True:
        print(render_status(run_dir), flush=True)
        print("-" * 80, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
