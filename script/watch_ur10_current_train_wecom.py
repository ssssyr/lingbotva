#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import time
from pathlib import Path

from watch_cfff_train_jobs import (
    DEFAULT_WECHAT_WEBHOOK,
    GPU_QUERY,
    build_gpu_line,
    build_job_line,
    build_wecom_config,
    parse_progress,
    send_wecom_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch the active UR10 training log on this host and report to WeCom."
    )
    parser.add_argument("--log-file", default=os.environ.get("WATCH_LOG_TARGET", ""))
    parser.add_argument(
        "--tmux-session",
        default=os.environ.get(
            "WATCH_TRAIN_TMUX",
            "ur10-10hz-pm1-flex-max256-ncclfix-4gpu-20260501-145302",
        ),
    )
    parser.add_argument("--job-label", default=os.environ.get("WATCH_JOB_LABEL", "ur10"))
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--tail-lines", type=int, default=500)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--notify-cooldown",
        type=int,
        default=int(os.environ.get("NOTIFY_COOLDOWN", "3600")),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--remote-root",
        default=os.environ.get("CFFF_REMOTE_ROOT", str(REPO_ROOT)),
    )
    parser.add_argument(
        "--wechat-webhook",
        default=os.environ.get("WECHAT_WEBHOOK", DEFAULT_WECHAT_WEBHOOK),
    )
    parser.add_argument(
        "--wechat-webhook-key",
        default=os.environ.get("WECHAT_WEBHOOK_KEY", ""),
    )
    parser.add_argument(
        "--wechat-timeout-seconds",
        type=float,
        default=float(os.environ.get("WECHAT_TIMEOUT_SECONDS", "10")),
    )
    parser.add_argument(
        "--title",
        default=os.environ.get("WATCH_TITLE", "UR10 Train Watch"),
    )
    parser.add_argument("--wecom-send-now", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def run_text(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def find_active_train_log() -> Path | None:
    proc = run_text(["pgrep", "-f", "python.*wan_va.train"])
    for raw_pid in proc.stdout.splitlines():
        pid = raw_pid.strip()
        if not pid:
            continue
        fd1 = Path("/proc") / pid / "fd" / "1"
        try:
            target = fd1.resolve(strict=True)
        except OSError:
            continue
        if target.is_file():
            return target

    candidates = sorted((REPO_ROOT / "logs" / "ur10_train").glob("*.log"))
    return candidates[-1] if candidates else None


def read_tail(path: Path, max_lines: int) -> str:
    proc = run_text(["tail", "-n", str(max_lines), str(path)])
    return (proc.stdout or "") + (proc.stderr or "")


def tmux_session_exists(session: str) -> bool:
    if not session:
        return False
    proc = run_text(["tmux", "has-session", "-t", session])
    return proc.returncode == 0


def fetch_gpu_snapshot() -> list[str]:
    proc = run_text(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def parse_progress_with_train_step(text: str) -> dict[str, object]:
    parsed = parse_progress(text)
    for line in reversed([raw.strip() for raw in text.splitlines() if raw.strip()]):
        if "train_step step=" not in line:
            continue
        for key, pattern, convert in (
            ("current_step", r"train_step step=(\d+)", int),
            ("latent_loss", r"latent_loss=([0-9.eE+-]+)", float),
            ("action_loss", r"action_loss=([0-9.eE+-]+)", float),
            ("grad_norm", r"grad_norm=([0-9.eE+-]+)", float),
            ("lr", r"lr=([0-9.eE+-]+)", float),
            ("sec_per_it", r"step_time_s=([0-9.eE+-]+)", float),
        ):
            match = re.search(pattern, line)
            if match:
                parsed[key] = convert(match.group(1))
        break
    return parsed


def main() -> int:
    args = parse_args()
    wecom = build_wecom_config(args)
    previous_step: int | None = None
    previous_report_ts: float | None = None
    last_wecom_ts: float | None = None if wecom is None or wecom.send_now else time.time()

    while True:
        now = time.time()
        summary_ts = dt.datetime.now()
        elapsed_since_report = None if previous_report_ts is None else now - previous_report_ts
        summary_lines: list[str] = []

        if not args.skip_gpu:
            gpu_line = build_gpu_line(fetch_gpu_snapshot())
            summary_lines.append(gpu_line)
            log(gpu_line)

        log_path = Path(args.log_file).expanduser() if args.log_file else find_active_train_log()
        text = read_tail(log_path, args.tail_lines) if log_path else ""
        parsed = parse_progress_with_train_step(text)
        running = tmux_session_exists(args.tmux_session)
        status = "running" if running else "stopped"
        line = build_job_line(
            label=args.job_label,
            status=status,
            parsed=parsed,
            previous_step=previous_step,
            elapsed_since_report=elapsed_since_report,
        )
        if log_path:
            line = f"{line} | log={log_path}"
        summary_lines.append(line)
        log(line)

        current_step = parsed.get("current_step")
        if isinstance(current_step, int):
            previous_step = current_step

        if wecom is not None:
            should_send = False
            if wecom.send_now and last_wecom_ts is None:
                should_send = True
            elif wecom.notify_seconds > 0 and (
                last_wecom_ts is None or (now - last_wecom_ts) >= wecom.notify_seconds
            ):
                should_send = True

            if should_send:
                try:
                    send_wecom_summary(wecom, summary_ts, args.remote_root, summary_lines)
                    last_wecom_ts = now
                    log(
                        f"wecom sent title={wecom.title!r} "
                        f"interval={wecom.notify_seconds}s"
                    )
                except Exception as exc:
                    log(f"wecom send failed: {exc}")

        previous_report_ts = now

        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
