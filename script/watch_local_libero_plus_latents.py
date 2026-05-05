#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path

from watch_cfff_train_jobs import (
    DEFAULT_WECHAT_WEBHOOK,
    build_wecom_config,
    build_gpu_line,
    log,
    send_wecom_summary,
)

DEFAULT_DATASET_ROOT = Path('/mnt/sda/syr/datasets/libero_plus_lerobot')
DEFAULT_LOG_FILE = Path('/home/syr/code/lingbot-va/logs/libero_plus_extract/libero_plus_lerobot_latents_tmux_overwrite_20260419_130226.log')
DEFAULT_PROCESS_PATTERN = 'extract_libero_plus_lerobot_latents.py --dataset-root /mnt/sda/syr/datasets/libero_plus_lerobot'
DEFAULT_TOTAL_JOBS = 28694


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Watch local LIBERO-plus latent extraction progress and send hourly WeCom summaries.'
    )
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument('--process-pattern', default=os.environ.get('LATENT_PROCESS_PATTERN', DEFAULT_PROCESS_PATTERN))
    parser.add_argument('--expected-files', type=int, default=int(os.environ.get('LATENT_EXPECTED_FILES', str(DEFAULT_TOTAL_JOBS))))
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--once', action='store_true')
    parser.add_argument(
        '--wechat-webhook',
        default=os.environ.get('WECHAT_WEBHOOK', DEFAULT_WECHAT_WEBHOOK),
    )
    parser.add_argument(
        '--wechat-webhook-key',
        default=os.environ.get('WECHAT_WEBHOOK_KEY', ''),
    )
    parser.add_argument(
        '--notify-cooldown',
        type=int,
        default=int(os.environ.get('NOTIFY_COOLDOWN', '3600')),
        help='Seconds between WeCom summaries.',
    )
    parser.add_argument(
        '--wechat-timeout-seconds',
        type=float,
        default=float(os.environ.get('WECHAT_TIMEOUT_SECONDS', '10')),
    )
    parser.add_argument(
        '--title',
        default=os.environ.get('WATCH_TITLE', 'LIBERO+ VAE Watch'),
    )
    parser.add_argument('--wecom-send-now', action='store_true')
    return parser.parse_args()


def count_latent_files(latent_root: Path) -> int:
    if not latent_root.exists():
        return 0
    return sum(1 for _ in latent_root.rglob('*.pth'))


def has_empty_emb(dataset_root: Path) -> bool:
    return (dataset_root / 'empty_emb.pt').exists()


def latest_log_line(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    for line in reversed(lines):
        if line.strip():
            clean = ' '.join(line.split())
            return clean[:220]
    return None


def parse_progress_from_log(path: Path | None) -> tuple[int | None, int | None]:
    if path is None or not path.exists():
        return None, None
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    pattern = re.compile(r'Extract latents:\s+\d+%\|.*?\|\s*(\d+)/(\d+)')
    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None




def fetch_local_gpu_snapshot() -> list[str]:
    proc = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True,
        encoding='utf-8',
        errors='replace',
        capture_output=True,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

def find_process_lines(pattern: str) -> list[str]:
    if not pattern.strip():
        return []
    proc = subprocess.run(
        ['/bin/bash', '-lc', f'pgrep -af {pattern!r} || true'],
        text=True,
        encoding='utf-8',
        errors='replace',
        capture_output=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def build_summary_lines(args, current_files, delta_files, delta_seconds, log_line, process_lines):
    expected = max(0, int(args.expected_files))
    progress = (current_files / expected) if expected > 0 else 0.0
    files_per_hour = 0.0 if delta_seconds <= 0 else delta_files * 3600.0 / delta_seconds
    status = 'running' if process_lines else 'stopped'
    log_progress_cur, log_progress_total = parse_progress_from_log(args.log_file)
    lines = [
        f'dataset_root: {args.dataset_root}',
        f'progress: {current_files}/{expected} ({progress:.2%}) | empty_emb={"yes" if has_empty_emb(args.dataset_root) else "no"}',
        f'hour_rate: +{delta_files} files in {int(delta_seconds)}s ({files_per_hour:.1f}/h)',
        f'status: {status}',
    ]
    if log_progress_cur is not None and log_progress_total is not None:
        lines.append(f'log_progress: {log_progress_cur}/{log_progress_total}')
    if process_lines:
        lines.append(f'proc: {process_lines[0][:180]}')
    if log_line:
        lines.append(f'last_log: {log_line}')
    gpu_lines = fetch_local_gpu_snapshot()
    if gpu_lines:
        lines.append(build_gpu_line(gpu_lines))
    return lines


def main() -> int:
    args = parse_args()
    wecom = build_wecom_config(args)
    latent_root = args.dataset_root / 'latents'
    baseline_files = count_latent_files(latent_root)
    baseline_time = time.time()
    last_wecom_ts = None if wecom is None or wecom.send_now else baseline_time

    while True:
        now = time.time()
        current_files = count_latent_files(latent_root)
        process_lines = find_process_lines(args.process_pattern)
        log_line = latest_log_line(args.log_file)
        delta_files = current_files - baseline_files
        delta_seconds = now - baseline_time
        lines = build_summary_lines(args, current_files, delta_files, delta_seconds, log_line, process_lines)
        for line in lines:
            log(line)

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
                    send_wecom_summary(wecom, dt.datetime.now(), str(args.dataset_root), lines)
                    last_wecom_ts = now
                    baseline_files = current_files
                    baseline_time = now
                    log(f"wecom sent title={wecom.title!r} interval={wecom.notify_seconds}s files={current_files}")
                except Exception as exc:
                    log(f'wecom send failed: {exc}')

        if args.once:
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == '__main__':
    raise SystemExit(main())
