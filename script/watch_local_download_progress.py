#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

from watch_cfff_train_jobs import (
    DEFAULT_WECHAT_WEBHOOK,
    WeComConfig,
    build_wecom_config,
    log,
    send_wecom_summary,
)


DEFAULT_DATASET_ROOT = Path("/mnt/sda/syr/datasets/libero_plus_lerobot")
DEFAULT_LOG_GLOB = "/mnt/sda/syr/logs/libero_plus_lerobot*.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a local dataset download and send hourly WeCom summaries."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Explicit repo metadata json. Defaults to dataset_root/.curl-mirror/repo_metadata.json "
        "or dataset_root/.hfd/repo_metadata.json.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Download log to tail. Defaults to the latest matching local libero_plus log.",
    )
    parser.add_argument(
        "--log-glob",
        default=DEFAULT_LOG_GLOB,
        help="Glob used to resolve the latest log when --log-file is omitted.",
    )
    parser.add_argument(
        "--process-pattern",
        default=os.environ.get("DOWNLOAD_PROCESS_PATTERN", "run_libero_plus_hf_mirror_curl"),
        help="Substring used to locate the active downloader process via pgrep -af.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--wechat-webhook",
        default=os.environ.get("WECHAT_WEBHOOK", DEFAULT_WECHAT_WEBHOOK),
    )
    parser.add_argument(
        "--wechat-webhook-key",
        default=os.environ.get("WECHAT_WEBHOOK_KEY", ""),
    )
    parser.add_argument(
        "--notify-cooldown",
        type=int,
        default=int(os.environ.get("NOTIFY_COOLDOWN", "3600")),
        help="Seconds between WeCom summaries.",
    )
    parser.add_argument(
        "--wechat-timeout-seconds",
        type=float,
        default=float(os.environ.get("WECHAT_TIMEOUT_SECONDS", "10")),
    )
    parser.add_argument(
        "--title",
        default=os.environ.get("WATCH_TITLE", "LIBERO+ Download Watch"),
    )
    parser.add_argument("--wecom-send-now", action="store_true")
    return parser.parse_args()


def resolve_metadata_json(dataset_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidates = [
        dataset_root / ".curl-mirror" / "repo_metadata.json",
        dataset_root / ".hfd" / "repo_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_log_file(explicit: Path | None, log_glob: str) -> Path | None:
    if explicit is not None:
        return explicit
    matches = sorted(Path("/").glob(log_glob.lstrip("/")))
    return matches[-1] if matches else None


def load_repo_metadata(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_stats(root: Path) -> tuple[int, int]:
    total_bytes = 0
    total_files = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            total_bytes += st.st_size
            total_files += 1
    return total_bytes, total_files


def format_bytes(num: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def latest_log_line(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if line.strip():
            clean = " ".join(line.split())
            return clean[:220]
    return None


def find_process_lines(pattern: str) -> list[str]:
    if not pattern.strip():
        return []
    proc = subprocess.run(
        ["/bin/bash", "-lc", f"pgrep -af {pattern!r} || true"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def build_summary_lines(
    dataset_root: Path,
    metadata: dict | None,
    current_bytes: int,
    current_files: int,
    delta_bytes: int,
    delta_seconds: float,
    log_line: str | None,
    process_lines: list[str],
) -> list[str]:
    total_bytes = metadata.get("usedStorage") if metadata else None
    total_siblings = len(metadata.get("siblings", [])) if metadata and isinstance(metadata.get("siblings"), list) else None

    if isinstance(total_bytes, int) and total_bytes > 0:
        progress = current_bytes / total_bytes
        progress_line = (
            f"progress: {progress:.2%} | bytes={format_bytes(current_bytes)}/{format_bytes(total_bytes)}"
        )
    else:
        progress_line = f"progress: unknown total | bytes={format_bytes(current_bytes)}"

    if isinstance(total_siblings, int) and total_siblings > 0:
        files_line = f"files: {current_files}/{total_siblings}"
    else:
        files_line = f"files: {current_files}"

    speed = delta_bytes / delta_seconds if delta_seconds > 0 else 0.0
    speed_line = (
        f"hour_avg_speed: {format_bytes(speed)}/s | delta={format_bytes(delta_bytes)} in {int(delta_seconds)}s"
    )

    status = "running" if process_lines else "stopped"
    status_line = f"status: {status}"
    if process_lines:
        first = process_lines[0]
        status_line = f"{status_line} | proc={first[:180]}"

    lines = [
        f"dataset_root: {dataset_root}",
        progress_line,
        files_line,
        speed_line,
        status_line,
    ]
    if log_line:
        lines.append(f"last_log: {log_line}")
    return lines


def main() -> int:
    args = parse_args()
    wecom: WeComConfig | None = build_wecom_config(args)
    metadata_path = resolve_metadata_json(args.dataset_root, args.metadata_json)
    metadata = load_repo_metadata(metadata_path)
    log_path = resolve_log_file(args.log_file, args.log_glob)

    current_bytes, current_files = collect_stats(args.dataset_root)
    baseline_bytes = current_bytes
    baseline_time = time.time()
    last_wecom_ts: float | None = None if wecom is None or wecom.send_now else baseline_time

    while True:
        now = time.time()
        current_bytes, current_files = collect_stats(args.dataset_root)
        process_lines = find_process_lines(args.process_pattern)
        log_line = latest_log_line(log_path)

        delta_bytes = current_bytes - baseline_bytes
        delta_seconds = now - baseline_time

        lines = build_summary_lines(
            dataset_root=args.dataset_root,
            metadata=metadata,
            current_bytes=current_bytes,
            current_files=current_files,
            delta_bytes=delta_bytes,
            delta_seconds=delta_seconds,
            log_line=log_line,
            process_lines=process_lines,
        )

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
                    baseline_bytes = current_bytes
                    baseline_time = now
                    log(
                        f"wecom sent title={wecom.title!r} interval={wecom.notify_seconds}s "
                        f"bytes={current_bytes}"
                    )
                except Exception as exc:
                    log(f"wecom send failed: {exc}")

        if args.once:
            break
        time.sleep(max(1, int(args.poll_seconds)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
