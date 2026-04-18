#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_ROOT = "/home/ct_24210860031/.cache/cfff-code/lingbot-va"
DEFAULT_WECHAT_WEBHOOK = (
    "https://qyapi.weixin.qq.com/cgi-bin/webhook/send"
    "?key=2c918f90-41d7-418b-a8cc-3519f0f01e82"
)
GPU_QUERY = "index,memory.used,memory.total,utilization.gpu"
ALERT_PATTERNS = (
    "illegal memory access",
    "cuda out of memory",
    "out of memory",
    "denseattnmaskbuilder.build_masks",
    "traceback",
    "runtimeerror",
)


@dataclass(frozen=True)
class JobSpec:
    label: str
    job_name: str


@dataclass(frozen=True)
class WeComConfig:
    webhook_url: str
    notify_seconds: int
    timeout_seconds: float
    title: str
    send_now: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch one or more remote CFFF training jobs."
    )
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        help="Job to watch. Use LABEL=JOB_NAME for shorter display labels.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--tail-lines", type=int, default=300)
    parser.add_argument(
        "--remote-root",
        default=os.environ.get("CFFF_REMOTE_ROOT", DEFAULT_REMOTE_ROOT),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
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
        default=os.environ.get("WATCH_TITLE", "CFFF Train Watch"),
    )
    parser.add_argument("--wecom-send-now", action="store_true")
    return parser.parse_args()


def parse_jobs(raw_jobs: list[str]) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    for raw in raw_jobs:
        if "=" in raw:
            label, job_name = raw.split("=", 1)
        else:
            label, job_name = raw, raw
        label = label.strip()
        job_name = job_name.strip()
        if not label or not job_name:
            raise SystemExit(f"Invalid --job value: {raw!r}")
        jobs.append(JobSpec(label=label, job_name=job_name))
    return jobs


def build_wecom_config(args: argparse.Namespace) -> WeComConfig | None:
    webhook_url = (args.wechat_webhook or "").strip()
    webhook_key = (args.wechat_webhook_key or "").strip()
    if not webhook_url and webhook_key:
        webhook_url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}"
    if not webhook_url:
        return None
    return WeComConfig(
        webhook_url=webhook_url,
        notify_seconds=max(0, int(args.notify_cooldown)),
        timeout_seconds=max(1.0, float(args.wechat_timeout_seconds)),
        title=(args.title or "CFFF Train Watch").strip() or "CFFF Train Watch",
        send_now=bool(args.wecom_send_now),
    )


def run_bash(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-lc", command],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def cfff_prefix(remote_root: str) -> str:
    return f"CFFF_REMOTE_ROOT={shlex.quote(remote_root)}"


def fetch_log(job_name: str, tail_lines: int, remote_root: str) -> tuple[str | None, str]:
    cmd = (
        f"{cfff_prefix(remote_root)} ./script/cfff-log-show.sh "
        f"{shlex.quote(job_name)} {int(tail_lines)}"
    )
    proc = run_bash(cmd)
    output = (proc.stdout or "") + (proc.stderr or "")
    log_path = None
    for line in output.splitlines():
        match = re.match(r"==> (.+) <==", line.strip())
        if match:
            log_path = match.group(1)
            break
    return log_path, output


def fetch_tmux_sessions(job_name: str, remote_root: str) -> int:
    cmd = (
        f"{cfff_prefix(remote_root)} ./script/cfff-run.sh "
        f"\"tmux ls 2>/dev/null | grep '^"
        f"{job_name}-' || true\""
    )
    proc = run_bash(cmd)
    return len([line for line in proc.stdout.splitlines() if line.strip()])


def fetch_gpu_snapshot(remote_root: str) -> list[str]:
    cmd = (
        f"{cfff_prefix(remote_root)} CFFF_CONDA_ENV='' ./script/cfff-run.sh "
        f"\"nvidia-smi --query-gpu={GPU_QUERY} --format=csv,noheader,nounits\""
    )
    proc = run_bash(cmd)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def parse_duration_seconds(token: str | None) -> float | None:
    if not token:
        return None
    parts = token.split(":")
    try:
        values = [int(part) for part in parts]
    except ValueError:
        return None
    total = 0
    for value in values:
        total = total * 60 + value
    return float(total)


def parse_progress(text: str) -> dict[str, object]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parsed: dict[str, object] = {
        "current_step": None,
        "total_steps": None,
        "latent_loss": None,
        "action_loss": None,
        "grad_norm": None,
        "lr": None,
        "sec_per_it": None,
        "elapsed_seconds": None,
        "last_checkpoint_step": None,
        "finished_exit_code": None,
        "alert": None,
    }

    for line in reversed(lines):
        lower = line.lower()
        if parsed["finished_exit_code"] is None:
            match = re.search(r"\[cfff-job\] finished_at=.* exit_code=(\d+)", line)
            if match:
                parsed["finished_exit_code"] = int(match.group(1))
        if parsed["alert"] is None:
            for pattern in ALERT_PATTERNS:
                if pattern in lower:
                    parsed["alert"] = line
                    break
        if parsed["last_checkpoint_step"] is None:
            match = re.search(r"checkpoint_step_(\d+)", line)
            if not match:
                match = re.search(r"Checkpoint saved successfully at step (\d+)", line)
            if match:
                parsed["last_checkpoint_step"] = int(match.group(1))
        if "Training:" in line:
            if parsed["total_steps"] is None:
                match = re.search(r"(\d+)/(\d+)", line)
                if match:
                    parsed["current_step"] = int(match.group(1))
                    parsed["total_steps"] = int(match.group(2))
            if parsed["elapsed_seconds"] is None:
                match = re.search(r"\[([0-9:]+)<", line)
                if match:
                    parsed["elapsed_seconds"] = parse_duration_seconds(match.group(1))
            if parsed["sec_per_it"] is None:
                match = re.search(r"([0-9.]+)s/it", line)
                if match:
                    parsed["sec_per_it"] = float(match.group(1))
            if "step=" in line and ("latent_loss=" in line or "action_loss=" in line):
                if parsed["current_step"] is None:
                    match = re.search(r"step=(\d+)", line)
                    if match:
                        parsed["current_step"] = int(match.group(1))
                if parsed["latent_loss"] is None:
                    match = re.search(r"latent_loss=([0-9.eE+-]+)", line)
                    if match:
                        parsed["latent_loss"] = float(match.group(1))
                if parsed["action_loss"] is None:
                    match = re.search(r"action_loss=([0-9.eE+-]+)", line)
                    if match:
                        parsed["action_loss"] = float(match.group(1))
                if parsed["grad_norm"] is None:
                    match = re.search(r"grad_norm=([0-9.eE+-]+)", line)
                    if match:
                        parsed["grad_norm"] = float(match.group(1))
                if parsed["lr"] is None:
                    match = re.search(r"lr=([0-9.eE+-]+)", line)
                    if match:
                        parsed["lr"] = float(match.group(1))
                if parsed["latent_loss"] is not None and parsed["action_loss"] is not None:
                    break

    if parsed["total_steps"] is None:
        for line in lines:
            match = re.search(r"\[steps\]\s+(\d+)", line)
            if match:
                parsed["total_steps"] = int(match.group(1))
                break

    return parsed


def format_optional_float(value: object, digits: int, scientific: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "?"
    if scientific:
        return f"{float(value):.{digits}e}"
    return f"{float(value):.{digits}f}"


def build_job_line(
    label: str,
    status: str,
    parsed: dict[str, object],
    previous_step: int | None,
    elapsed_since_report: float | None,
) -> str:
    parts = [label, f"status={status}"]

    current_step = parsed["current_step"]
    total_steps = parsed["total_steps"]
    if isinstance(current_step, int) and isinstance(total_steps, int):
        percent = 100.0 * current_step / total_steps
        parts.append(f"step={current_step}/{total_steps} ({percent:.2f}%)")
    elif isinstance(current_step, int):
        parts.append(f"step={current_step}")
    else:
        parts.append("step=?")

    parts.append(f"latent={format_optional_float(parsed['latent_loss'], 4)}")
    parts.append(f"action={format_optional_float(parsed['action_loss'], 4)}")
    parts.append(f"grad={format_optional_float(parsed['grad_norm'], 2)}")
    parts.append(f"lr={format_optional_float(parsed['lr'], 2, scientific=True)}")

    sec_per_it = parsed["sec_per_it"]
    if isinstance(sec_per_it, (int, float)) and sec_per_it > 0:
        parts.append(f"speed={60.0 / float(sec_per_it):.2f} step/min")

    if (
        isinstance(current_step, int)
        and isinstance(previous_step, int)
        and elapsed_since_report is not None
        and elapsed_since_report > 0
    ):
        delta = current_step - previous_step
        parts.append(f"delta={delta:+d}")
        parts.append(f"recent={60.0 * delta / elapsed_since_report:.2f} step/min")

    if isinstance(parsed["last_checkpoint_step"], int):
        parts.append(f"ckpt={parsed['last_checkpoint_step']}")

    alert = parsed["alert"]
    if isinstance(alert, str):
        clean_alert = re.sub(r"\s+", " ", alert).strip()
        if len(clean_alert) > 120:
            clean_alert = clean_alert[:117] + "..."
        parts.append(f"alert={clean_alert}")

    return " | ".join(parts)


def build_gpu_line(lines: list[str]) -> str:
    if not lines:
        return "GPU unavailable"
    formatted: list[str] = []
    for line in lines:
        match = re.match(r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)", line)
        if not match:
            formatted.append(line)
            continue
        idx, used, total, util = match.groups()
        formatted.append(f"gpu{idx}={used}/{total}MiB {util}%")
    return "GPU " + " | ".join(formatted)


def build_wecom_content(
    title: str,
    summary_ts: dt.datetime,
    remote_root: str,
    lines: list[str],
) -> str:
    content_lines = [
        title,
        f"time: {summary_ts.strftime('%Y-%m-%d %H:%M:%S')}",
        f"remote_root: {remote_root}",
    ]
    content_lines.extend(lines)
    content = "\n".join(content_lines)
    if len(content) > 1800:
        content = content[:1797] + "..."
    return content


def send_wecom_summary(
    config: WeComConfig,
    summary_ts: dt.datetime,
    remote_root: str,
    lines: list[str],
) -> None:
    content = build_wecom_content(config.title, summary_ts, remote_root, lines)
    payload = {
        "msgtype": "text",
        "text": {
            "content": content,
        },
    }
    request = urllib.request.Request(
        config.webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"unexpected response: {body}") from exc

    if parsed.get("errcode") != 0:
        raise RuntimeError(
            f"errcode={parsed.get('errcode')} errmsg={parsed.get('errmsg')}"
        )


def log(message: str) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def main() -> int:
    args = parse_args()
    jobs = parse_jobs(args.job)
    wecom = build_wecom_config(args)
    previous_steps: dict[str, int | None] = {job.job_name: None for job in jobs}
    previous_report_ts: float | None = None
    last_wecom_ts: float | None = None if wecom is None or wecom.send_now else time.time()

    while True:
        now = time.time()
        summary_ts = dt.datetime.now()
        elapsed_since_report = None if previous_report_ts is None else now - previous_report_ts
        summary_lines: list[str] = []

        if not args.skip_gpu:
            gpu_line = build_gpu_line(fetch_gpu_snapshot(args.remote_root))
            summary_lines.append(gpu_line)
            log(gpu_line)

        for job in jobs:
            log_path, text = fetch_log(job.job_name, args.tail_lines, args.remote_root)
            sessions = fetch_tmux_sessions(job.job_name, args.remote_root)
            parsed = parse_progress(text)
            current_step = parsed["current_step"]
            running = parsed["finished_exit_code"] is None and (
                sessions > 0 or isinstance(current_step, int)
            )
            status = "running" if running else "stopped"
            line = build_job_line(
                label=job.label,
                status=status,
                parsed=parsed,
                previous_step=previous_steps.get(job.job_name),
                elapsed_since_report=elapsed_since_report,
            )
            if log_path:
                line = f"{line} | log={log_path}"
            summary_lines.append(line)
            log(line)

            current_step = parsed["current_step"]
            if isinstance(current_step, int):
                previous_steps[job.job_name] = current_step

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
                        f"wecom sent title={wecom.title!r} interval={wecom.notify_seconds}s "
                        f"jobs={len(jobs)}"
                    )
                except Exception as exc:
                    log(f"wecom send failed: {exc}")

        previous_report_ts = now

        if args.once:
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
