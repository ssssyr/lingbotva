#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import time
from dataclasses import dataclass, asdict
from string import Template
from textwrap import dedent

from watch_cfff_train_jobs import (
    DEFAULT_WECHAT_WEBHOOK,
    WeComConfig,
    build_gpu_line,
    build_wecom_config,
    cfff_prefix,
    fetch_gpu_snapshot,
    log,
    run_bash,
    send_wecom_summary,
)


DEFAULT_REMOTE_ROOT = "/home/ct_24210860031/812/SYR/code/lingbot-va-libero"
DEFAULT_OUTPUT_ROOT = "/home/ct_24210860031/812/SYR/outputs/libero_plus_full_eval"
DEFAULT_LOG_ROOT = "/home/ct_24210860031/812/SYR/logs/libero_plus_eval"
DEFAULT_BENCHMARK = "libero_10"
DEFAULT_TOTAL_TASKS = 2519


@dataclass(frozen=True)
class ShardSpec:
    label: str
    start: int
    end: int
    port: int
    out_dir: str
    client_log: str
    server_log: str


@dataclass(frozen=True)
class EvalSnapshot:
    benchmark: str
    completed: int
    succ_num: float
    total_num: float
    shards: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch remote LIBERO-plus evaluation progress and send WeCom summaries."
    )
    parser.add_argument("--remote-root", default=os.environ.get("CFFF_REMOTE_ROOT", DEFAULT_REMOTE_ROOT))
    parser.add_argument("--output-root", default=os.environ.get("LIBERO_PLUS_EVAL_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--log-root", default=os.environ.get("LIBERO_PLUS_EVAL_LOG_ROOT", DEFAULT_LOG_ROOT))
    parser.add_argument("--benchmark", default=os.environ.get("LIBERO_PLUS_EVAL_BENCHMARK", DEFAULT_BENCHMARK))
    parser.add_argument(
        "--total-tasks",
        type=int,
        default=int(os.environ.get("LIBERO_PLUS_EVAL_TOTAL_TASKS", str(DEFAULT_TOTAL_TASKS))),
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
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
        default=os.environ.get("WATCH_TITLE", "LIBERO+ Eval Watch"),
    )
    parser.add_argument("--wecom-send-now", action="store_true")
    return parser.parse_args()


def default_shards(output_root: str, log_root: str) -> list[ShardSpec]:
    return [
        ShardSpec(
            label="gpu0",
            start=0,
            end=630,
            port=29056,
            out_dir=f"{output_root}/gpu0",
            client_log=f"{log_root}/client_gpu0_full.log",
            server_log=f"{log_root}/server_gpu0_full.log",
        ),
        ShardSpec(
            label="gpu1",
            start=630,
            end=1260,
            port=29057,
            out_dir=f"{output_root}/gpu1",
            client_log=f"{log_root}/client_gpu1_full.log",
            server_log=f"{log_root}/server_gpu1_full.log",
        ),
        ShardSpec(
            label="gpu2",
            start=1260,
            end=1890,
            port=29058,
            out_dir=f"{output_root}/gpu2",
            client_log=f"{log_root}/client_gpu2_full.log",
            server_log=f"{log_root}/server_gpu2_full.log",
        ),
        ShardSpec(
            label="gpu3",
            start=1890,
            end=2519,
            port=29059,
            out_dir=f"{output_root}/gpu3",
            client_log=f"{log_root}/client_gpu3_full.log",
            server_log=f"{log_root}/server_gpu3_full.log",
        ),
    ]


def build_remote_snapshot_command(benchmark: str, shards: list[ShardSpec]) -> str:
    shard_payload = json.dumps([asdict(shard) for shard in shards], ensure_ascii=True)
    remote_py = Template(dedent('''
        import json
        import pathlib
        import re
        import shlex
        import subprocess
        import time
        import urllib.request

        benchmark = $benchmark_json
        shards = json.loads($shards_json)


        def last_nonempty_line(path_str):
            path = pathlib.Path(path_str)
            if not path.exists():
                return None
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                return None
            for line in reversed(lines):
                clean = " ".join(line.split())
                if clean:
                    return clean[:180]
            return None


        def process_matches(pattern):
            proc = subprocess.run(
                [
                    "/bin/bash",
                    "-lc",
                    f"ps -ef | grep -F -- {shlex.quote(pattern)} | grep -v grep || true",
                ],
                text=True,
                capture_output=True,
            )
            return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


        def health_ok(port):
            try:
                text = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/healthz", timeout=3
                ).read().decode("utf-8", errors="replace").strip()
                return text == "OK"
            except Exception:
                return False


        snapshot = {
            "benchmark": benchmark,
            "completed": 0,
            "succ_num": 0.0,
            "total_num": 0.0,
            "shards": [],
        }

        for shard in shards:
            out_dir = pathlib.Path(shard["out_dir"])
            json_paths = sorted(out_dir.glob(f"{benchmark}_*.json"))
            completed = 0
            succ_num = 0.0
            total_num = 0.0
            latest_task = None
            latest_age_s = None

            for path in json_paths:
                match = re.match(rf"{re.escape(benchmark)}_(\\d+)\\.json$$", path.name)
                if match:
                    task_id = int(match.group(1))
                    latest_task = task_id if latest_task is None else max(latest_task, task_id)
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                completed += 1
                succ_num += float(payload.get("succ_num", 0.0) or 0.0)
                total_num += float(payload.get("total_num", 0.0) or 0.0)
                try:
                    age_s = time.time() - path.stat().st_mtime
                except FileNotFoundError:
                    continue
                latest_age_s = age_s if latest_age_s is None else min(latest_age_s, age_s)

            client_pattern = (
                f"python -u evaluation/libero/client.py --libero-benchmark {benchmark} --port {shard['port']}"
            )
            server_pattern = (
                f"wan_va/wan_va_server.py --config-name libero --port {shard['port']}"
            )
            client_lines = process_matches(client_pattern)
            server_lines = process_matches(server_pattern)

            shard_state = {
                "label": shard["label"],
                "start": int(shard["start"]),
                "end": int(shard["end"]),
                "expected": int(shard["end"]) - int(shard["start"]),
                "port": int(shard["port"]),
                "completed": completed,
                "succ_num": succ_num,
                "total_num": total_num,
                "client_running": bool(client_lines),
                "server_running": bool(server_lines),
                "server_ok": health_ok(int(shard["port"])) or bool(server_lines),
                "latest_task": latest_task,
                "latest_age_s": latest_age_s,
                "last_log": last_nonempty_line(shard["client_log"]),
            }
            snapshot["shards"].append(shard_state)
            snapshot["completed"] += completed
            snapshot["succ_num"] += succ_num
            snapshot["total_num"] += total_num

        print(json.dumps(snapshot, ensure_ascii=True))
    ''')).safe_substitute(
        benchmark_json=json.dumps(benchmark),
        shards_json=json.dumps(shard_payload),
    )
    return "python3 - <<'PY'\n" + remote_py + "\nPY"


def fetch_eval_snapshot(benchmark: str, remote_root: str, shards: list[ShardSpec]) -> EvalSnapshot:
    remote_cmd = build_remote_snapshot_command(benchmark, shards)
    cmd = (
        f"CFFF_CONDA_ENV='' {cfff_prefix(remote_root)} ./script/cfff-run.sh "
        f"{shlex.quote(remote_cmd)}"
    )
    proc = run_bash(cmd)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "remote snapshot failed").strip())

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    payload_line = None
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            payload_line = line
            break
    if payload_line is None:
        raise RuntimeError(f"remote snapshot missing json: {proc.stdout!r}")
    payload = json.loads(payload_line)
    return EvalSnapshot(
        benchmark=str(payload["benchmark"]),
        completed=int(payload["completed"]),
        succ_num=float(payload["succ_num"]),
        total_num=float(payload["total_num"]),
        shards=list(payload["shards"]),
    )


def format_ratio(numerator: float, denominator: float) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{numerator / denominator:.2%}"


def format_count(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def format_age(age_s: float | None) -> str:
    if age_s is None:
        return "n/a"
    if age_s < 60:
        return f"{int(age_s)}s"
    if age_s < 3600:
        return f"{age_s / 60:.1f}m"
    return f"{age_s / 3600:.1f}h"


def build_summary_lines(
    benchmark: str,
    total_tasks: int,
    snapshot: EvalSnapshot,
    gpu_lines: list[str],
    delta_completed: int,
    delta_succ: float,
    delta_seconds: float,
) -> list[str]:
    progress = snapshot.completed / total_tasks if total_tasks > 0 else 0.0
    active_clients = sum(1 for shard in snapshot.shards if shard.get("client_running"))
    healthy_servers = sum(1 for shard in snapshot.shards if shard.get("server_ok"))
    tasks_per_hour = 0.0 if delta_seconds <= 0 else delta_completed * 3600.0 / delta_seconds

    lines = [
        (
            f"benchmark: {benchmark} | progress={snapshot.completed}/{total_tasks} ({progress:.2%}) "
            f"| success={format_count(snapshot.succ_num)}/{format_count(snapshot.total_num)} "
            f"| rate={format_ratio(snapshot.succ_num, snapshot.total_num)}"
        ),
        (
            f"delta: +{delta_completed} tasks, +{format_count(delta_succ)} succ in {int(delta_seconds)}s "
            f"({tasks_per_hour:.1f}/h) | clients={active_clients}/{len(snapshot.shards)} "
            f"| servers={healthy_servers}/{len(snapshot.shards)}"
        ),
    ]
    if gpu_lines:
        lines.append(build_gpu_line(gpu_lines))

    for shard in snapshot.shards:
        shard_progress = shard["completed"] / shard["expected"] if shard["expected"] > 0 else 0.0
        line = (
            f"{shard['label']}: run={'Y' if shard['client_running'] else 'N'} "
            f"srv={'Y' if shard['server_ok'] else 'N'} "
            f"done={shard['completed']}/{shard['expected']} ({shard_progress:.1%}) "
            f"succ={format_count(shard['succ_num'])}/{format_count(shard['total_num'])} "
            f"rate={format_ratio(shard['succ_num'], shard['total_num'])} "
            f"last={shard['latest_task'] if shard['latest_task'] is not None else 'n/a'} "
            f"age={format_age(shard['latest_age_s'])}"
        )
        if (not shard["client_running"] and shard["completed"] < shard["expected"]) or not shard["server_ok"]:
            log_line = shard.get("last_log")
            if isinstance(log_line, str) and log_line:
                clean = re.sub(r"\s+", " ", log_line).strip()[:120]
                line = f"{line} | note={clean}"
        lines.append(line)

    return lines


def collect_alerts(snapshot: EvalSnapshot) -> list[str]:
    alerts: list[str] = []
    for shard in snapshot.shards:
        if not shard["server_ok"]:
            alerts.append(f"{shard['label']} server unhealthy on :{shard['port']}")
        if not shard["client_running"] and shard["completed"] < shard["expected"]:
            alerts.append(
                f"{shard['label']} client stopped at {shard['completed']}/{shard['expected']}"
            )
    return alerts


def main() -> int:
    args = parse_args()
    wecom: WeComConfig | None = build_wecom_config(args)
    shards = default_shards(args.output_root, args.log_root)

    last_wecom_ts: float | None = None if wecom is None or wecom.send_now else time.time()
    baseline_completed = 0
    baseline_succ = 0.0
    baseline_time = time.time()
    last_alert_signature: tuple[str, ...] = ()
    sent_done = False

    while True:
        now = time.time()
        snapshot = fetch_eval_snapshot(args.benchmark, args.remote_root, shards)
        gpu_lines = [] if args.skip_gpu else fetch_gpu_snapshot(args.remote_root)
        delta_completed = snapshot.completed - baseline_completed
        delta_succ = snapshot.succ_num - baseline_succ
        delta_seconds = max(1.0, now - baseline_time)
        lines = build_summary_lines(
            benchmark=args.benchmark,
            total_tasks=args.total_tasks,
            snapshot=snapshot,
            gpu_lines=gpu_lines,
            delta_completed=delta_completed,
            delta_succ=delta_succ,
            delta_seconds=delta_seconds,
        )
        alerts = collect_alerts(snapshot)
        for alert in alerts:
            lines.append(f"ALERT: {alert}")
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
            alert_signature = tuple(alerts)
            done_now = snapshot.completed >= args.total_tasks and args.total_tasks > 0

            if should_send:
                try:
                    send_wecom_summary(wecom, dt.datetime.now(), args.remote_root, lines)
                    last_wecom_ts = now
                    baseline_completed = snapshot.completed
                    baseline_succ = snapshot.succ_num
                    baseline_time = now
                    sent_done = done_now
                    log(
                        f"wecom sent title={wecom.title!r} interval={wecom.notify_seconds}s "
                        f"completed={snapshot.completed}/{args.total_tasks}"
                    )
                except Exception as exc:
                    log(f"wecom send failed: {exc}")
            last_alert_signature = alert_signature

        if args.once:
            return 0

        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
