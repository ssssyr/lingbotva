#!/usr/bin/env python3
import argparse
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
TASK_RE = re.compile(r"running task=(\S+) test_num=(\d+) gpu_id=(\d+)")
STEP_RE = re.compile(r"step:\s*(\d+)\s*/\s*(\d+)")
RATE_RE = re.compile(
    r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%.*current seed:\s*([0-9]+)")
TASK_NAME_RE = re.compile(r"Task Name:\s*(\S+)")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def now_str() -> str:
    return datetime.now().strftime("%F %T")


def default_log_path(repo_root: Path) -> Path | None:
    candidates = sorted(repo_root.glob("train_out/**/batch_eval_logs/*_client.log"))
    return candidates[-1] if candidates else None


@dataclass
class EpisodeResult:
    task_name: str
    completed: int
    target: int | None
    succeeded: int
    rate_percent: float
    result: str
    seed: int


class ProgressState:

    def __init__(self, step_interval: int, history_limit: int = 10):
        self.step_interval = max(1, step_interval)
        self.task_name: str | None = None
        self.task_target: int | None = None
        self.task_gpu: int | None = None
        self.completed_episodes = 0
        self.successes = 0
        self.current_step = 0
        self.step_total = 0
        self.pending_result: str | None = None
        self.last_seed: int | None = None
        self.last_episode: EpisodeResult | None = None
        self.last_step_reported = 0
        self.last_emit_time = 0.0
        self.history: deque[str] = deque(maxlen=history_limit)

    def current_episode_index(self) -> int | None:
        if self.task_target is None:
            return None
        return min(self.completed_episodes + 1, self.task_target)

    def snapshot(self) -> str:
        if not self.task_name:
            return "No active task detected yet."
        episode_idx = self.current_episode_index()
        episode_text = "?"
        if episode_idx is not None and self.task_target is not None:
            episode_text = f"{episode_idx}/{self.task_target}"
        step_text = "?"
        if self.step_total:
            step_text = f"{self.current_step}/{self.step_total}"
        return (
            f"CURRENT task={self.task_name} episode={episode_text} "
            f"step={step_text} succ={self.successes}/{self.completed_episodes}"
        )

    def record_event(self, text: str) -> None:
        stamped = f"[{now_str()}] {text}"
        self.history.append(stamped)
        print(stamped, flush=True)
        self.last_emit_time = time.time()

    def process_line(self, raw_line: str, emit: bool) -> None:
        line = strip_ansi(raw_line).strip()
        if not line:
            return

        task_match = TASK_RE.search(line)
        if task_match:
            self.task_name = task_match.group(1)
            self.task_target = int(task_match.group(2))
            self.task_gpu = int(task_match.group(3))
            self.completed_episodes = 0
            self.successes = 0
            self.current_step = 0
            self.step_total = 0
            self.pending_result = None
            self.last_seed = None
            self.last_step_reported = 0
            if emit:
                self.record_event(
                    f"TASK start task={self.task_name} target={self.task_target} gpu={self.task_gpu}"
                )
            else:
                self.history.append(
                    f"[history] TASK start task={self.task_name} target={self.task_target} gpu={self.task_gpu}"
                )
            return

        task_name_match = TASK_NAME_RE.search(line)
        if task_name_match and not self.task_name:
            self.task_name = task_name_match.group(1)
            return

        if line == "Fail!":
            self.pending_result = "fail"
            return
        if line == "Success!":
            self.pending_result = "success"
            return

        step_match = STEP_RE.search(line)
        if step_match:
            self.current_step = int(step_match.group(1))
            self.step_total = int(step_match.group(2))
            if not emit or not self.task_name or not self.step_total:
                return

            should_report = (
                self.current_step in (1, self.step_total) or
                self.current_step - self.last_step_reported >= self.step_interval
            )
            if should_report:
                episode_idx = self.current_episode_index()
                episode_text = "?"
                if episode_idx is not None and self.task_target is not None:
                    episode_text = f"{episode_idx}/{self.task_target}"
                self.record_event(
                    f"STEP task={self.task_name} episode={episode_text} "
                    f"step={self.current_step}/{self.step_total} "
                    f"succ={self.successes}/{self.completed_episodes}"
                )
                self.last_step_reported = self.current_step
            return

        rate_match = RATE_RE.search(line)
        if rate_match:
            new_successes = int(rate_match.group(1))
            completed = int(rate_match.group(2))
            rate_percent = float(rate_match.group(3))
            seed = int(rate_match.group(4))
            result = self.pending_result
            if result is None:
                result = "success" if new_successes > self.successes else "fail"
            self.successes = new_successes
            self.completed_episodes = completed
            self.last_seed = seed
            self.current_step = 0
            self.last_step_reported = 0
            self.pending_result = None
            if self.task_name:
                self.last_episode = EpisodeResult(
                    task_name=self.task_name,
                    completed=completed,
                    target=self.task_target,
                    succeeded=new_successes,
                    rate_percent=rate_percent,
                    result=result,
                    seed=seed,
                )
            if emit and self.task_name:
                target_text = self.task_target if self.task_target is not None else "?"
                self.record_event(
                    f"EPISODE task={self.task_name} episode={completed}/{target_text} "
                    f"result={result} seed={seed} succ={new_successes}/{completed} "
                    f"rate={rate_percent:.1f}%"
                )
                if self.task_target is not None and completed >= self.task_target:
                    self.record_event(
                        f"TASK done task={self.task_name} succ={new_successes}/{completed} "
                        f"rate={rate_percent:.1f}%"
                    )
            return

        if emit and (
            line.startswith("Traceback") or "RuntimeError:" in line or
            "torch.OutOfMemoryError" in line or line.startswith("XIO:")
        ):
            self.record_event(f"ERROR {line}")


class LogMonitor:

    def __init__(
        self,
        log_path: Path,
        step_interval: int,
        poll_seconds: float,
        heartbeat_seconds: float,
    ):
        self.log_path = log_path
        self.poll_seconds = poll_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.state = ProgressState(step_interval=step_interval)
        self.partial = ""

    def feed(self, text: str, emit: bool) -> None:
        if not text:
            return
        normalized = (self.partial + text).replace("\r", "\n")
        lines = normalized.split("\n")
        self.partial = lines.pop()
        for line in lines:
            self.state.process_line(line, emit=emit)

    def load_existing(self) -> None:
        with self.log_path.open("r", encoding="utf-8", errors="replace") as f:
            self.feed(f.read(), emit=False)

    def print_snapshot(self) -> None:
        print(f"[{now_str()}] Monitoring {self.log_path}", flush=True)
        if self.state.last_episode is not None:
            ep = self.state.last_episode
            target_text = ep.target if ep.target is not None else "?"
            print(
                f"[{now_str()}] LAST task={ep.task_name} episode={ep.completed}/{target_text} "
                f"result={ep.result} seed={ep.seed} succ={ep.succeeded}/{ep.completed} "
                f"rate={ep.rate_percent:.1f}%",
                flush=True,
            )
        print(f"[{now_str()}] {self.state.snapshot()}", flush=True)

    def follow(self) -> None:
        with self.log_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            while True:
                chunk = f.read()
                if chunk:
                    self.feed(chunk, emit=True)
                    continue

                if (
                    self.heartbeat_seconds > 0 and
                    time.time() - self.state.last_emit_time >= self.heartbeat_seconds
                ):
                    print(f"[{now_str()}] HEARTBEAT {self.state.snapshot()}",
                          flush=True)
                    self.state.last_emit_time = time.time()

                time.sleep(self.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor RoboTwin batch client logs with task/episode/step updates."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to the client log. Defaults to the newest *_client.log under train_out/**/batch_eval_logs/.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="How often to poll for new log data.",
    )
    parser.add_argument(
        "--step-interval",
        type=int,
        default=25,
        help="Print step progress every N steps within an episode.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=60.0,
        help="Emit a heartbeat snapshot if no new event was printed for this many seconds.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Print the current parsed status and exit without following.",
    )
    parser.add_argument(
        "--wait-for-file",
        action="store_true",
        help="Wait until the log file appears instead of failing immediately.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    log_path = args.log or default_log_path(repo_root)
    if log_path is None:
        print("No client log found under train_out/**/batch_eval_logs/.",
              file=sys.stderr)
        return 1

    while not log_path.exists():
        if not args.wait_for_file:
            print(f"Log file does not exist: {log_path}", file=sys.stderr)
            return 1
        print(f"[{now_str()}] Waiting for log file: {log_path}", flush=True)
        time.sleep(args.poll_seconds)

    monitor = LogMonitor(
        log_path=log_path,
        step_interval=args.step_interval,
        poll_seconds=args.poll_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    monitor.load_existing()
    monitor.print_snapshot()
    if args.snapshot:
        return 0

    try:
        monitor.follow()
    except KeyboardInterrupt:
        print(f"\n[{now_str()}] Stopped.", flush=True)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
