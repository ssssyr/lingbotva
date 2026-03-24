#!/usr/bin/env python3
# Remote high-utilization keepalive start command:
# CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
# CFFF_CONDA_SH=/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/miniconda3/etc/profile.d/conda.sh \
# CFFF_CONDA_ENV=lingbot-va-cu118 \
# ./script/cfff-run.sh "timestamp=\$(date +%Y%m%d-%H%M%S); log_file=logs/gpu-keepalive-python-\$timestamp.log; nohup python gpu_keepalive.py --gpus 0,1,2,3 --size 24576 --work-iters 2 --sleep-s 0.0 --report-every 10 --dtype bf16 > \"\$log_file\" 2>&1 < /dev/null & echo pid=\$!; echo log=\$log_file"
#
# Local direct run example:
# python script/gpu_keepalive.py --gpus 0,1,2,3 --size 24576 --work-iters 2 --sleep-s 0.0 --report-every 10 --dtype bf16
#
# Stop remote keepalive:
# CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive ./script/cfff-run.sh "pkill -f gpu_keepalive.py"

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from typing import Iterable

import torch


STOP = False


def handle_signal(signum, frame):
    del signum, frame
    global STOP
    STOP = True


def parse_visible_devices(raw: str | None, total: int) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(range(total))
    devices = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= total:
            raise ValueError(f"GPU index {idx} is out of range for {total} visible devices")
        devices.append(idx)
    if not devices:
        raise ValueError("No GPU devices selected")
    return devices


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return table[name]


def keepalive_worker(
    device_idx: int,
    size: int,
    work_iters: int,
    sleep_s: float,
    report_every: int,
    dtype_name: str,
) -> None:
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    torch.cuda.set_device(device_idx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dtype = dtype_from_name(dtype_name)
    device = torch.device(f"cuda:{device_idx}")

    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)
    c = torch.randn((size, size), device=device, dtype=dtype)

    # Warm up kernels and memory pools so the steady-state load is stable.
    for _ in range(3):
        c = torch.matmul(a, b) + c
        a, b, c = b, c, a
    torch.cuda.synchronize(device)

    iteration = 0
    start_time = time.time()
    last_report = start_time

    print(
        f"[gpu-keepalive] gpu={device_idx} name={torch.cuda.get_device_name(device_idx)} "
        f"dtype={dtype_name} size={size} work_iters={work_iters} sleep_s={sleep_s}",
        flush=True,
    )

    while not STOP:
        step_start = time.time()
        for _ in range(work_iters):
            c = torch.matmul(a, b) + c
            a, b, c = b, c, a
        torch.cuda.synchronize(device)
        iteration += 1

        if report_every > 0 and iteration % report_every == 0:
            now = time.time()
            elapsed = now - last_report
            total_elapsed = now - start_time
            print(
                f"[gpu-keepalive] gpu={device_idx} iter={iteration} "
                f"burst_time_s={now - step_start:.3f} since_last_report_s={elapsed:.3f} "
                f"total_s={total_elapsed:.1f}",
                flush=True,
            )
            last_report = now

        if sleep_s > 0:
            time.sleep(sleep_s)

    torch.cuda.synchronize(device)
    print(f"[gpu-keepalive] gpu={device_idx} stopping after {iteration} iterations", flush=True)


def launch_workers(devices: Iterable[int], args: argparse.Namespace) -> list[mp.Process]:
    workers = []
    for device_idx in devices:
        proc = mp.Process(
            target=keepalive_worker,
            args=(
                device_idx,
                args.size,
                args.work_iters,
                args.sleep_s,
                args.report_every,
                args.dtype,
            ),
            daemon=False,
        )
        proc.start()
        workers.append(proc)
    return workers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sustain moderate GPU load on selected CUDA devices."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU indices to use. Default: all visible GPUs.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=6144,
        help="Square GEMM size per worker. Larger values increase load and memory use.",
    )
    parser.add_argument(
        "--work-iters",
        type=int,
        default=2,
        help="Number of GEMM bursts to run before each sleep.",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.35,
        help="Sleep seconds between compute bursts. Increase this to lower utilization.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=20,
        help="Print one log line every N bursts per GPU.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Compute dtype to use.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 1

    total = torch.cuda.device_count()
    devices = parse_visible_devices(args.gpus, total)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(
        f"[gpu-keepalive] visible_cuda_devices={total} selected={devices} "
        f"pid={os.getpid()}",
        flush=True,
    )

    mp.set_start_method("spawn", force=True)
    workers = launch_workers(devices, args)

    try:
        while not STOP:
            time.sleep(1.0)
            for proc in workers:
                if not proc.is_alive() and proc.exitcode not in (0, None):
                    print(
                        f"[gpu-keepalive] worker pid={proc.pid} exitcode={proc.exitcode}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
    finally:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join(timeout=10)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
