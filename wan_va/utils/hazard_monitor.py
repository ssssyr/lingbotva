from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


class HazardRunMonitor:
    def __init__(self, run_dir: Path, run_name: str, config, rank: int, world_size: int):
        self.run_dir = Path(run_dir)
        self.run_name = str(run_name)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.config = config
        self.started_at_ts = time.time()
        self.started_at = _now_iso()
        self.base_status = {
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "started_at": self.started_at,
            "hostname": socket.gethostname(),
            "rank": self.rank,
            "world_size": self.world_size,
        }

        monitoring_cfg = getattr(config, "monitoring", None)
        self.heartbeat_interval_seconds = float(
            getattr(monitoring_cfg, "heartbeat_interval_seconds", 30.0)
        )
        self._last_heartbeat_ts = 0.0
        self.latest_checkpoint = None
        self.latest_metrics = None

        self.logs_dir = self.run_dir / "logs"
        self.metrics_dir = self.run_dir / "metrics"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.status_path = self.run_dir / "status.json"
        self.metrics_jsonl_path = self.metrics_dir / "train_metrics.jsonl"
        self.checkpoints_jsonl_path = self.metrics_dir / "checkpoints.jsonl"
        self.run_info_path = self.artifacts_dir / "run_info.json"
        self.resolved_config_path = self.artifacts_dir / "resolved_config.json"

        if self.rank == 0:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(
                self.resolved_config_path,
                _to_jsonable(dict(config)),
            )
            self._write_json(
                self.run_info_path,
                {
                    **self.base_status,
                    "pid": os.getpid(),
                    "train_model_path": str(config.paths.train_model_path),
                    "dataset_path": str(config.paths.dataset_path),
                    "save_root": str(config.paths.save_root),
                    "config_name": getattr(getattr(config, "launcher", None), "config_name", None),
                },
            )
            self._write_status(
                {
                    **self.base_status,
                    "state": "initializing",
                    "last_heartbeat_at": self.started_at,
                    "latest_checkpoint": None,
                    "latest_metrics": None,
                }
            )

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(payload), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def _write_status(self, payload: dict[str, Any]) -> None:
        self._write_json(self.status_path, payload)

    def _build_status_payload(
        self,
        state: str,
        step: int | None = None,
        micro_step: int | None = None,
        max_steps: int | None = None,
        note: str | None = None,
        error: str | None = None,
        traceback_text: str | None = None,
    ) -> dict[str, Any]:
        now_iso = _now_iso()
        payload: dict[str, Any] = {
            **self.base_status,
            "state": state,
            "last_heartbeat_at": now_iso,
            "latest_checkpoint": self.latest_checkpoint,
            "latest_metrics": self.latest_metrics,
        }
        if step is not None:
            payload["step"] = int(step)
        if micro_step is not None:
            payload["micro_step"] = int(micro_step)
        if max_steps is not None:
            payload["max_steps"] = int(max_steps)
            if step is not None and max_steps > 0:
                payload["progress"] = float(step) / float(max_steps)
        if note is not None:
            payload["note"] = note
        if error is not None:
            payload["error"] = error
        if traceback_text is not None:
            payload["traceback"] = traceback_text
        return payload

    def maybe_heartbeat(
        self,
        step: int,
        micro_step: int,
        max_steps: int,
        note: str | None = None,
        force: bool = False,
    ) -> None:
        if self.rank != 0:
            return
        now = time.time()
        if (not force) and (now - self._last_heartbeat_ts < self.heartbeat_interval_seconds):
            return
        self._last_heartbeat_ts = now
        self._write_status(
            self._build_status_payload(
                state="running",
                step=step,
                micro_step=micro_step,
                max_steps=max_steps,
                note=note,
            )
        )

    def log_metrics(
        self,
        step: int,
        micro_step: int,
        max_steps: int,
        metrics: dict[str, Any],
    ) -> None:
        if self.rank != 0:
            return

        elapsed_seconds = max(time.time() - self.started_at_ts, 1e-6)
        steps_per_hour = float(step) / elapsed_seconds * 3600.0 if step > 0 else 0.0
        eta_seconds = None
        if step > 0 and max_steps > step:
            eta_seconds = (max_steps - step) / max(float(step) / elapsed_seconds, 1e-6)

        gpu_mem_alloc_gb = None
        gpu_mem_reserved_gb = None
        if torch.cuda.is_available():
            gpu_mem_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_mem_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

        record = {
            "timestamp": _now_iso(),
            "step": int(step),
            "micro_step": int(micro_step),
            "max_steps": int(max_steps),
            "progress": float(step) / float(max_steps) if max_steps > 0 else 0.0,
            "elapsed_seconds": elapsed_seconds,
            "steps_per_hour": steps_per_hour,
            "eta_seconds": eta_seconds,
            "latest_checkpoint": self.latest_checkpoint,
            "gpu_mem_alloc_gb_rank0": gpu_mem_alloc_gb,
            "gpu_mem_reserved_gb_rank0": gpu_mem_reserved_gb,
            **metrics,
        }

        self.latest_metrics = record
        self._append_jsonl(self.metrics_jsonl_path, record)
        self._write_status(
            self._build_status_payload(
                state="running",
                step=step,
                micro_step=micro_step,
                max_steps=max_steps,
            )
        )
        self._last_heartbeat_ts = time.time()

    def note_checkpoint(self, step: int, checkpoint_path: Path | str, final: bool = False) -> None:
        if self.rank != 0:
            return
        checkpoint_str = str(checkpoint_path)
        self.latest_checkpoint = checkpoint_str
        record = {
            "timestamp": _now_iso(),
            "step": int(step),
            "checkpoint_path": checkpoint_str,
            "final": bool(final),
        }
        self._append_jsonl(self.checkpoints_jsonl_path, record)
        self._write_status(
            self._build_status_payload(
                state="running",
                step=step,
                micro_step=None,
                max_steps=getattr(getattr(self.config, "training", None), "num_steps", None),
                note="checkpoint_saved",
            )
        )

    def mark_finished(self, step: int, max_steps: int) -> None:
        if self.rank != 0:
            return
        payload = self._build_status_payload(
            state="completed",
            step=step,
            micro_step=None,
            max_steps=max_steps,
            note="training_completed",
        )
        payload["finished_at"] = _now_iso()
        self._write_status(payload)

    def mark_failed(
        self,
        step: int,
        micro_step: int,
        max_steps: int,
        error: str,
        traceback_text: str | None = None,
    ) -> None:
        if self.rank != 0:
            return
        payload = self._build_status_payload(
            state="failed",
            step=step,
            micro_step=micro_step,
            max_steps=max_steps,
            note="training_failed",
            error=error,
            traceback_text=traceback_text,
        )
        payload["failed_at"] = _now_iso()
        self._write_status(payload)
