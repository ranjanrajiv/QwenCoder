"""Metrics and GPU-memory persistence during training (spec 09 sections 55, 78, 84, 93).

Console output is not a record. Every log the trainer emits is appended to
``metrics/metrics.jsonl`` as it happens, so a run that is interrupted still leaves behind
everything it measured up to that point.

**DPO reward metric names are passed through, not enumerated.** The spec (section 78)
notes they vary with the TRL version — ``rewards/chosen``, ``rewards/margins`` and friends
are what TRL 1.10 emits, but hard-coding that list would silently drop metrics on any
other version. Whatever the trainer logs is what gets recorded.

Candidate code is never logged (section 55): only numbers reach this file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("python_dpo.training.callbacks")

# Spec section 79: the metric that actually answers "is chosen pulling ahead of rejected?"
REWARD_MARGIN_KEY = "rewards/margins"


def gpu_memory_snapshot() -> dict[str, int]:
    """Allocated/reserved/peak GPU memory (spec section 84), empty without CUDA."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def reset_peak_memory_stats() -> None:
    """Reset peak counters so a run's peak reflects that run, not the process."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class JsonlMetricsRecorder:
    """Appends training logs to ``metrics.jsonl`` (sections 78, 93).

    Kept independent of ``transformers.TrainerCallback`` so it can be unit-tested without
    importing the backend; :func:`build_metrics_callback` wraps it in a real callback.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []

    def record(self, logs: dict[str, Any], *, step: int | None = None,
               epoch: float | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {"step": step, "epoch": epoch}
        # Pass every scalar through untouched; non-numeric values are dropped rather than
        # serialized, which is what keeps candidate code out of the metrics file (§55).
        for key, value in logs.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            row[key] = value
        row["gpu_memory"] = gpu_memory_snapshot()

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.rows.append(row)
        return row

    def final_metrics(self) -> dict[str, Any]:
        """The last observed value of each metric, for the final report (section 94)."""
        result: dict[str, Any] = {}
        for row in self.rows:
            for key, value in row.items():
                if key in ("step", "epoch", "gpu_memory"):
                    continue
                result[key] = value
        return result

    def peak_gpu_memory(self) -> int | None:
        peaks = [
            row["gpu_memory"].get("peak_allocated_bytes", 0)
            for row in self.rows
            if isinstance(row.get("gpu_memory"), dict)
        ]
        return max(peaks) if peaks else None

    def reward_metrics(self) -> dict[str, float]:
        """Only the DPO reward family, for the section 79/94 summary."""
        final = self.final_metrics()
        return {k: v for k, v in final.items() if k.startswith("rewards/")}


def build_metrics_callback(recorder: JsonlMetricsRecorder):
    """Wrap a recorder in a real ``TrainerCallback``. Imports transformers lazily."""
    from transformers import TrainerCallback

    class _MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not logs:
                return
            recorder.record(
                dict(logs),
                step=getattr(state, "global_step", None),
                epoch=getattr(state, "epoch", None),
            )

    return _MetricsCallback()


__all__ = [
    "REWARD_MARGIN_KEY",
    "JsonlMetricsRecorder",
    "build_metrics_callback",
    "gpu_memory_snapshot",
    "reset_peak_memory_stats",
]
