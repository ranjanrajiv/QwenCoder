"""Training-curve diagnostics (spec 11 sections 85-87).

Over- and under-training are *trend* properties: they need a sequence of logged steps to
be visible at all. With fewer than two points the honest answer is ``insufficient_data``,
reported as such rather than as a fabricated verdict -- which is the answer on this
project's real run, whose ``metrics.jsonl`` holds a single step.

Section 87's preference-overfitting check compares held-out performance against
*training-set* performance, which Stage 10 deliberately does not measure (its section 69
scopes evaluation to held-out problems only). So that check reports ``not_applicable`` with
the reason attached, exactly as Stage 10's own report does for the same gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..atomic_io import JsonlError, iter_jsonl

VERDICTS = ("insufficient_data", "healthy", "undertrained", "overtrained")


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def analyse_training_curve(metrics_path: Path, final_report: Any | None = None) -> dict[str, Any]:
    """Sections 85-87 over one training run's ``metrics/metrics.jsonl``."""
    rows: list[dict[str, Any]] = []
    try:
        rows = [record for _, record in iter_jsonl(Path(metrics_path))]
    except (JsonlError, OSError):
        rows = []

    train_losses = [
        (r.get("step"), _numeric(r.get("loss")))
        for r in rows
        if r.get("loss") is not None
    ]
    train_losses = [(s, v) for s, v in train_losses if v is not None]
    eval_losses = [
        (r.get("step"), _numeric(r.get("eval_loss")))
        for r in rows
        if r.get("eval_loss") is not None
    ]
    eval_losses = [(s, v) for s, v in eval_losses if v is not None]

    distinct_steps = {step for step, _ in train_losses if step is not None}

    result: dict[str, Any] = {
        "logged_rows": len(rows),
        "distinct_steps": len(distinct_steps),
        "train_loss_points": len(train_losses),
        "eval_loss_points": len(eval_losses),
        "first_train_loss": train_losses[0][1] if train_losses else None,
        "final_train_loss": train_losses[-1][1] if train_losses else None,
        "final_eval_loss": eval_losses[-1][1] if eval_losses else None,
        "preference_overfitting": "not_applicable",
        "preference_overfitting_reason": (
            "Stage 10 evaluates held-out problems only and does not measure training-set "
            "performance, so no train-vs-held-out gap can be computed (spec section 69)"
        ),
    }

    if len(distinct_steps) < 2:
        result["verdict"] = "insufficient_data"
        result["reason"] = (
            f"{len(distinct_steps)} distinct training step(s) logged; over- and "
            "undertraining are trend properties and need at least two"
        )
        if final_report is not None:
            result["reward_metrics"] = getattr(final_report, "reward_metrics", None)
        return result

    first, last = train_losses[0][1], train_losses[-1][1]
    if last > first:
        result["verdict"] = "undertrained"
        result["reason"] = "training loss rose from first to last logged step"
    elif eval_losses and len(eval_losses) >= 2 and eval_losses[-1][1] > eval_losses[0][1]:
        result["verdict"] = "overtrained"
        result["reason"] = "evaluation loss rose while training loss fell"
    else:
        result["verdict"] = "healthy"
        result["reason"] = "training loss fell and evaluation loss did not rise"
    return result


__all__ = ["VERDICTS", "analyse_training_curve"]
