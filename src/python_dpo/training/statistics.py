"""Text rendering for training preflight and reports (spec 09 sections 19, 27, 34, 94).

Formatters only. The records themselves live in ``models.py``/``hardware.py``/
``lengths.py``, following the convention Stage 7 established: a formatter that two
commands share belongs in the package, not inlined in the CLI.
"""

from __future__ import annotations

from typing import Any

from .dataset import TrainingDataset
from .lengths import LengthAnalysis
from .loader import ParameterCounts
from .models import FinalReport

_BYTES_PER_GIB = 1024**3


def format_dataset_statistics(dataset: TrainingDataset) -> str:
    """Spec sections 27, 105, 106."""
    lines = [
        f"Preference run: {dataset.preference_run_id}",
        f"  policy: {dataset.provenance['selection_policy']} "
        f"({dataset.provenance['selection_policy_version']})",
        f"  training examples:   {dataset.split_counts['train']}",
        f"  validation examples: {dataset.split_counts['validation']}",
        f"  test examples:       {dataset.split_counts['test']}  (held out, never trained on)",
        f"  unique problems:     train={len(dataset.split_problem_ids['train'])} "
        f"validation={len(dataset.split_problem_ids['validation'])} "
        f"test={len(dataset.split_problem_ids['test'])}",
    ]
    for split in ("train", "validation"):
        stats = dataset.statistics.get(split, {})
        if not stats.get("count"):
            continue
        lines.append(f"  {split} characters:")
        lines.append(
            f"    prompt   mean {stats['mean_prompt_chars']:>8.1f}  max {stats['max_prompt_chars']}"
        )
        lines.append(
            f"    chosen   mean {stats['mean_chosen_chars']:>8.1f}  max {stats['max_chosen_chars']}"
        )
        lines.append(
            f"    rejected mean {stats['mean_rejected_chars']:>8.1f}  max {stats['max_rejected_chars']}"
        )

    balance = dataset.balance
    if balance.pairs:
        lines += [
            "  source pair balance:",
            f"    mean chosen score:   {balance.mean_chosen_score:.4f}",
            f"    mean rejected score: {balance.mean_rejected_score:.4f}",
            f"    mean score margin:   {balance.mean_score_margin:.4f}",
            f"    strong pairs:        {balance.strong_pair_percentage:.1f}%",
            f"    medium pairs:        {balance.medium_pair_percentage:.1f}%",
        ]
    return "\n".join(lines) + "\n"


def format_length_analysis(analysis: LengthAnalysis) -> str:
    """Spec sections 34, 35."""
    header = f"{'QUANTITY':<16}{'P50':>8}{'P90':>8}{'P95':>8}{'P99':>8}{'MAX':>8}"
    lines = ["Token lengths:", header]
    for name in ("prompt", "chosen", "rejected", "prompt_chosen", "prompt_rejected"):
        dist = analysis.distributions[name]
        lines.append(
            f"{name:<16}{dist.p50:>8}{dist.p90:>8}{dist.p95:>8}{dist.p99:>8}{dist.maximum:>8}"
        )
    lines.append(
        f"Truncation: {analysis.truncated_examples}/{analysis.examples} example(s) "
        f"({100 * analysis.truncation_rate:.1f}%) exceed max_length={analysis.max_length}"
    )
    if analysis.prompt_overflow_examples:
        lines.append(
            f"  {analysis.prompt_overflow_examples} prompt(s) alone exceed "
            f"max_prompt_length={analysis.max_prompt_length}"
        )
    return "\n".join(lines) + "\n"


def format_parameter_counts(counts: ParameterCounts) -> str:
    """Spec section 19."""
    return (
        f"Total parameters:     {counts.total:>15,}\n"
        f"Trainable parameters: {counts.trainable:>15,}\n"
        f"Trainable percentage: {counts.percentage:>14.4f}%\n"
    )


def format_final_report(report: FinalReport) -> str:
    """Spec section 94."""
    lines = [
        f"Training run:  {report.training_run_id}",
        f"Experiment:    {report.experiment_name}",
        f"Status:        {report.status}",
        f"Model:         {report.model_name}"
        + (f" @ {report.model_revision}" if report.model_revision else ""),
        f"Dataset:       {report.preference_run_id}",
        f"Examples:      {report.number_of_examples}",
        f"Epochs:        {report.epochs}",
        f"Steps:         {report.steps}",
        f"Effective batch size: {report.effective_batch_size}",
        f"Optimizer:     {report.optimizer}",
        f"Compute dtype: {report.compute_dtype}",
        f"Trainable:     {report.trainable_parameters:,} of {report.total_parameters:,} "
        f"({report.trainable_percentage:.4f}%)",
    ]
    if report.final_train_loss is not None:
        lines.append(f"Final train loss: {report.final_train_loss:.6f}")
    if report.final_eval_loss is not None:
        lines.append(f"Final eval loss:  {report.final_eval_loss:.6f}")
    if report.reward_metrics:
        lines.append("DPO reward metrics:")
        for key in sorted(report.reward_metrics):
            lines.append(f"  {key}: {report.reward_metrics[key]:.6f}")
    if report.peak_gpu_memory_bytes:
        lines.append(
            f"Peak GPU memory: {report.peak_gpu_memory_bytes / _BYTES_PER_GIB:.2f} GiB"
        )
    if report.training_duration_seconds is not None:
        lines.append(f"Duration:      {report.training_duration_seconds:.1f}s")
    lines.append(f"Adapter:       {report.adapter_path}")
    lines.append(f"Adapter reload: {'OK' if report.adapter_reload_ok else 'NOT VERIFIED'}")
    return "\n".join(lines) + "\n"


def format_run_table(manifests: list[Any]) -> str:
    """``train list`` — one row per training run, newest first."""
    header = (
        f"{'TRAINING_RUN_ID':<28}{'EXPERIMENT':<26}{'MODE':<12}{'STATUS':<12}{'CREATED_AT'}"
    )
    lines = [header]
    for m in manifests:
        lines.append(
            f"{m.training_run_id:<28}{m.experiment_name:<26}{m.mode:<12}"
            f"{m.status:<12}{m.created_at}"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "format_dataset_statistics",
    "format_final_report",
    "format_length_analysis",
    "format_parameter_counts",
    "format_run_table",
]
