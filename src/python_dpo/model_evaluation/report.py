"""Aggregate metrics, the base-vs-DPO comparison report, and failure analysis (spec
sections 83-85, 124-132, 141, 149).

Every number in the rendered report is computed from persisted
:class:`~python_dpo.model_evaluation.models.GenerationRecord`/:class:`~python_dpo.model_evaluation.models.EvaluationRecord`
lists, never from in-memory counters accumulated during a run -- so ``evaluate-model
report`` reproduces the same numbers whether it runs immediately after generation or a
week later against the same artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .comparison import ComparisonResult
from .config import SuccessCriteria
from .metrics import (
    execution_success_rate,
    generation_failure_rate,
    mean_pass_at_k,
    mean_test_pass_rate,
    solve_rate as solve_rate_metric,
    syntax_success_rate,
    test_failure_distribution,
    timeout_rate as timeout_rate_metric,
)
from .models import EvaluationRecord, GenerationRecord, MetricsSummary, utc_now_iso
from .statistics import BootstrapResult

VARIANT_LABELS = {"base": "Base Qwen", "dpo": "DPO Qwen"}


def build_metrics_summary(
    evaluation_run_id: str,
    generation_records: dict[str, list[GenerationRecord]],
    evaluation_records: dict[str, list[EvaluationRecord]],
    pass_at_k_values: tuple[int, ...],
) -> MetricsSummary:
    """Compute every spec section 41-53 metric for each variant present (spec section 83)."""
    pass_at_k: dict[str, dict[str, float]] = {}
    test_pass_rate: dict[str, float] = {}
    solve_rate: dict[str, float] = {}
    syntax_success: dict[str, float] = {}
    execution_success: dict[str, float] = {}
    timeout_rate: dict[str, float] = {}
    gen_failure_rate: dict[str, float] = {}
    failure_dist: dict[str, dict[str, int]] = {}

    for variant, gen_records in generation_records.items():
        eval_records = evaluation_records.get(variant, [])
        eval_by_candidate = {r.candidate_id: r for r in eval_records}
        by_problem: dict[str, list[EvaluationRecord]] = {}
        for record in eval_records:
            by_problem.setdefault(record.problem_id, []).append(record)

        generated = [g for g in gen_records if g.status == "generated"]
        valid_records = [r for r in eval_records if not r.is_infrastructure_error]

        variant_pass_at_k: dict[str, float] = {}
        for k in pass_at_k_values:
            per_problem_nc = []
            for records in by_problem.values():
                valid = [r for r in records if not r.is_infrastructure_error]
                if len(valid) >= k:
                    per_problem_nc.append((len(valid), sum(1 for r in valid if r.correct)))
            variant_pass_at_k[str(k)] = mean_pass_at_k(per_problem_nc, k)
        pass_at_k[variant] = variant_pass_at_k

        test_pass_rate[variant] = mean_test_pass_rate([r.pass_rate for r in valid_records])

        solved_flags = [
            any(r.correct for r in records if not r.is_infrastructure_error)
            for records in by_problem.values()
        ]
        solve_rate[variant] = solve_rate_metric(solved_flags)

        syntax_success[variant] = syntax_success_rate([bool(g.syntax_valid) for g in generated])

        exec_flags = []
        for g in generated:
            record = eval_by_candidate.get(g.candidate_id)
            if record is None:
                continue
            exec_flags.append(
                record.status not in ("syntax_error", "infrastructure_error")
                and record.tests_error == 0
            )
        execution_success[variant] = execution_success_rate(exec_flags)

        timeout_flags = [
            bool(eval_by_candidate.get(g.candidate_id) and eval_by_candidate[g.candidate_id].status == "timeout")
            for g in generated
        ]
        timeout_rate[variant] = timeout_rate_metric(timeout_flags)

        gen_failure_rate[variant] = generation_failure_rate(
            [g.status == "generation_error" for g in gen_records]
        )

        failure_dist[variant] = test_failure_distribution([r.pass_rate for r in valid_records])

    return MetricsSummary(
        evaluation_run_id=evaluation_run_id,
        pass_at_k=pass_at_k,
        test_pass_rate=test_pass_rate,
        solve_rate=solve_rate,
        syntax_success_rate=syntax_success,
        execution_success_rate=execution_success,
        timeout_rate=timeout_rate,
        generation_failure_rate=gen_failure_rate,
        test_failure_distribution=failure_dist,
        computed_at=utc_now_iso(),
    )


def build_failure_analysis(
    generation_records: dict[str, list[GenerationRecord]],
    evaluation_records: dict[str, list[EvaluationRecord]],
) -> dict[str, dict[str, int]]:
    """Spec sections 124-126: deterministic failure categorization, no LLM."""
    categories = (
        "generation_error", "syntax_error", "import_error", "runtime_error",
        "assertion_failure", "timeout", "infrastructure_error",
    )
    result: dict[str, dict[str, int]] = {}
    for variant, gen_records in generation_records.items():
        counts = {name: 0 for name in categories}
        counts["generation_error"] = sum(1 for g in gen_records if g.status == "generation_error")
        for record in evaluation_records.get(variant, []):
            if record.error_type:
                counts[record.error_type] = counts.get(record.error_type, 0) + 1
        result[variant] = counts
    return result


def _first(records: list, problem_id: str):
    candidates = [r for r in records if r.problem_id == problem_id]
    return min(candidates, key=lambda r: r.sample_index) if candidates else None


def classify_problem_examples(
    comparison: ComparisonResult,
    generation_records: dict[str, list[GenerationRecord]],
    evaluation_records: dict[str, list[EvaluationRecord]],
    prompts_by_problem: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Spec sections 130-132: one representative sample per problem, for qualitative
    review (spec section 133) -- never used to modify the model or the benchmark."""
    improvements: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    ties: list[dict[str, Any]] = []

    for pc in comparison.per_problem:
        base_gen = _first(generation_records.get("base", []), pc.problem_id)
        dpo_gen = _first(generation_records.get("dpo", []), pc.problem_id)
        base_eval = _first(evaluation_records.get("base", []), pc.problem_id)
        dpo_eval = _first(evaluation_records.get("dpo", []), pc.problem_id)
        row = {
            "problem_id": pc.problem_id,
            "prompt": prompts_by_problem.get(pc.problem_id, ""),
            "base_code": base_gen.extracted_code if base_gen else None,
            "dpo_code": dpo_gen.extracted_code if dpo_gen else None,
            "base_test_result": base_eval.to_dict() if base_eval else None,
            "dpo_test_result": dpo_eval.to_dict() if dpo_eval else None,
        }
        if pc.improvement == 1:
            improvements.append(row)
        elif pc.improvement == -1:
            regressions.append(row)
        else:
            ties.append(row)

    return improvements, regressions, ties


@dataclass(frozen=True)
class SuccessEvaluation:
    """Spec sections 86-88, 143: the configurable success gate, legible clause by clause."""

    clauses: dict[str, bool | str]
    dpo_success: bool

    def to_dict(self) -> dict[str, Any]:
        return {**self.clauses, "DPO_SUCCESS": self.dpo_success}


def evaluate_success_criteria(
    summary: MetricsSummary,
    paired_pass_at_1_ci: BootstrapResult | None,
    criteria: SuccessCriteria,
) -> SuccessEvaluation:
    """Spec section 143's default success rule, evaluated clause by clause so a ``false``
    verdict is legible rather than a bare boolean."""
    base_pk = summary.pass_at_k.get("base", {})
    dpo_pk = summary.pass_at_k.get("dpo", {})
    base_p1, dpo_p1 = base_pk.get("1"), dpo_pk.get("1")
    base_p5, dpo_p5 = base_pk.get("5"), dpo_pk.get("5")
    base_syntax = summary.syntax_success_rate.get("base")
    dpo_syntax = summary.syntax_success_rate.get("dpo")
    base_timeout = summary.timeout_rate.get("base")
    dpo_timeout = summary.timeout_rate.get("dpo")

    clauses: dict[str, bool | str] = {}
    clauses["pass_at_1_improves"] = (
        base_p1 is not None and dpo_p1 is not None
        and (dpo_p1 - base_p1) >= criteria.minimum_pass_at_1_improvement
    )
    clauses["pass_at_5_not_regressed"] = (
        base_p5 is None or dpo_p5 is None
        or dpo_p5 >= base_p5 - criteria.maximum_allowed_regression
    )
    clauses["syntax_success_not_regressed"] = (
        base_syntax is None or dpo_syntax is None
        or dpo_syntax >= base_syntax - criteria.maximum_allowed_regression
    )
    clauses["timeout_rate_not_increased"] = (
        base_timeout is None or dpo_timeout is None
        or dpo_timeout <= base_timeout + criteria.maximum_allowed_regression
    )
    # Spec section 62: "does not strongly support a regression" -- the interval must not
    # sit entirely below zero.
    clauses["paired_ci_supports_improvement"] = (
        paired_pass_at_1_ci is None or paired_pass_at_1_ci.upper >= 0
    )
    # Spec section 71: pass@1 improved while another major metric collapsed.
    clauses["catastrophic_regression_detected"] = bool(
        clauses["pass_at_1_improves"]
        and not (clauses["syntax_success_not_regressed"] and clauses["timeout_rate_not_increased"])
    )
    # Spec section 70 needs training-set performance, which spec section 69 forbids
    # evaluating in this stage -- reported as not applicable rather than fabricated.
    clauses["overfitting_check"] = (
        "not_applicable: Stage 10 does not evaluate training-set performance (spec section 69)"
    )

    dpo_success = bool(
        clauses["pass_at_1_improves"]
        and clauses["pass_at_5_not_regressed"]
        and clauses["syntax_success_not_regressed"]
        and clauses["timeout_rate_not_increased"]
        and clauses["paired_ci_supports_improvement"]
        and not clauses["catastrophic_regression_detected"]
    )
    return SuccessEvaluation(clauses=clauses, dpo_success=dpo_success)


def render_executive_summary(summary: MetricsSummary) -> str:
    """Spec section 85: states plainly what was/was not demonstrated, no causal language."""
    base_pk = summary.pass_at_k.get("base", {})
    dpo_pk = summary.pass_at_k.get("dpo", {})
    if not base_pk or not dpo_pk:
        return "Insufficient data to compare pass@1 -- one or both model variants were not evaluated.\n"

    k = "1" if "1" in base_pk and "1" in dpo_pk else sorted(base_pk, key=int)[0]
    base_v, dpo_v = base_pk[k], dpo_pk[k]
    diff_pp = (dpo_v - base_v) * 100

    if dpo_v > base_v:
        return (
            f"DPO improved pass@{k} from {base_v * 100:.1f}% to {dpo_v * 100:.1f}%, "
            f"an absolute improvement of {diff_pp:.1f} percentage points.\n"
        )
    if dpo_v < base_v:
        return f"DPO did not improve pass@{k}.\nBase: {base_v * 100:.1f}%\nDPO: {dpo_v * 100:.1f}%\n"
    return f"DPO showed no measurable difference in pass@{k} ({base_v * 100:.1f}% for both models).\n"


def _percentile(sorted_values: list[int], percentile: int) -> int:
    if not sorted_values:
        return 0
    rank = max(1, min(len(sorted_values), round(percentile / 100 * len(sorted_values))))
    return sorted_values[rank - 1]


def _latency_stats(records: list[GenerationRecord]) -> dict[str, float]:
    times = sorted(r.generation_time_ms for r in records)
    if not times:
        return {"mean_ms": 0.0, "p50_ms": 0, "p95_ms": 0}
    return {
        "mean_ms": sum(times) / len(times),
        "p50_ms": _percentile(times, 50),
        "p95_ms": _percentile(times, 95),
    }


def _token_throughput(records: list[GenerationRecord]) -> float:
    total_tokens = sum(r.generated_tokens for r in records)
    total_ms = sum(r.generation_time_ms for r in records)
    return (total_tokens / (total_ms / 1000)) if total_ms > 0 else 0.0


def render_markdown_report(
    *,
    evaluation_run_id: str,
    benchmark_version: str,
    benchmark_problem_count: int,
    base_model_name: str,
    base_model_revision: str | None,
    adapter_path: str,
    training_run_id: str,
    generation_config: dict[str, Any],
    summary: MetricsSummary,
    comparison: ComparisonResult,
    pass_at_k_cis: dict[str, dict[str, BootstrapResult]],
    paired_pass_at_1_ci: BootstrapResult | None,
    mcnemar_result: Any | None,
    success: SuccessEvaluation,
    failure_analysis: dict[str, dict[str, int]],
    generation_records: dict[str, list[GenerationRecord]],
    peak_gpu_memory_bytes: dict[str, int] | None = None,
) -> str:
    """Assemble the full spec section 84 report."""
    peak_gpu_memory_bytes = peak_gpu_memory_bytes or {}
    lines: list[str] = []
    lines.append("# Python DPO Evaluation")
    lines.append("")
    lines.append(f"Evaluation run: `{evaluation_run_id}`")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(render_executive_summary(summary))

    lines.append("## Benchmark Description")
    lines.append("")
    lines.append(f"- Benchmark: `{benchmark_version}`")
    lines.append(f"- Problems: {benchmark_problem_count}")
    lines.append(f"- Paired problems (evaluated for both variants): {comparison.paired_problems}")
    lines.append(
        f"- Base successfully evaluated: {comparison.base_successfully_evaluated} / "
        f"DPO successfully evaluated: {comparison.dpo_successfully_evaluated}"
    )
    lines.append("")

    lines.append("## Model Configuration")
    lines.append("")
    lines.append(f"- Base model: `{base_model_name}` (revision: `{base_model_revision or 'default'}`)")
    lines.append(f"- Adapter: `{adapter_path}`")
    lines.append(f"- Training run: `{training_run_id}`")
    lines.append("")

    lines.append("## Generation Configuration")
    lines.append("")
    for key, value in sorted(generation_config.items()):
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines.append("## pass@k Results")
    lines.append("")
    lines.append("| k | Base | DPO | 95% CI (Base) | 95% CI (DPO) |")
    lines.append("|---|------|-----|----------------|----------------|")
    base_pk = summary.pass_at_k.get("base", {})
    dpo_pk = summary.pass_at_k.get("dpo", {})
    for k in sorted(set(base_pk) | set(dpo_pk), key=int):
        base_v = base_pk.get(k)
        dpo_v = dpo_pk.get(k)
        base_ci = pass_at_k_cis.get("base", {}).get(k)
        dpo_ci = pass_at_k_cis.get("dpo", {}).get(k)
        base_str = f"{base_v * 100:.1f}%" if base_v is not None else "n/a"
        dpo_str = f"{dpo_v * 100:.1f}%" if dpo_v is not None else "n/a"
        base_ci_str = f"[{base_ci.lower * 100:.1f}%, {base_ci.upper * 100:.1f}%]" if base_ci else "n/a"
        dpo_ci_str = f"[{dpo_ci.lower * 100:.1f}%, {dpo_ci.upper * 100:.1f}%]" if dpo_ci else "n/a"
        lines.append(f"| {k} | {base_str} | {dpo_str} | {base_ci_str} | {dpo_ci_str} |")
    lines.append("")

    lines.append("## Test Pass Rate")
    lines.append("")
    lines.append(
        f"- Base mean test pass rate: {summary.test_pass_rate.get('base', 0.0) * 100:.1f}%"
    )
    lines.append(
        f"- DPO mean test pass rate: {summary.test_pass_rate.get('dpo', 0.0) * 100:.1f}%"
    )
    lines.append("")
    lines.append("Test failure distribution (Base): " + str(summary.test_failure_distribution.get("base", {})))
    lines.append("")
    lines.append("Test failure distribution (DPO): " + str(summary.test_failure_distribution.get("dpo", {})))
    lines.append("")

    lines.append("## Win/Tie/Loss")
    lines.append("")
    lines.append(f"- DPO wins: {comparison.dpo_wins}")
    lines.append(f"- Ties: {comparison.ties}")
    lines.append(f"- DPO losses: {comparison.dpo_losses}")
    lines.append(f"- DPO win rate (ties excluded): {comparison.dpo_win_rate * 100:.1f}%")
    lines.append("")

    lines.append("## Statistical Analysis")
    lines.append("")
    if paired_pass_at_1_ci is not None:
        lines.append(
            f"- Paired bootstrap, pass@1 difference (DPO - Base): "
            f"{paired_pass_at_1_ci.point * 100:+.1f} pp, "
            f"95% CI [{paired_pass_at_1_ci.lower * 100:+.1f}, {paired_pass_at_1_ci.upper * 100:+.1f}] pp "
            f"({paired_pass_at_1_ci.iterations} iterations, seed {paired_pass_at_1_ci.seed})"
        )
    if mcnemar_result is not None:
        lines.append(
            f"- McNemar's exact test: base-only={mcnemar_result.base_only}, "
            f"dpo-only={mcnemar_result.dpo_only}, p={mcnemar_result.p_value:.4f}"
        )
    lines.append("")

    lines.append("## Regression Analysis")
    lines.append("")
    for clause, value in success.clauses.items():
        lines.append(f"- {clause}: {value}")
    lines.append(f"- **DPO_SUCCESS: {success.dpo_success}**")
    lines.append("")

    lines.append("## Latency")
    lines.append("")
    for variant in ("base", "dpo"):
        records = generation_records.get(variant, [])
        if not records:
            continue
        stats = _latency_stats(records)
        throughput = _token_throughput(records)
        lines.append(
            f"- {VARIANT_LABELS[variant]}: mean {stats['mean_ms']:.0f} ms, "
            f"p50 {stats['p50_ms']} ms, p95 {stats['p95_ms']} ms, "
            f"{throughput:.1f} tokens/s"
        )
    lines.append("")

    lines.append("## Memory")
    lines.append("")
    if peak_gpu_memory_bytes:
        for variant, value in peak_gpu_memory_bytes.items():
            lines.append(f"- {VARIANT_LABELS.get(variant, variant)} peak GPU memory: {value / (1024**3):.2f} GiB")
    else:
        lines.append("- Peak GPU memory was not recorded for this run.")
    lines.append("")

    lines.append("## Failure Analysis")
    lines.append("")
    for variant, counts in failure_analysis.items():
        lines.append(f"- {VARIANT_LABELS.get(variant, variant)}: {counts}")
    lines.append("")

    lines.append("## Conclusions")
    lines.append("")
    lines.append(render_executive_summary(summary))
    if comparison.benchmark_problems < 100:
        lines.append(
            f"This benchmark contains only {comparison.benchmark_problems} problem(s). Per spec "
            "section 144, a benchmark this size is suitable for pipeline validation, not for "
            "reliable model-performance conclusions.\n"
        )

    return "\n".join(lines) + "\n"


def build_comparison_report_json(
    *,
    evaluation_run_id: str,
    benchmark_version: str,
    summary: MetricsSummary,
    comparison: ComparisonResult,
    pass_at_k_cis: dict[str, dict[str, BootstrapResult]],
    paired_pass_at_1_ci: BootstrapResult | None,
    mcnemar_result: Any | None,
    success: SuccessEvaluation,
    failure_analysis: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Spec section 84: ``reports/base_vs_dpo.json``."""
    return {
        "evaluation_run_id": evaluation_run_id,
        "benchmark_version": benchmark_version,
        "executive_summary": render_executive_summary(summary),
        "metrics": summary.to_dict(),
        "comparison": comparison.to_dict(),
        "pass_at_k_confidence_intervals": {
            variant: {k: ci.to_dict() for k, ci in by_k.items()}
            for variant, by_k in pass_at_k_cis.items()
        },
        "paired_pass_at_1_ci": paired_pass_at_1_ci.to_dict() if paired_pass_at_1_ci else None,
        "mcnemar": mcnemar_result.to_dict() if mcnemar_result else None,
        "success_criteria": success.to_dict(),
        "failure_analysis": failure_analysis,
        "generated_at": utc_now_iso(),
    }


__all__ = [
    "SuccessEvaluation",
    "build_comparison_report_json",
    "build_failure_analysis",
    "build_metrics_summary",
    "classify_problem_examples",
    "evaluate_success_criteria",
    "render_executive_summary",
    "render_markdown_report",
]
