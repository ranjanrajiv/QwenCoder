"""Tests for metrics aggregation, success criteria, and report rendering (spec sections
83-88, 124-126, 141, 143)."""

from __future__ import annotations

from python_dpo.model_evaluation.config import SuccessCriteria
from python_dpo.model_evaluation.models import EvaluationRecord, GenerationRecord
from python_dpo.model_evaluation.report import (
    build_failure_analysis,
    build_metrics_summary,
    evaluate_success_criteria,
    render_executive_summary,
)
from python_dpo.model_evaluation.statistics import BootstrapResult


def gen_record(variant: str, problem_id: str, sample_index: int, *, extracted: bool = True) -> GenerationRecord:
    if extracted:
        return GenerationRecord(
            evaluation_run_id="eval_x",
            problem_id=problem_id,
            model_variant=variant,
            sample_index=sample_index,
            seed=1000,
            prompt_sha256="a" * 64,
            raw_response="```python\ndef f(): pass\n```",
            extraction_format="python_fence",
            generation_time_ms=100,
            generated_tokens=5,
            status="generated",
            extracted_code="def f(): pass",
            syntax_valid=True,
        )
    return GenerationRecord(
        evaluation_run_id="eval_x",
        problem_id=problem_id,
        model_variant=variant,
        sample_index=sample_index,
        seed=1000,
        prompt_sha256="a" * 64,
        raw_response="nope",
        extraction_format="unknown",
        generation_time_ms=100,
        generated_tokens=5,
        status="generation_error",
        error="no code",
    )


def eval_record(variant: str, problem_id: str, sample_index: int, *, correct: bool) -> EvaluationRecord:
    if correct:
        return EvaluationRecord(
            evaluation_run_id="eval_x", problem_id=problem_id, model_variant=variant,
            sample_index=sample_index, tests_total=5, tests_passed=5, tests_failed=0,
            tests_error=0, tests_skipped=0, timeout=False, status="passed", duration_ms=50,
        )
    return EvaluationRecord(
        evaluation_run_id="eval_x", problem_id=problem_id, model_variant=variant,
        sample_index=sample_index, tests_total=5, tests_passed=0, tests_failed=5,
        tests_error=0, tests_skipped=0, timeout=False, status="failed", duration_ms=50,
        error_type="assertion_failure",
    )


def test_build_metrics_summary_computes_pass_at_1():
    generation_records = {
        "base": [gen_record("base", "p001", 0), gen_record("base", "p002", 0)],
        "dpo": [gen_record("dpo", "p001", 0), gen_record("dpo", "p002", 0)],
    }
    evaluation_records = {
        "base": [eval_record("base", "p001", 0, correct=True), eval_record("base", "p002", 0, correct=False)],
        "dpo": [eval_record("dpo", "p001", 0, correct=True), eval_record("dpo", "p002", 0, correct=True)],
    }
    summary = build_metrics_summary("eval_x", generation_records, evaluation_records, (1,))
    assert summary.pass_at_k["base"]["1"] == 0.5
    assert summary.pass_at_k["dpo"]["1"] == 1.0
    assert summary.syntax_success_rate["base"] == 1.0


def test_build_metrics_summary_counts_generation_failures():
    generation_records = {
        "base": [gen_record("base", "p001", 0, extracted=False), gen_record("base", "p001", 1, extracted=True)]
    }
    evaluation_records = {"base": [eval_record("base", "p001", 1, correct=True)]}
    summary = build_metrics_summary("eval_x", generation_records, evaluation_records, (1,))
    assert summary.generation_failure_rate["base"] == 0.5


def test_render_executive_summary_reports_improvement():
    generation_records = {"base": [gen_record("base", "p001", 0)], "dpo": [gen_record("dpo", "p001", 0)]}
    evaluation_records = {
        "base": [eval_record("base", "p001", 0, correct=False)],
        "dpo": [eval_record("dpo", "p001", 0, correct=True)],
    }
    summary = build_metrics_summary("eval_x", generation_records, evaluation_records, (1,))
    text = render_executive_summary(summary)
    assert "improved" in text.lower()


def test_render_executive_summary_reports_no_improvement():
    generation_records = {"base": [gen_record("base", "p001", 0)], "dpo": [gen_record("dpo", "p001", 0)]}
    evaluation_records = {
        "base": [eval_record("base", "p001", 0, correct=True)],
        "dpo": [eval_record("dpo", "p001", 0, correct=False)],
    }
    summary = build_metrics_summary("eval_x", generation_records, evaluation_records, (1,))
    text = render_executive_summary(summary)
    assert "did not improve" in text.lower()


def test_build_failure_analysis_categorizes_by_error_type():
    generation_records = {
        "base": [gen_record("base", "p001", 0, extracted=False), gen_record("base", "p002", 0)]
    }
    evaluation_records = {"base": [eval_record("base", "p002", 0, correct=False)]}
    analysis = build_failure_analysis(generation_records, evaluation_records)
    assert analysis["base"]["generation_error"] == 1
    assert analysis["base"]["assertion_failure"] == 1


def test_evaluate_success_criteria_true_when_all_clauses_pass():
    from python_dpo.model_evaluation.models import MetricsSummary

    summary = MetricsSummary(
        evaluation_run_id="eval_x",
        pass_at_k={"base": {"1": 0.40, "5": 0.60}, "dpo": {"1": 0.48, "5": 0.62}},
        test_pass_rate={"base": 0.5, "dpo": 0.6},
        solve_rate={"base": 0.4, "dpo": 0.48},
        syntax_success_rate={"base": 0.97, "dpo": 0.98},
        execution_success_rate={"base": 0.9, "dpo": 0.92},
        timeout_rate={"base": 0.02, "dpo": 0.015},
        generation_failure_rate={"base": 0.0, "dpo": 0.0},
        test_failure_distribution={"base": {}, "dpo": {}},
        computed_at="2026-08-18T14:00:00Z",
    )
    paired_ci = BootstrapResult(point=0.08, lower=0.01, upper=0.15, iterations=1000, seed=42, confidence=0.95)
    result = evaluate_success_criteria(summary, paired_ci, SuccessCriteria())
    assert result.dpo_success is True
    assert result.clauses["pass_at_1_improves"] is True


def test_evaluate_success_criteria_false_on_catastrophic_regression():
    from python_dpo.model_evaluation.models import MetricsSummary

    summary = MetricsSummary(
        evaluation_run_id="eval_x",
        pass_at_k={"base": {"1": 0.40, "5": 0.60}, "dpo": {"1": 0.48, "5": 0.62}},
        test_pass_rate={"base": 0.5, "dpo": 0.6},
        solve_rate={"base": 0.4, "dpo": 0.48},
        syntax_success_rate={"base": 0.97, "dpo": 0.60},  # syntax collapsed
        execution_success_rate={"base": 0.9, "dpo": 0.5},
        timeout_rate={"base": 0.02, "dpo": 0.015},
        generation_failure_rate={"base": 0.0, "dpo": 0.0},
        test_failure_distribution={"base": {}, "dpo": {}},
        computed_at="2026-08-18T14:00:00Z",
    )
    paired_ci = BootstrapResult(point=0.08, lower=0.01, upper=0.15, iterations=1000, seed=42, confidence=0.95)
    result = evaluate_success_criteria(summary, paired_ci, SuccessCriteria())
    assert result.clauses["catastrophic_regression_detected"] is True
    assert result.dpo_success is False


def test_evaluate_success_criteria_false_when_ci_straddles_zero_below():
    from python_dpo.model_evaluation.models import MetricsSummary

    summary = MetricsSummary(
        evaluation_run_id="eval_x",
        pass_at_k={"base": {"1": 0.40}, "dpo": {"1": 0.48}},
        test_pass_rate={"base": 0.5, "dpo": 0.6},
        solve_rate={"base": 0.4, "dpo": 0.48},
        syntax_success_rate={"base": 0.97, "dpo": 0.98},
        execution_success_rate={"base": 0.9, "dpo": 0.92},
        timeout_rate={"base": 0.02, "dpo": 0.015},
        generation_failure_rate={"base": 0.0, "dpo": 0.0},
        test_failure_distribution={"base": {}, "dpo": {}},
        computed_at="2026-08-18T14:00:00Z",
    )
    # CI entirely below zero -- "strongly supports a regression".
    paired_ci = BootstrapResult(point=0.08, lower=-0.10, upper=-0.01, iterations=1000, seed=42, confidence=0.95)
    result = evaluate_success_criteria(summary, paired_ci, SuccessCriteria())
    assert result.clauses["paired_ci_supports_improvement"] is False
    assert result.dpo_success is False
