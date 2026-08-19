"""The end-to-end analysis (spec 11 sections 6-9, 121).

One function, :func:`run_analysis`, sequencing every module in the package and persisting
the section 121 artifact tree. It computes; it never trains and never regenerates data
(sections 5, 113) -- the most it produces is ``next_experiment.yaml``, which is a proposal
for a human to act on.
"""

from __future__ import annotations

import logging
from typing import Any

from .classification import build_error_profile, classify_variant, compare_error_rates
from .config import AnalysisConfig
from .coverage import (
    build_gaps,
    correlate_errors_with_coverage,
    preference_coverage,
    strategy_gaps,
    training_problem_ids,
)
from .diversity import build_diversity_report
from .failures import build_test_failure_stats, interesting
from .ingest import AnalysisInputs, load_analysis_inputs
from .outcomes import build_problem_outcomes, partition
from .recommend import build_recommendations, decide_iteration
from .refinement import (
    build_hard_examples,
    build_refined_preferences,
    build_regression_examples,
    build_successful_dpo_examples,
    plan_refinement,
)
from .report import build_summary, render_analysis_md, render_next_experiment
from .run_repository import AnalysisRunRepository
from .training_curve import analyse_training_curve

logger = logging.getLogger("python_dpo.analysis.driver")


def _paired_ci_width(inputs: AnalysisInputs, eval_repo: Any) -> float | None:
    """Section 95's second gate, read from Stage 10's own recorded bootstrap result."""
    metrics = eval_repo.read_metrics(inputs.lineage.evaluation_run_id, "bootstrap") or {}
    paired = metrics.get("paired_pass_at_1") or {}
    lower, upper = paired.get("lower"), paired.get("upper")
    if lower is None or upper is None:
        return None
    return float(upper) - float(lower)


def run_analysis(
    config: Any,
    evaluation_run_id: str,
    *,
    analysis_config: AnalysisConfig | None = None,
    preference_run_id: str | None = None,
    training_run_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run the full analysis and persist it. Returns ``(analysis_run_id, summary)``."""
    from ..model_evaluation.run_repository import ModelEvaluationRunRepository

    analysis_config = analysis_config or AnalysisConfig.load()
    inputs = load_analysis_inputs(
        config, evaluation_run_id,
        preference_run_id=preference_run_id, training_run_id=training_run_id,
    )
    thresholds = analysis_config.thresholds

    repo = AnalysisRunRepository(config.paths.analysis / "runs")
    manifest = repo.create_run(
        lineage=inputs.lineage,
        benchmark_version=getattr(inputs.evaluation_manifest, "benchmark_version", None),
    )
    run_id = manifest.analysis_run_id
    repo.write_config(run_id, analysis_config.to_dict())
    repo.start_run(run_id)
    logger.info("Analysis run %s created for evaluation %s", run_id, evaluation_run_id)

    try:
        # ---- classification (sections 14-16)
        profiles: dict[str, Any] = {}
        for variant in inputs.variants:
            classifications = classify_variant(
                inputs.generations[variant], inputs.evaluations[variant],
                inputs.test_results.get(variant, []),
            )
            repo.write_jsonl(
                run_id, f"classifications/{variant}_errors.jsonl",
                [c.to_dict() for c in classifications],
            )
            profiles[variant] = build_error_profile(variant, classifications)
            repo.write_json(
                run_id, f"classifications/{variant}_error_profile.json",
                profiles[variant].to_dict(),
            )
        if "base" in profiles and "dpo" in profiles:
            repo.write_json(
                run_id, "classifications/error_rate_comparison.json",
                [c.to_dict() for c in compare_error_rates(profiles["base"], profiles["dpo"])],
            )

        # ---- outcomes (sections 17-25)
        outcomes = build_problem_outcomes(
            inputs.evaluations, inputs.problems,
            regression_threshold=thresholds.regression_threshold,
        )
        buckets = partition(outcomes)
        repo.write_jsonl(
            run_id, "improvements/improvements.jsonl",
            [o.to_dict() for o in buckets["improvements"]],
        )
        repo.write_jsonl(
            run_id, "regressions/regressions.jsonl",
            [o.to_dict() for o in buckets["regressions"]],
        )

        # ---- test-level, diversity (sections 26-31, 45-49)
        test_stats = build_test_failure_stats(
            inputs.test_results,
            hard_test_failure_rate=thresholds.hard_test_failure_rate,
            variant_specific_delta=thresholds.variant_specific_test_delta,
        )
        notable = interesting(test_stats)
        repo.write_json(
            run_id, "analysis/test_failures.json", [s.to_dict() for s in notable]
        )
        diversity = build_diversity_report(
            inputs.generations, mode_collapse_reduction=thresholds.mode_collapse_reduction
        )
        repo.write_json(run_id, "analysis/diversity.json", diversity.to_dict())

        # ---- training curve (sections 85-87)
        from ..training.run_repository import TrainingRunRepository

        training_repo = TrainingRunRepository(config.paths.training / "runs")
        curve = analyse_training_curve(
            training_repo.metrics_path(inputs.lineage.training_run_id),
            training_repo.read_final_report(inputs.lineage.training_run_id),
        )
        repo.write_json(run_id, "analysis/training_curve.json", curve)

        # ---- coverage (sections 32-42)
        trained_ids = training_problem_ids(inputs.split_manifest, inputs.preference_pairs)
        category_gaps = build_gaps(
            attribute="category", pairs=inputs.preference_pairs, problems=inputs.problems,
            benchmark_problem_ids=inputs.benchmark_problem_ids, trained_problem_ids=trained_ids,
            under=thresholds.coverage_underrepresented, over=thresholds.coverage_overrepresented,
        )
        difficulty_gaps = build_gaps(
            attribute="difficulty", pairs=inputs.preference_pairs, problems=inputs.problems,
            benchmark_problem_ids=inputs.benchmark_problem_ids, trained_problem_ids=trained_ids,
            under=thresholds.coverage_underrepresented, over=thresholds.coverage_overrepresented,
        )
        cov = preference_coverage(inputs.preference_pairs, inputs.problems, trained_ids)
        repo.write_json(run_id, "data_gaps/category_gaps.json", [g.to_dict() for g in category_gaps])
        repo.write_json(
            run_id, "data_gaps/difficulty_gaps.json", [g.to_dict() for g in difficulty_gaps]
        )
        repo.write_json(run_id, "data_gaps/preference_coverage.json", cov)
        repo.write_json(
            run_id, "data_gaps/strategy_gaps.json",
            strategy_gaps(inputs.preference_pairs, trained_ids),
        )
        repo.write_json(
            run_id, "data_gaps/error_coverage_correlation.json",
            correlate_errors_with_coverage(category_gaps, outcomes),
        )

        # ---- recommendations and the decision (sections 53-58, 89-95)
        eval_repo = ModelEvaluationRunRepository(config.paths.model_evaluations / "runs")
        decision = decide_iteration(
            config=analysis_config,
            benchmark_problem_count=len(inputs.benchmark_problem_ids),
            paired_ci_width=_paired_ci_width(inputs, eval_repo),
            outcomes=outcomes, category_gaps=category_gaps, diversity=diversity,
        )
        recommendations = build_recommendations(
            config=analysis_config, outcomes=outcomes, category_gaps=category_gaps,
            difficulty_gaps=difficulty_gaps, diversity=diversity,
            error_comparisons=[], training_curve=curve, coverage=cov,
        )
        repo.write_json(
            run_id, "recommendations/recommendations.json",
            [r.to_dict() for r in recommendations],
        )

        # ---- refinement (sections 59-79) -- leakage guard runs before any write
        benchmark_ids = inputs.benchmark_problem_ids
        benchmark_version = getattr(inputs.evaluation_manifest, "benchmark_version", None)
        repo.write_jsonl(
            run_id, "refined_dataset/hard_examples.jsonl",
            build_hard_examples(outcomes, evaluation_run_id=evaluation_run_id,
                                benchmark_version=benchmark_version),
        )
        repo.write_jsonl(
            run_id, "refined_dataset/regression_examples.jsonl",
            build_regression_examples(outcomes, evaluation_run_id=evaluation_run_id,
                                      benchmark_version=benchmark_version),
        )
        repo.write_jsonl(
            run_id, "refined_dataset/successful_dpo_examples.jsonl",
            build_successful_dpo_examples(outcomes, evaluation_run_id=evaluation_run_id,
                                          benchmark_version=benchmark_version),
        )
        plan = plan_refinement(
            inputs.preference_pairs,
            minimum_score_margin=analysis_config.refinement.minimum_score_margin,
            drop_duplicate_code=analysis_config.refinement.drop_duplicate_code,
            drop_infrastructure_errors=analysis_config.refinement.drop_infrastructure_errors,
            benchmark_problem_ids=benchmark_ids,
        )
        repo.write_json(run_id, "refined_dataset/refined_preference_plan.json", plan)
        refined = build_refined_preferences(
            inputs.preference_pairs, plan,
            parent_preference_run_id=inputs.lineage.preference_run_id,
            benchmark_problem_ids=benchmark_ids,
        )
        repo.write_jsonl(run_id, "refined_dataset/refined_preferences.jsonl", refined)

        # ---- reports (sections 96-103)
        summary = build_summary(
            analysis_run_id=run_id, lineage=inputs.lineage,
            benchmark_version=benchmark_version, outcomes=outcomes, profiles=profiles,
            diversity=diversity, category_gaps=category_gaps, difficulty_gaps=difficulty_gaps,
            decision=decision, recommendations=recommendations, training_curve=curve,
            coverage=cov,
        )
        repo.write_json(run_id, "summary.json", summary)
        repo.write_text(
            run_id, "reports/analysis.md", render_analysis_md(summary, test_failures=notable)
        )
        repo.write_yaml(
            run_id, "recommendations/next_experiment.yaml",
            render_next_experiment(summary, source_evaluation_run_id=evaluation_run_id),
        )
    except Exception as exc:  # noqa: BLE001 - record the failure, then re-raise
        repo.fail_run(run_id, error={"error_type": type(exc).__name__, "message": str(exc)})
        raise

    repo.complete_run(run_id)
    logger.info("Analysis run %s completed | decision=%s", run_id, decision["decision"])
    return run_id, summary


__all__ = ["run_analysis"]
