"""Stage 10 as a pipeline stage: base-vs-DPO model evaluation (spec 12 section 5, item 7).

``_write_evaluation_report``, ``_build_model_evaluation_runner``,
``_reset_peak_gpu_memory``, ``_peak_gpu_memory_bytes`` and ``_group_records_by_problem``
are moved here from ``python_dpo.cli`` -- all five were already argparse-independent.
Generated Python is executed only through the Stage 5/6 sandbox path
(:class:`~python_dpo.evaluation.PytestRunner` over
:class:`~python_dpo.sandbox.SandboxExecutor`), reused unmodified, exactly as Stage 10 was
built (CLAUDE.md's Security rule).

Benchmark protection (plan decision, spec sections 56, 92): this stage requires an
existing benchmark manifest and never overwrites one. It builds a benchmark only when
``model_evaluation.build_benchmark_if_missing`` is explicitly set, and always re-checks
for training-split leakage before running.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...evaluation import CandidateEvaluator, PytestRunner, build_evaluation_sandbox_config
from ...model_evaluation import (
    Benchmark,
    BenchmarkError,
    BenchmarkLeakageError,
    ComparisonResult,
    EvaluationDriver,
    EvaluationExperimentConfig,
    EvaluationRecord,
    GenerationDriver,
    ModelDependencyError,
    ModelEvaluationConfigError,
    ModelEvaluationError,
    ModelEvaluationRunRepository,
    SuccessCriteria,
    bootstrap_ci,
    build_benchmark,
    build_comparison_report_json,
    build_failure_analysis,
    build_metrics_summary,
    check_leakage,
    classify_problem_examples,
    compare,
    evaluate_success_criteria,
    load_benchmark,
    mcnemar,
    mean_pass_at_k,
    paired_bootstrap,
    render_markdown_report,
    save_benchmark,
)
from ...model_evaluation.config import DEFAULT_CONFIG_PATH as EVAL_DEFAULT_CONFIG_PATH
from ...model_evaluation.run_repository import run_log_file as eval_run_log_file
from ...model_evaluation.runners import AdapterModelRunner, BaseModelRunner
from ...preferences import PreferenceRunNotFoundError, PreferenceRunRepository
from ...problems import DatasetError, dataset_path, load_problems
from ...sandbox import SandboxExecutor
from ...training.errors import TrainingRunNotFoundError
from ...training.hardware import check_hardware, format_hardware_report
from ...training.run_repository import TrainingRunRepository
from ...training.versions import capture_environment
from ..errors import StageFailedError
from ..hashing import sha256_tree
from ._context import StageContext, StageResult

logger = logging.getLogger("python_dpo.pipeline.stages.model_evaluation")


def _benchmarks_root(config: Any) -> Path:
    return config.project_root / "benchmarks"


def _build_model_evaluation_runner(
    variant: str, training_manifest: Any, training_run_repo: TrainingRunRepository,
    training_run_id: str, experiment: EvaluationExperimentConfig,
):
    if variant == "base":
        return BaseModelRunner(
            model_name=training_manifest.model_name,
            model_revision=training_manifest.model_revision,
            quantization=experiment.quantization,
            generation=experiment.generation,
        )
    return AdapterModelRunner(
        model_name=training_manifest.model_name,
        model_revision=training_manifest.model_revision,
        adapter_dir=training_run_repo.adapter_dir(training_run_id),
        quantization=experiment.quantization,
        generation=experiment.generation,
    )


def _reset_peak_gpu_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_memory_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _group_records_by_problem(records: list[EvaluationRecord]) -> dict[str, list[EvaluationRecord]]:
    by_problem: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        by_problem.setdefault(record.problem_id, []).append(record)
    return by_problem


def _write_evaluation_report(
    eval_run_repo: ModelEvaluationRunRepository,
    evaluation_run_id: str,
    *,
    problems_dir: Path,
    peak_gpu_memory_bytes: dict[str, int] | None = None,
) -> ComparisonResult | None:
    manifest = eval_run_repo.get_run(evaluation_run_id)
    generation_records = {
        variant: eval_run_repo.load_generation_records(evaluation_run_id, variant)
        for variant in manifest.models_requested
    }
    evaluation_records = {
        variant: eval_run_repo.load_evaluation_records(evaluation_run_id, variant)
        for variant in manifest.models_requested
    }

    resolved_config = eval_run_repo.read_config(evaluation_run_id) or {}
    statistics_config = resolved_config.get("statistics", manifest.statistics_config)
    success_criteria_config = resolved_config.get("success_criteria", {})
    try:
        criteria = SuccessCriteria.from_mapping(success_criteria_config)
    except ModelEvaluationConfigError:
        criteria = SuccessCriteria()

    pass_at_k_values = tuple(int(k) for k in statistics_config.get("pass_at_k", [1, 5, 10]))
    bootstrap_iterations = statistics_config.get("bootstrap_iterations", 1000)
    bootstrap_seed = statistics_config.get("bootstrap_seed", 42)
    confidence_level = statistics_config.get("confidence_level", 0.95)

    summary = build_metrics_summary(evaluation_run_id, generation_records, evaluation_records, pass_at_k_values)
    eval_run_repo.write_metrics(evaluation_run_id, "summary", summary.to_dict())

    pass_at_k_cis: dict[str, dict[str, Any]] = {}
    for variant, records in evaluation_records.items():
        by_problem = _group_records_by_problem(records)
        per_problem_valid = {
            pid: [r for r in recs if not r.is_infrastructure_error] for pid, recs in by_problem.items()
        }
        variant_cis: dict[str, Any] = {}
        for k in pass_at_k_values:
            eligible = [
                (len(valid), sum(1 for r in valid if r.correct))
                for valid in per_problem_valid.values()
                if len(valid) >= k
            ]
            if not eligible:
                continue
            variant_cis[str(k)] = bootstrap_ci(
                eligible, lambda data, k=k: mean_pass_at_k(list(data), k),
                iterations=bootstrap_iterations, seed=bootstrap_seed, confidence=confidence_level,
            )
        pass_at_k_cis[variant] = variant_cis
    eval_run_repo.write_metrics(
        evaluation_run_id, "pass_at_k",
        {variant: {k: ci.to_dict() for k, ci in by_k.items()} for variant, by_k in pass_at_k_cis.items()},
    )

    comparison: ComparisonResult | None = None
    paired_ci = None
    mcnemar_result = None
    if "base" in evaluation_records and "dpo" in evaluation_records:
        base_by_problem = _group_records_by_problem(evaluation_records["base"])
        dpo_by_problem = _group_records_by_problem(evaluation_records["dpo"])
        all_problem_ids = sorted(set(base_by_problem) | set(dpo_by_problem))
        comparison = compare(all_problem_ids, base_by_problem, dpo_by_problem, allow_incomplete=True)

        common = sorted(set(base_by_problem) & set(dpo_by_problem))
        base_series, dpo_series = [], []
        base_solved: list[bool] = []
        dpo_solved: list[bool] = []
        for pid in common:
            base_valid = [r for r in base_by_problem[pid] if not r.is_infrastructure_error]
            dpo_valid = [r for r in dpo_by_problem[pid] if not r.is_infrastructure_error]
            if not base_valid or not dpo_valid:
                continue
            base_series.append((len(base_valid), sum(1 for r in base_valid if r.correct)))
            dpo_series.append((len(dpo_valid), sum(1 for r in dpo_valid if r.correct)))
            base_solved.append(any(r.correct for r in base_valid))
            dpo_solved.append(any(r.correct for r in dpo_valid))

        if base_series:
            paired_ci = paired_bootstrap(
                base_series, dpo_series, lambda data: mean_pass_at_k(list(data), 1),
                iterations=bootstrap_iterations, seed=bootstrap_seed, confidence=confidence_level,
            )
            eval_run_repo.write_metrics(evaluation_run_id, "bootstrap", {"paired_pass_at_1": paired_ci.to_dict()})
        if base_solved:
            mcnemar_result = mcnemar(base_solved, dpo_solved)

    failure_analysis = build_failure_analysis(generation_records, evaluation_records)
    eval_run_repo.write_report_json(evaluation_run_id, "failure_analysis", failure_analysis)

    success = evaluate_success_criteria(summary, paired_ci, criteria)

    if comparison is not None:
        try:
            problems = load_problems(dataset_path(problems_dir))
            prompts_by_problem = {p.id: p.prompt for p in problems}
        except DatasetError:
            prompts_by_problem = {}
        improvements, regressions, ties = classify_problem_examples(
            comparison, generation_records, evaluation_records, prompts_by_problem
        )
        eval_run_repo.write_report_jsonl(evaluation_run_id, "improvements", improvements)
        eval_run_repo.write_report_jsonl(evaluation_run_id, "regressions", regressions)
        eval_run_repo.write_report_jsonl(evaluation_run_id, "ties", ties)

        report_json = build_comparison_report_json(
            evaluation_run_id=evaluation_run_id, benchmark_version=manifest.benchmark_version,
            summary=summary, comparison=comparison, pass_at_k_cis=pass_at_k_cis,
            paired_pass_at_1_ci=paired_ci, mcnemar_result=mcnemar_result, success=success,
            failure_analysis=failure_analysis,
        )
        eval_run_repo.write_report_json(evaluation_run_id, "base_vs_dpo", report_json)

        markdown = render_markdown_report(
            evaluation_run_id=evaluation_run_id, benchmark_version=manifest.benchmark_version,
            benchmark_problem_count=comparison.benchmark_problems, base_model_name=manifest.base_model_name,
            base_model_revision=manifest.base_model_revision, adapter_path=manifest.adapter_path,
            training_run_id=manifest.training_run_id, generation_config=manifest.generation_config,
            summary=summary, comparison=comparison, pass_at_k_cis=pass_at_k_cis,
            paired_pass_at_1_ci=paired_ci, mcnemar_result=mcnemar_result, success=success,
            failure_analysis=failure_analysis, generation_records=generation_records,
            peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        )
        eval_run_repo.write_report_text(evaluation_run_id, "base_vs_dpo", markdown)

    return comparison


def _resolve_benchmark(config: Any, settings: Any, benchmark_name: str, problems: Any) -> Any:
    """Section 56/92: never overwrite an existing benchmark; build one only when the
    experiment explicitly opts in, and only when none exists yet."""
    root = _benchmarks_root(config)
    try:
        return load_benchmark(root, benchmark_name, problems)
    except BenchmarkError as exc:
        if not settings.get("build_benchmark_if_missing", False):
            raise StageFailedError(
                f"benchmark {benchmark_name!r} does not exist at {root} and "
                f"model_evaluation.build_benchmark_if_missing is not set: {exc}"
            ) from exc
        manifest = build_benchmark(benchmark_name, problems, [p.id for p in problems])
        save_benchmark(root, manifest)
        return load_benchmark(root, benchmark_name, problems)


def run(context: StageContext) -> StageResult:
    config = context.project_config
    settings = context.stage_config

    training_run_id = context.upstream_run_id("dpo_training")

    config_path = Path(settings.get("config")) if settings.get("config") else EVAL_DEFAULT_CONFIG_PATH
    try:
        experiment = EvaluationExperimentConfig.load(config_path)
    except ModelEvaluationConfigError as exc:
        raise StageFailedError(str(exc)) from exc

    num_samples = settings.get("num_samples")
    benchmark_name = settings.get("benchmark")
    if num_samples is not None or benchmark_name is not None:
        # Applied as one reconstruction, not via with_overrides(), because
        # EvaluationExperimentConfig validates internally that every configured pass_at_k
        # is estimable from num_samples -- with_overrides(num_samples=...) alone would
        # apply the new sample count and immediately re-validate against the *old*
        # pass_at_k list, before this code ever got a chance to cap it (exactly the
        # --smoke-test CLI flag's ordering, applied here in one step instead of two).
        data = experiment.to_dict()
        if num_samples is not None:
            data["generation"]["num_samples"] = num_samples
            data["statistics"]["pass_at_k"] = [
                k for k in data["statistics"]["pass_at_k"] if k <= num_samples
            ] or [1]
        if benchmark_name is not None:
            data["benchmark"]["name"] = benchmark_name
        try:
            experiment = EvaluationExperimentConfig.from_mapping(data)
        except ModelEvaluationConfigError as exc:
            raise StageFailedError(str(exc)) from exc

    try:
        problems = load_problems(dataset_path(config.paths.problems))
    except DatasetError as exc:
        raise StageFailedError(f"could not load problem dataset: {exc}") from exc
    problems_by_id = {p.id: p for p in problems}

    benchmark = _resolve_benchmark(config, settings, experiment.benchmark.name, problems)

    training_run_repo = TrainingRunRepository(config.paths.training / "runs")
    try:
        training_manifest = training_run_repo.get_run(training_run_id)
    except TrainingRunNotFoundError as exc:
        raise StageFailedError(str(exc)) from exc
    adapter_dir = training_run_repo.adapter_dir(training_run_id)

    # Section 56/92: the training run already records which preference run it trained
    # on, so leakage is checked against *that* run automatically -- no separate setting
    # needed, and no way to silently skip the check by omitting one.
    try:
        split_manifest = PreferenceRunRepository(config.paths.preferences / "runs").read_split_manifest(
            training_manifest.preference_run_id
        )
    except PreferenceRunNotFoundError as exc:
        raise StageFailedError(str(exc)) from exc
    if split_manifest is not None:
        try:
            check_leakage(benchmark, split_manifest.to_dict())
        except BenchmarkLeakageError as exc:
            raise StageFailedError(str(exc)) from exc

    ordered_problems = sorted(benchmark.problems, key=lambda p: p.id)
    limit = settings.get("limit")
    if limit is not None:
        if limit < 1:
            raise StageFailedError("model_evaluation.limit must be at least 1")
        ordered_problems = ordered_problems[:limit]
    if not ordered_problems:
        raise StageFailedError(f"no problems selected from benchmark {benchmark.benchmark_version!r}")
    working_benchmark = Benchmark(manifest=benchmark.manifest, problems=tuple(ordered_problems))

    model = settings.get("model", "both")
    models_requested: tuple[str, ...] = ("base", "dpo") if model == "both" else (model,)

    hardware_report = check_hardware(require_quantization=experiment.quantization.enabled)
    if not hardware_report.passed:
        raise StageFailedError(format_hardware_report(hardware_report))
    environment = capture_environment()

    eval_sandbox_config = build_evaluation_sandbox_config(config.sandbox, config.evaluation)

    eval_run_repo = ModelEvaluationRunRepository(config.paths.model_evaluations / "runs")
    manifest = eval_run_repo.create_run(
        benchmark_version=working_benchmark.benchmark_version,
        benchmark_hash=working_benchmark.manifest.dataset_hash,
        base_model_name=training_manifest.model_name,
        base_model_revision=training_manifest.model_revision,
        adapter_path=str(adapter_dir),
        training_run_id=training_run_id,
        models_requested=models_requested,
        generation_config=experiment.generation.to_dict(),
        quantization=experiment.quantization.to_dict(),
        statistics_config=experiment.statistics.to_dict(),
        seeds={"base_seed": experiment.generation.base_seed},
        hardware=hardware_report.info.to_dict(),
        environment=environment,
    )
    run_id = manifest.evaluation_run_id
    eval_run_repo.write_config(run_id, experiment.to_dict())
    eval_run_repo.write_benchmark_manifest(run_id, working_benchmark.manifest.to_dict())
    eval_run_repo.start_run(run_id)
    logger.info(
        "Evaluation run %s created | benchmark=%s (%d problem(s)) | models=%s",
        run_id, working_benchmark.benchmark_version, len(working_benchmark), ",".join(models_requested),
    )

    peak_gpu_memory_bytes: dict[str, int] = {}
    try:
        with eval_run_log_file(eval_run_repo.log_path(run_id)):
            for variant in models_requested:
                logger.info("Generating with the %s model...", variant)
                runner = _build_model_evaluation_runner(
                    variant, training_manifest, training_run_repo, training_run_id, experiment
                )
                _reset_peak_gpu_memory()
                runner.ensure_loaded()

                gen_driver = GenerationDriver(run_id)
                variant_generations = gen_driver.run(
                    runner, working_benchmark, experiment.generation,
                    on_record=lambda record, v=variant: eval_run_repo.append_generation_record(run_id, v, record),
                )
                memory_bytes = _peak_gpu_memory_bytes()
                if memory_bytes is not None:
                    peak_gpu_memory_bytes[variant] = memory_bytes
                runner.unload()

                logger.info("Evaluating %s candidates through the sandbox...", variant)
                repository = eval_run_repo.sandbox_repository(run_id, variant)
                evaluator = CandidateEvaluator(
                    runner=PytestRunner(SandboxExecutor(config=eval_sandbox_config)), repository=repository
                )
                eval_driver = EvaluationDriver(
                    evaluator=evaluator, repository=repository, evaluation_run_id=run_id,
                    model_variant=variant, model_name=training_manifest.model_name,
                    model_revision=training_manifest.model_revision,
                    generation_config=experiment.generation.to_dict(),
                )
                eval_driver.run(
                    variant_generations, problems_by_id,
                    on_record=lambda record, v=variant: eval_run_repo.append_evaluation_record(run_id, v, record),
                )
    except (ModelEvaluationError, ModelDependencyError) as exc:
        eval_run_repo.fail_run(run_id, error_type=type(exc).__name__, error_message=str(exc))
        raise StageFailedError(f"model evaluation run {run_id} failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - record, then re-report
        eval_run_repo.fail_run(run_id, error_type=type(exc).__name__, error_message=str(exc))
        raise StageFailedError(f"model evaluation run {run_id} failed ({type(exc).__name__}): {exc}") from exc

    eval_run_repo.complete_run(run_id)
    if peak_gpu_memory_bytes:
        eval_run_repo.write_metrics(run_id, "peak_gpu_memory", peak_gpu_memory_bytes)
    _write_evaluation_report(
        eval_run_repo, run_id, problems_dir=config.paths.problems, peak_gpu_memory_bytes=peak_gpu_memory_bytes
    )

    return StageResult(
        stage_run_id=run_id,
        output_artifacts={"model_evaluation": sha256_tree(eval_run_repo.run_dir(run_id))},
    )


__all__ = ["run"]
