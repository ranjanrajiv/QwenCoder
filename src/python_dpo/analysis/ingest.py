"""Loading the Stage 10 run and resolving its lineage (spec 11 sections 6, 7).

Section 7 makes the experiment lineage **mandatory**, not an enrichment: an analysis that
does not know which preference dataset trained the adapter it is analysing cannot make a
coverage claim about it. So a broken hop raises :class:`LineageError` rather than
degrading to a partial analysis.

One collision is worth naming, because it is easy to get backwards: the *training*
manifest's ``evaluation_run_id`` is a **Stage 6** candidate-evaluation run id, not the
Stage 10 model-evaluation run this stage starts from. They share a field name and an
``eval_`` prefix but are different runs in different stores. The Stage 10 id is carried as
``evaluation_run_id`` on the lineage; the Stage 6 one is kept as
``sandbox_evaluation_run_id`` so nothing downstream has to re-derive which is which.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..evaluation.models import TestCaseResult
from ..model_evaluation.errors import EvaluationRunNotFoundError
from ..model_evaluation.run_repository import ModelEvaluationRunRepository
from ..preferences import PreferenceRunRepository
from ..preferences.errors import PreferenceRunNotFoundError
from ..problems import DatasetError, dataset_path, load_problems
from ..problems.models import Problem
from ..training.errors import TrainingRunNotFoundError
from ..training.run_repository import TrainingRunRepository
from .errors import AnalysisInputError, LineageError
from .models import ExperimentLineage

logger = logging.getLogger("python_dpo.analysis.ingest")


@dataclass(frozen=True)
class AnalysisInputs:
    """Everything the analysis modules read, gathered once (spec section 6)."""

    lineage: ExperimentLineage
    evaluation_manifest: Any
    benchmark_manifest: dict[str, Any] | None
    generations: dict[str, list[Any]]
    evaluations: dict[str, list[Any]]
    test_results: dict[str, list[TestCaseResult]]
    problems: dict[str, Problem]
    training_manifest: Any | None = None
    preference_pairs: list[Any] = field(default_factory=list)
    quality_report: Any | None = None
    split_manifest: Any | None = None

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(sorted(self.generations))

    @property
    def benchmark_problem_ids(self) -> tuple[str, ...]:
        if not self.benchmark_manifest:
            return ()
        return tuple(self.benchmark_manifest.get("problem_ids") or ())


def resolve_lineage(
    evaluation_run_id: str,
    eval_repo: ModelEvaluationRunRepository,
    training_repo: TrainingRunRepository,
    *,
    preference_run_id: str | None = None,
    training_run_id: str | None = None,
) -> ExperimentLineage:
    """Walk the manifest chain from a Stage 10 run back to the candidate run.

    ``preference_run_id`` / ``training_run_id`` override the resolved values. A mismatch
    between an explicit override and what the manifests actually say is an error, not a
    preference -- silently analysing a different dataset than the one that trained the
    adapter is exactly the failure this stage exists to prevent.
    """
    try:
        eval_manifest = eval_repo.get_run(evaluation_run_id)
    except EvaluationRunNotFoundError as exc:
        raise LineageError(f"no Stage 10 evaluation run {evaluation_run_id!r}: {exc}") from exc

    resolved_training = eval_manifest.training_run_id
    if not resolved_training:
        raise LineageError(
            f"evaluation run {evaluation_run_id!r} records no training_run_id; the lineage "
            "cannot be resolved (spec section 7)"
        )
    if training_run_id and training_run_id != resolved_training:
        raise LineageError(
            f"--training-run-id {training_run_id!r} does not match the training run recorded "
            f"on evaluation run {evaluation_run_id!r} ({resolved_training!r})"
        )

    try:
        training_manifest = training_repo.get_run(resolved_training)
    except TrainingRunNotFoundError as exc:
        raise LineageError(
            f"training run {resolved_training!r} named by evaluation run "
            f"{evaluation_run_id!r} is missing: {exc}"
        ) from exc

    resolved_preference = training_manifest.preference_run_id
    if preference_run_id and preference_run_id != resolved_preference:
        raise LineageError(
            f"--preference-run-id {preference_run_id!r} does not match the preference run "
            f"recorded on training run {resolved_training!r} ({resolved_preference!r})"
        )

    for name, value in (
        ("preference_run_id", resolved_preference),
        ("ranking_run_id", training_manifest.ranking_run_id),
        ("candidate_run_id", training_manifest.candidate_run_id),
    ):
        if not value:
            raise LineageError(
                f"training run {resolved_training!r} records no {name}; the lineage chain "
                "is incomplete (spec section 7)"
            )

    return ExperimentLineage(
        evaluation_run_id=evaluation_run_id,
        training_run_id=resolved_training,
        preference_run_id=resolved_preference,
        ranking_run_id=training_manifest.ranking_run_id,
        candidate_run_id=training_manifest.candidate_run_id,
        # Deliberately NOT evaluation_run_id: this is the Stage 6 sandbox run, a different
        # store entirely. See the module docstring.
        sandbox_evaluation_run_id=training_manifest.evaluation_run_id,
    )


def load_analysis_inputs(
    config: Any,
    evaluation_run_id: str,
    *,
    preference_run_id: str | None = None,
    training_run_id: str | None = None,
) -> AnalysisInputs:
    """Gather every artifact the analysis reads (spec sections 6, 7)."""
    eval_repo = ModelEvaluationRunRepository(config.paths.model_evaluations / "runs")
    training_repo = TrainingRunRepository(config.paths.training / "runs")

    lineage = resolve_lineage(
        evaluation_run_id, eval_repo, training_repo,
        preference_run_id=preference_run_id, training_run_id=training_run_id,
    )
    logger.info(
        "Lineage resolved: %s -> %s -> %s -> %s -> %s",
        lineage.evaluation_run_id, lineage.training_run_id, lineage.preference_run_id,
        lineage.ranking_run_id, lineage.candidate_run_id,
    )

    eval_manifest = eval_repo.get_run(evaluation_run_id)
    variants = tuple(eval_manifest.models_requested)

    generations = {
        v: eval_repo.load_generation_records(evaluation_run_id, v) for v in variants
    }
    evaluations = {
        v: eval_repo.load_evaluation_records(evaluation_run_id, v) for v in variants
    }
    # Section 45-49's per-test forensics are only reachable through the Stage 6 repository
    # nested inside the Stage 10 run; the variant is knowable only from which directory the
    # file came from, which is why this is keyed by variant here.
    test_results: dict[str, list[TestCaseResult]] = {}
    for variant in variants:
        try:
            test_results[variant] = eval_repo.sandbox_repository(
                evaluation_run_id, variant
            ).load_test_results()
        except Exception as exc:  # noqa: BLE001 - absent forensics degrade, never crash
            logger.warning("No per-test results for variant %s: %s", variant, exc)
            test_results[variant] = []

    try:
        problems = {p.id: p for p in load_problems(dataset_path(config.paths.problems))}
    except DatasetError as exc:
        raise AnalysisInputError(f"could not load the problem dataset: {exc}") from exc

    training_manifest = training_repo.get_run(lineage.training_run_id)

    pref_repo = PreferenceRunRepository(config.paths.preferences / "runs")
    preference_pairs: list[Any] = []
    quality_report = None
    split_manifest = None
    try:
        preference_pairs = pref_repo.results(lineage.preference_run_id).load_pairs()
    except (PreferenceRunNotFoundError, AttributeError, OSError) as exc:
        logger.warning("Preference pairs unavailable for %s: %s", lineage.preference_run_id, exc)
    try:
        quality_report = pref_repo.read_quality_report(lineage.preference_run_id)
    except Exception as exc:  # noqa: BLE001 - optional enrichment
        logger.warning("Quality report unavailable: %s", exc)
    try:
        split_manifest = pref_repo.read_split_manifest(lineage.preference_run_id)
    except Exception as exc:  # noqa: BLE001 - optional enrichment
        logger.warning("Split manifest unavailable: %s", exc)

    return AnalysisInputs(
        lineage=lineage,
        evaluation_manifest=eval_manifest,
        benchmark_manifest=eval_repo.read_benchmark_manifest(evaluation_run_id),
        generations=generations,
        evaluations=evaluations,
        test_results=test_results,
        problems=problems,
        training_manifest=training_manifest,
        preference_pairs=preference_pairs,
        quality_report=quality_report,
        split_manifest=split_manifest,
    )


__all__ = ["AnalysisInputs", "load_analysis_inputs", "resolve_lineage"]
