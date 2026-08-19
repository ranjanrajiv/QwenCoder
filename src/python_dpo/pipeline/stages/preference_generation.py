"""Stage 8 as a pipeline stage: DPO preference pair generation (spec 12 section 5, item 5).

``_finalize_preference_run`` and the core loop of ``_cmd_preferences_generate`` are moved
here from ``python_dpo.cli``. Pure computation over Stage 7's ranking artifacts -- no
model call of any kind, and never a byte of candidate code touched (spec section 8's
preference_generation stage; CLAUDE.md's Security rule is not implicated here at all).
"""

from __future__ import annotations

import logging
from typing import Any

from ...preferences import (
    BUILDER_VERSION,
    PREFERENCE_VERSION,
    PreferencePairBuilder,
    PreferencePolicyError,
    PreferenceRunRepository,
    PreferenceStatistics,
    ProblemSplitter,
    build_quality_report,
    derive_candidates_considered,
    make_policy,
)
from ...problems import DatasetError, dataset_path, load_problems
from ...ranking import RankingRunRepository
from ...runs import RunRepository
from ..errors import StageFailedError
from ..hashing import sha256_tree
from ._context import StageContext, StageResult

logger = logging.getLogger("python_dpo.pipeline.stages.preference_generation")


def _finalize_preference_run(
    preference_run_repo: PreferenceRunRepository,
    preference_run_id: str,
    assessments_by_problem: dict,
) -> PreferenceStatistics:
    manifest = preference_run_repo.get_run(preference_run_id)
    repository = preference_run_repo.results(preference_run_id)
    pairs = repository.load_pairs()
    rejections = repository.load_rejections()

    stats = PreferenceStatistics.from_records(
        manifest,
        pairs,
        rejections,
        candidates_considered=derive_candidates_considered(pairs, rejections),
    )
    preference_run_repo.write_statistics(stats)

    training_problem_ids = {p.problem_id for p in pairs if not p.duplicate_training_record}
    split_manifest = ProblemSplitter(
        ratios=manifest.split_ratios, seed=manifest.split_seed
    ).split(training_problem_ids)
    preference_run_repo.write_split_manifest(preference_run_id, split_manifest)
    repository.write_dataset(split_manifest)

    problem_correctness = {
        problem_id: [a.correctness for a in assessments]
        for problem_id, assessments in assessments_by_problem.items()
    }
    quality_report = build_quality_report(manifest, pairs, problem_correctness)
    preference_run_repo.write_quality_report(quality_report)

    return stats


def run(context: StageContext) -> StageResult:
    config = context.project_config
    settings = context.stage_config

    ranking_run_id = context.upstream_run_id("candidate_evaluation")

    ranking_run_repo = RankingRunRepository(config.paths.rankings / "runs")
    ranking_manifest = ranking_run_repo.get_run(ranking_run_id)
    assessments = ranking_run_repo.results(ranking_run_id).load_assessments()
    if not assessments:
        raise StageFailedError(f"ranking run {ranking_run_id!r} has no assessments to build preferences from")

    try:
        problems = {p.id: p for p in load_problems(dataset_path(config.paths.problems))}
    except DatasetError as exc:
        raise StageFailedError(f"could not load problem dataset: {exc}") from exc

    candidate_repo = RunRepository(config.paths.candidates / "runs").candidates(
        ranking_manifest.candidate_run_id
    )
    candidates_by_id = {c.candidate_id: c for c in candidate_repo.load_all()}

    assessments_by_problem: dict[str, list[Any]] = {}
    for assessment in assessments:
        assessments_by_problem.setdefault(assessment.problem_id, []).append(assessment)

    policy_name = settings.get("policy", config.preferences.policy)
    try:
        policy = make_policy(policy_name)
    except PreferencePolicyError as exc:
        raise StageFailedError(str(exc)) from exc

    minimum_score_margin = settings.get("minimum_score_margin", config.preferences.minimum_score_margin)
    max_pairs_per_problem = settings.get("max_pairs_per_problem", config.preferences.max_pairs_per_problem)
    split_seed = settings.get("split_seed", config.preferences.split.seed)
    split_ratios = settings.get("split_ratios") or config.preferences.split.as_ratios()

    preference_run_repo = PreferenceRunRepository(config.paths.preferences / "runs")
    manifest = preference_run_repo.create_run(
        ranking_run_id=ranking_run_id,
        evaluation_run_id=ranking_manifest.evaluation_run_id,
        candidate_run_id=ranking_manifest.candidate_run_id,
        preference_version=PREFERENCE_VERSION,
        selection_policy=policy.name,
        selection_policy_version=policy.version,
        minimum_score_margin=minimum_score_margin,
        split_ratios=split_ratios,
        split_seed=split_seed,
        builder_version=BUILDER_VERSION,
        max_pairs_per_problem=max_pairs_per_problem,
    )
    manifest = preference_run_repo.start_run(manifest.preference_run_id)
    logger.info(
        "Preference run %s created | ranking run %s | policy=%s margin=%s",
        manifest.preference_run_id,
        ranking_run_id,
        policy.name,
        minimum_score_margin,
    )

    repository = preference_run_repo.results(manifest.preference_run_id)
    builder = PreferencePairBuilder(
        policy, minimum_score_margin=minimum_score_margin, max_pairs_per_problem=max_pairs_per_problem
    )

    for problem_id in sorted(assessments_by_problem):
        problem = problems.get(problem_id)
        if problem is None:
            raise StageFailedError(
                f"problem {problem_id!r} (referenced by ranking run {ranking_run_id!r}) not "
                "found in the current problem dataset"
            )
        result = builder.build_problem(
            ranking_run_id=ranking_run_id,
            evaluation_run_id=manifest.evaluation_run_id,
            candidate_run_id=manifest.candidate_run_id,
            problem=problem,
            assessments=assessments_by_problem[problem_id],
            candidates_by_id=candidates_by_id,
        )
        repository.save_pairs(result.pairs)
        repository.save_rejections(result.rejections)

    stats = _finalize_preference_run(preference_run_repo, manifest.preference_run_id, assessments_by_problem)
    complete = set(assessments_by_problem) <= repository.paired_problem_ids()

    if complete:
        preference_run_repo.complete_run(manifest.preference_run_id)
    else:
        preference_run_repo.interrupt_run(manifest.preference_run_id)
        raise StageFailedError(f"preference run {manifest.preference_run_id} did not complete")

    logger.info(
        "Preference run %s completed | pairs=%d rejected=%d training_records=%d",
        manifest.preference_run_id,
        stats.pairs_generated,
        stats.pairs_rejected,
        stats.training_records,
    )

    return StageResult(
        stage_run_id=manifest.preference_run_id,
        output_artifacts={
            "preferences": sha256_tree(preference_run_repo.run_dir(manifest.preference_run_id))
        },
    )


__all__ = ["run"]
