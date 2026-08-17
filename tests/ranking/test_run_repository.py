"""Tests for RankingRunRepository (spec 07 sections 26, 27, 42, 54, 55)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from python_dpo.ranking import (
    CandidateAssessment,
    RankingRepository,
    RankingRunError,
    RankingRunNotFoundError,
    RankingRunRepository,
    RankingStatistics,
)


def make_manifest_kwargs(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "evaluation_run_id": "eval_20260817_115154_dcd4",
        "candidate_run_id": "20260817_055411",
        "ranking_version": "v1",
        "scoring_version": "v1",
        "comparator_version": "v1",
        "requested_problem_ids": ["p001"],
    }
    fields.update(overrides)
    return fields


def test_run_id_format(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    moment = datetime(2026, 8, 17, 13, 37, 0, tzinfo=timezone.utc)
    run_id = repo.new_run_id(moment)
    assert run_id.startswith("rank_20260817_133700_")
    suffix = run_id.removeprefix("rank_20260817_133700_")
    assert len(suffix) == 4
    int(suffix, 16)  # hex


def test_create_get_and_list_runs(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())

    assert manifest.status == "created"
    assert repo.get_run(manifest.ranking_run_id) == manifest
    assert [m.ranking_run_id for m in repo.list_runs()] == [manifest.ranking_run_id]


def test_get_run_raises_for_unknown_id(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    with pytest.raises(RankingRunNotFoundError):
        repo.get_run("rank_does_not_exist")


def test_list_runs_newest_first(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    first = repo.create_run(**make_manifest_kwargs(ranking_run_id="rank_20260817_100000_0001"))
    second = repo.create_run(**make_manifest_kwargs(ranking_run_id="rank_20260817_110000_0002"))
    assert [m.ranking_run_id for m in repo.list_runs()] == [
        second.ranking_run_id,
        first.ranking_run_id,
    ]


def test_latest_run_for_evaluation_run(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    repo.create_run(
        **make_manifest_kwargs(evaluation_run_id="eval_A", ranking_run_id="rank_20260817_100000_0001")
    )
    latest = repo.create_run(
        **make_manifest_kwargs(evaluation_run_id="eval_A", ranking_run_id="rank_20260817_110000_0002")
    )
    repo.create_run(
        **make_manifest_kwargs(evaluation_run_id="eval_B", ranking_run_id="rank_20260817_120000_0003")
    )
    found = repo.latest_run_for_evaluation_run("eval_A")
    assert found.ranking_run_id == latest.ranking_run_id


def test_latest_run_for_evaluation_run_is_none_when_never_ranked(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    assert repo.latest_run_for_evaluation_run("eval_never_ranked") is None


def test_status_lifecycle_created_running_completed(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())

    running = repo.start_run(manifest.ranking_run_id)
    assert running.status == "running"
    assert running.started_at is not None

    completed = repo.complete_run(manifest.ranking_run_id)
    assert completed.status == "completed"
    assert completed.completed_at is not None


def test_interrupted_run_can_be_resumed(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())
    repo.start_run(manifest.ranking_run_id)
    repo.interrupt_run(manifest.ranking_run_id)

    resumed = repo.resume_run(manifest.ranking_run_id)
    assert resumed.status == "running"


def test_resume_refuses_a_completed_run(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())
    repo.start_run(manifest.ranking_run_id)
    repo.complete_run(manifest.ranking_run_id)

    with pytest.raises(RankingRunError, match="already completed"):
        repo.resume_run(manifest.ranking_run_id)


def test_fail_run_records_the_error(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())
    repo.start_run(manifest.ranking_run_id)

    failed = repo.fail_run(manifest.ranking_run_id, error_type="ranking_error", error_message="boom")
    assert failed.status == "failed"
    assert failed.error["error_type"] == "ranking_error"


def test_cancel_run(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())
    cancelled = repo.cancel_run(manifest.ranking_run_id)
    assert cancelled.status == "cancelled"


def test_write_and_read_statistics(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())

    stats = RankingStatistics.from_records(manifest, [], [])
    repo.write_statistics(stats)

    assert repo.read_statistics(manifest.ranking_run_id) == stats
    assert repo.read_statistics("rank_missing") is None


def test_results_returns_a_repository_scoped_to_the_run_dir(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())

    results_repo = repo.results(manifest.ranking_run_id)
    assert isinstance(results_repo, RankingRepository)
    assert results_repo.directory == repo.run_dir(manifest.ranking_run_id)


def test_an_assessment_saved_via_results_is_visible_after_reload(tmp_path):
    repo = RankingRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())

    assessment = CandidateAssessment(
        ranking_run_id=manifest.ranking_run_id,
        evaluation_run_id=manifest.evaluation_run_id,
        candidate_run_id=manifest.candidate_run_id,
        candidate_id="p001_c001",
        problem_id="p001",
        correctness="correct",
        all_tests_passed=True,
        pass_rate=1.0,
        score=1.0,
        tests_total=7,
        tests_passed=7,
        tests_failed=0,
        tests_error=0,
        tests_skipped=0,
        timeout=False,
        infrastructure_error=False,
    )
    repo.results(manifest.ranking_run_id).save_assessment(assessment)

    reloaded = RankingRunRepository(Path(tmp_path)).results(manifest.ranking_run_id)
    assert reloaded.load_assessments() == [assessment]
