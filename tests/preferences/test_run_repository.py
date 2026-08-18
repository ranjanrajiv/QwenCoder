"""Tests for PreferenceRunRepository (spec 08 sections 50, 83, 84)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from python_dpo.preferences.errors import PreferenceRunNotFoundError
from python_dpo.preferences.models import PreferenceModelError
from python_dpo.preferences.run_repository import PreferenceRunError, PreferenceRunRepository

RANKING_RUN_ID = "rank_20260817_161726_a84d"
EVAL_RUN_ID = "eval_20260817_115154_dcd4"
CANDIDATE_RUN_ID = "run_20260817_055411"


def make_manifest_kwargs(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "ranking_run_id": RANKING_RUN_ID,
        "evaluation_run_id": EVAL_RUN_ID,
        "candidate_run_id": CANDIDATE_RUN_ID,
        "preference_version": "v1",
        "selection_policy": "strict",
        "selection_policy_version": "strict_v1",
        "minimum_score_margin": 0.2,
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "split_seed": 42,
        "builder_version": "v1",
    }
    fields.update(overrides)
    return fields


def test_new_run_id_format(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    run_id = repo.new_run_id(now=datetime(2026, 8, 18, 3, 8, 5, tzinfo=timezone.utc))
    assert run_id.startswith("pref_20260818_030805_")
    assert len(run_id) == len("pref_20260818_030805_") + 4


def test_new_run_id_retries_on_collision(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    now = datetime(2026, 8, 18, 3, 8, 5, tzinfo=timezone.utc)
    first = repo.create_run(**make_manifest_kwargs(), preference_run_id=None)
    second_id = repo.new_run_id(now=now)
    assert second_id != first.preference_run_id


def test_create_and_get_run(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    manifest = repo.create_run(**make_manifest_kwargs())
    fetched = repo.get_run(manifest.preference_run_id)
    assert fetched == manifest
    assert fetched.status == "created"


def test_get_run_missing_raises(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    with pytest.raises(PreferenceRunNotFoundError):
        repo.get_run("pref_nonexistent")


def test_list_runs_is_newest_first(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    a = repo.create_run(**make_manifest_kwargs(preference_run_id="pref_20260818_030800_0001"))
    b = repo.create_run(**make_manifest_kwargs(preference_run_id="pref_20260818_030900_0002"))
    runs = repo.list_runs()
    assert [r.preference_run_id for r in runs] == [b.preference_run_id, a.preference_run_id]


def test_status_lifecycle(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    manifest = repo.create_run(**make_manifest_kwargs())
    running = repo.start_run(manifest.preference_run_id)
    assert running.status == "running"
    completed = repo.complete_run(manifest.preference_run_id)
    assert completed.status == "completed"
    with pytest.raises(PreferenceRunError):
        repo.start_run(manifest.preference_run_id)


def test_resume_refuses_a_completed_run(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    manifest = repo.create_run(**make_manifest_kwargs())
    repo.start_run(manifest.preference_run_id)
    repo.complete_run(manifest.preference_run_id)
    with pytest.raises(PreferenceRunError):
        repo.resume_run(manifest.preference_run_id)


def test_resume_reopens_an_interrupted_run(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    manifest = repo.create_run(**make_manifest_kwargs())
    repo.start_run(manifest.preference_run_id)
    repo.interrupt_run(manifest.preference_run_id)
    resumed = repo.resume_run(manifest.preference_run_id)
    assert resumed.status == "running"


def test_fail_run_records_error(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    manifest = repo.create_run(**make_manifest_kwargs())
    repo.start_run(manifest.preference_run_id)
    failed = repo.fail_run(
        manifest.preference_run_id, error_type="builder_crash", error_message="boom"
    )
    assert failed.status == "failed"
    assert failed.error["error_type"] == "builder_crash"


def test_force_creates_a_second_run_leaving_the_first_untouched(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    first = repo.create_run(**make_manifest_kwargs())
    repo.start_run(first.preference_run_id)
    repo.complete_run(first.preference_run_id)
    before = repo.get_run(first.preference_run_id)

    second = repo.create_run(**make_manifest_kwargs())
    assert second.preference_run_id != first.preference_run_id

    after = repo.get_run(first.preference_run_id)
    assert after == before  # byte-identical, spec section 84


def test_results_returns_a_run_scoped_repository(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    manifest = repo.create_run(**make_manifest_kwargs())
    results = repo.results(manifest.preference_run_id)
    assert results.directory == repo.run_dir(manifest.preference_run_id)


def test_latest_run_for_ranking_run(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    a = repo.create_run(**make_manifest_kwargs(preference_run_id="pref_20260818_030800_0001"))
    b = repo.create_run(**make_manifest_kwargs(preference_run_id="pref_20260818_030900_0002"))
    latest = repo.latest_run_for_ranking_run(RANKING_RUN_ID)
    assert latest.preference_run_id == b.preference_run_id


def test_read_statistics_returns_none_when_absent(tmp_path):
    repo = PreferenceRunRepository(tmp_path)
    manifest = repo.create_run(**make_manifest_kwargs())
    assert repo.read_statistics(manifest.preference_run_id) is None
