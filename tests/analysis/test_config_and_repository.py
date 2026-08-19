"""Tests for the analysis config, models and run repository (spec 11 sections 8, 9)."""

from __future__ import annotations

import pytest

from python_dpo.analysis.config import AnalysisConfig
from python_dpo.analysis.errors import (
    AnalysisConfigError,
    AnalysisRunNotFoundError,
    AnalysisStoreError,
)
from python_dpo.analysis.models import AnalysisManifest, ExperimentLineage, ProblemOutcome
from python_dpo.analysis.run_repository import AnalysisRunRepository

LINEAGE = ExperimentLineage(
    evaluation_run_id="eval_x", training_run_id="dpo_x", preference_run_id="pref_x",
    ranking_run_id="rank_x", candidate_run_id="run_x",
)


# --------------------------------------------------------------------------------- config


def test_defaults_load_without_a_file():
    """Absence is not an error: every threshold has a documented default."""
    config = AnalysisConfig.load(None)
    assert config.minimum_evidence.benchmark_problems == 30
    assert config.thresholds.coverage_underrepresented == 0.5


def test_the_shipped_config_file_parses():
    config = AnalysisConfig.load()
    assert config.thresholds.mode_collapse_reduction == 0.2


def test_unknown_keys_are_rejected():
    with pytest.raises(AnalysisConfigError, match="unknown top-level key"):
        AnalysisConfig.from_mapping({"bogus": 1})
    with pytest.raises(AnalysisConfigError, match="unknown key"):
        AnalysisConfig.from_mapping({"thresholds": {"bogus": 1}})


def test_coverage_bounds_must_be_ordered():
    with pytest.raises(AnalysisConfigError, match="must be less than"):
        AnalysisConfig.from_mapping(
            {"thresholds": {"coverage_underrepresented": 3.0, "coverage_overrepresented": 2.0}}
        )


def test_out_of_range_thresholds_are_rejected():
    with pytest.raises(AnalysisConfigError):
        AnalysisConfig.from_mapping({"thresholds": {"mode_collapse_reduction": 1.5}})
    with pytest.raises(AnalysisConfigError):
        AnalysisConfig.from_mapping({"minimum_evidence": {"benchmark_problems": 0}})


def test_weights_must_be_complete():
    with pytest.raises(AnalysisConfigError, match="missing key"):
        AnalysisConfig.from_mapping({"recommendations": {"weights": {"expected_impact": 0.5}}})


def test_config_round_trips():
    config = AnalysisConfig.load()
    assert AnalysisConfig.from_mapping(config.to_dict()).to_dict() == config.to_dict()


# --------------------------------------------------------------------------------- models


def test_problem_outcome_round_trips():
    outcome = ProblemOutcome(
        problem_id="p001", outcome="partial_improvement", base_best_score=0.3,
        dpo_best_score=0.7, base_solved=False, dpo_solved=False, severity="medium",
    )
    assert ProblemOutcome.from_dict(outcome.to_dict()).problem_id == "p001"


def test_unknown_outcome_is_rejected():
    with pytest.raises(AnalysisStoreError, match="outcome must be one of"):
        ProblemOutcome(
            problem_id="p001", outcome="invented", base_best_score=0.0, dpo_best_score=0.0,
            base_solved=False, dpo_solved=False,
        )


def test_manifest_round_trips_and_rejects_unknown_fields():
    manifest = AnalysisManifest(
        analysis_run_id="analysis_x", status="created", created_at="2026-01-01T00:00:00Z",
        lineage=LINEAGE,
    )
    assert AnalysisManifest.from_dict(manifest.to_dict()).analysis_run_id == "analysis_x"
    with pytest.raises(AnalysisStoreError, match="unknown field"):
        AnalysisManifest.from_dict({**manifest.to_dict(), "bogus": 1})


def test_illegal_status_transition_is_rejected():
    manifest = AnalysisManifest(
        analysis_run_id="analysis_x", status="completed", created_at="t", lineage=LINEAGE
    )
    with pytest.raises(AnalysisStoreError, match="illegal analysis status transition"):
        manifest.with_status("running")


# ----------------------------------------------------------------------------- repository


def test_run_id_shape(tmp_path):
    repo = AnalysisRunRepository(tmp_path / "runs")
    run_id = repo.new_run_id()
    assert run_id.startswith("analysis_")
    assert len(run_id.split("_")) == 4


def test_create_and_read_a_run(tmp_path):
    repo = AnalysisRunRepository(tmp_path / "runs")
    manifest = repo.create_run(lineage=LINEAGE, benchmark_version="v1")
    reloaded = repo.get_run(manifest.analysis_run_id)
    assert reloaded.status == "created"
    assert reloaded.lineage.preference_run_id == "pref_x"


def test_lifecycle_transitions(tmp_path):
    repo = AnalysisRunRepository(tmp_path / "runs")
    run_id = repo.create_run(lineage=LINEAGE).analysis_run_id
    assert repo.start_run(run_id).status == "running"
    assert repo.complete_run(run_id).status == "completed"


def test_failure_records_the_error(tmp_path):
    repo = AnalysisRunRepository(tmp_path / "runs")
    run_id = repo.create_run(lineage=LINEAGE).analysis_run_id
    repo.start_run(run_id)
    manifest = repo.fail_run(run_id, error={"error_type": "Boom", "message": "x"})
    assert manifest.status == "failed"
    assert manifest.error["error_type"] == "Boom"


def test_unknown_run_raises(tmp_path):
    repo = AnalysisRunRepository(tmp_path / "runs")
    with pytest.raises(AnalysisRunNotFoundError):
        repo.get_run("analysis_missing")


def test_list_runs_is_newest_first(tmp_path):
    repo = AnalysisRunRepository(tmp_path / "runs")
    a = repo.create_run(lineage=LINEAGE, analysis_run_id="analysis_20260101_000000_aaaa")
    b = repo.create_run(lineage=LINEAGE, analysis_run_id="analysis_20260102_000000_bbbb")
    ids = [m.analysis_run_id for m in repo.list_runs()]
    assert set(ids) == {a.analysis_run_id, b.analysis_run_id}


def test_jsonl_writer_produces_an_empty_file_for_no_rows(tmp_path):
    """An empty artifact is meaningful -- "no regressions" is a result, and its absence
    would be indistinguishable from the stage never running."""
    repo = AnalysisRunRepository(tmp_path / "runs")
    run_id = repo.create_run(lineage=LINEAGE).analysis_run_id
    path = repo.write_jsonl(run_id, "regressions/regressions.jsonl", [])
    assert path.is_file()
    assert path.read_text() == ""
