"""Tests for artifact lineage (spec 12 section 27)."""

from __future__ import annotations

from python_dpo.pipeline.lineage import build_lineage
from python_dpo.pipeline.manifest import StageManifest


def make_manifest(stage_name: str, stage_run_id: str) -> StageManifest:
    return StageManifest(
        stage_name=stage_name, stage_run_id=stage_run_id, status="COMPLETED", code_version="0.12.0"
    )


def test_build_lineage_reproduces_the_full_chain():
    upstream = {
        "problem_dataset": make_manifest("problem_dataset", "exp_x_problem_dataset"),
        "candidate_generation": make_manifest("candidate_generation", "run_20260819_000000_aaaa"),
        "candidate_execution": make_manifest("candidate_execution", "eval_20260819_000000_bbbb"),
        "candidate_evaluation": make_manifest("candidate_evaluation", "rank_20260819_000000_cccc"),
        "preference_generation": make_manifest("preference_generation", "pref_20260819_000000_dddd"),
        "dpo_training": make_manifest("dpo_training", "dpo_20260819_000000_eeee"),
        "model_evaluation": make_manifest("model_evaluation", "eval_20260819_000000_ffff"),
        "packaging": make_manifest("packaging", "exp_x_packaging"),
    }
    lineage = build_lineage(upstream)

    assert lineage["model_adapter"]["training_run_id"] == "dpo_20260819_000000_eeee"
    assert lineage["model_adapter"]["preference_run_id"] == "pref_20260819_000000_dddd"
    assert lineage["model_adapter"]["ranking_run_id"] == "rank_20260819_000000_cccc"
    assert lineage["model_adapter"]["evaluation_run_id"] == "eval_20260819_000000_bbbb"
    assert lineage["model_adapter"]["candidate_run_id"] == "run_20260819_000000_aaaa"
    assert lineage["model_adapter"]["problem_dataset_run_id"] == "exp_x_problem_dataset"
    assert lineage["model_evaluation_run_id"] == "eval_20260819_000000_ffff"
    assert lineage["packaging_run_id"] == "exp_x_packaging"


def test_build_lineage_handles_missing_stages_as_none():
    lineage = build_lineage({"problem_dataset": make_manifest("problem_dataset", "exp_x_problem_dataset")})
    assert lineage["model_adapter"]["problem_dataset_run_id"] == "exp_x_problem_dataset"
    assert lineage["model_adapter"]["training_run_id"] is None
    assert lineage["model_evaluation_run_id"] is None
    assert lineage["packaging_run_id"] is None


def test_build_lineage_of_empty_upstream_is_all_none():
    lineage = build_lineage({})
    assert all(v is None for v in lineage["model_adapter"].values())
    assert lineage["model_evaluation_run_id"] is None
    assert lineage["packaging_run_id"] is None
