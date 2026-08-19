"""Tests for lineage resolution (spec 11 section 7).

Section 7 makes the lineage a precondition rather than an enrichment: an analysis that does
not know which preference dataset trained the adapter cannot make a coverage claim about
it, so a broken hop raises instead of degrading.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from python_dpo.analysis.errors import LineageError
from python_dpo.analysis.ingest import resolve_lineage
from python_dpo.model_evaluation.errors import EvaluationRunNotFoundError
from python_dpo.training.errors import TrainingRunNotFoundError


@dataclass
class FakeEvalManifest:
    training_run_id: str | None = "dpo_x"
    benchmark_version: str = "v1"


@dataclass
class FakeTrainingManifest:
    preference_run_id: str | None = "pref_x"
    ranking_run_id: str | None = "rank_x"
    candidate_run_id: str | None = "run_x"
    # Deliberately a Stage 6 id, not a Stage 10 one -- the collision this test pins down.
    evaluation_run_id: str = "eval_stage6_x"


class FakeEvalRepo:
    def __init__(self, manifest=None, raises=False):
        self._manifest = manifest or FakeEvalManifest()
        self._raises = raises

    def get_run(self, run_id):
        if self._raises:
            raise EvaluationRunNotFoundError(f"no run {run_id}")
        return self._manifest


class FakeTrainingRepo:
    def __init__(self, manifest=None, raises=False):
        self._manifest = manifest or FakeTrainingManifest()
        self._raises = raises

    def get_run(self, run_id):
        if self._raises:
            raise TrainingRunNotFoundError(f"no run {run_id}")
        return self._manifest


def test_lineage_resolves_the_full_chain():
    lineage = resolve_lineage("eval_x", FakeEvalRepo(), FakeTrainingRepo())
    assert lineage.evaluation_run_id == "eval_x"
    assert lineage.training_run_id == "dpo_x"
    assert lineage.preference_run_id == "pref_x"
    assert lineage.ranking_run_id == "rank_x"
    assert lineage.candidate_run_id == "run_x"


def test_the_stage6_evaluation_id_is_kept_separate_from_the_stage10_one():
    """Both are called `evaluation_run_id` and both start `eval_`, but they name runs in
    different stores. Conflating them would point the analysis at the wrong artifacts."""
    lineage = resolve_lineage("eval_stage10_x", FakeEvalRepo(), FakeTrainingRepo())
    assert lineage.evaluation_run_id == "eval_stage10_x"
    assert lineage.sandbox_evaluation_run_id == "eval_stage6_x"


def test_a_missing_evaluation_run_raises_lineage_error():
    with pytest.raises(LineageError, match="no Stage 10 evaluation run"):
        resolve_lineage("missing", FakeEvalRepo(raises=True), FakeTrainingRepo())


def test_an_evaluation_without_a_training_run_raises():
    repo = FakeEvalRepo(FakeEvalManifest(training_run_id=None))
    with pytest.raises(LineageError, match="records no training_run_id"):
        resolve_lineage("eval_x", repo, FakeTrainingRepo())


def test_a_missing_training_run_raises():
    with pytest.raises(LineageError, match="is missing"):
        resolve_lineage("eval_x", FakeEvalRepo(), FakeTrainingRepo(raises=True))


@pytest.mark.parametrize(
    "field", ["preference_run_id", "ranking_run_id", "candidate_run_id"]
)
def test_a_broken_hop_raises_rather_than_analysing_a_partial_chain(field):
    manifest = FakeTrainingManifest(**{field: None})
    with pytest.raises(LineageError, match=f"records no {field}"):
        resolve_lineage("eval_x", FakeEvalRepo(), FakeTrainingRepo(manifest))


def test_an_override_that_contradicts_the_manifest_is_an_error():
    """Silently analysing a different dataset than the one that trained the adapter is
    exactly the failure this stage exists to prevent."""
    with pytest.raises(LineageError, match="does not match"):
        resolve_lineage(
            "eval_x", FakeEvalRepo(), FakeTrainingRepo(), preference_run_id="pref_other"
        )
    with pytest.raises(LineageError, match="does not match"):
        resolve_lineage(
            "eval_x", FakeEvalRepo(), FakeTrainingRepo(), training_run_id="dpo_other"
        )


def test_an_override_that_agrees_with_the_manifest_is_accepted():
    lineage = resolve_lineage(
        "eval_x", FakeEvalRepo(), FakeTrainingRepo(),
        preference_run_id="pref_x", training_run_id="dpo_x",
    )
    assert lineage.preference_run_id == "pref_x"
