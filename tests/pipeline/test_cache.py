"""Tests for the cache key, reuse decision, and invalidation cascade
(spec 12 sections 17, 18, 19, 22, 91)."""

from __future__ import annotations

from python_dpo.pipeline.cache import cache_key, invalidate, is_reusable
from python_dpo.pipeline.manifest import StageManifest


def make_key(**overrides):
    fields = dict(
        stage="dpo_training",
        input_hashes={"preference_dataset": "a" * 64},
        configuration_hash="b" * 64,
        code_version="0.12.0",
        model_version="Qwen/Qwen2.5-Coder-3B-Instruct@main",
    )
    fields.update(overrides)
    return cache_key(**fields)


def test_cache_key_is_deterministic():
    assert make_key() == make_key()


def test_cache_key_changes_when_configuration_hash_changes():
    assert make_key(configuration_hash="b" * 64) != make_key(configuration_hash="c" * 64)


def test_cache_key_changes_when_input_hashes_change():
    assert make_key(input_hashes={"x": "1"}) != make_key(input_hashes={"x": "2"})


def test_cache_key_changes_when_code_version_changes():
    assert make_key(code_version="0.12.0") != make_key(code_version="0.13.0")


def test_cache_key_changes_when_model_version_changes():
    assert make_key(model_version="a@main") != make_key(model_version="b@main")


def test_cache_key_is_independent_of_input_hash_dict_order():
    left = make_key(input_hashes={"a": "1", "b": "2"})
    right = cache_key(
        stage="dpo_training",
        input_hashes={"b": "2", "a": "1"},
        configuration_hash="b" * 64,
        code_version="0.12.0",
        model_version="Qwen/Qwen2.5-Coder-3B-Instruct@main",
    )
    assert left == right


def test_cache_key_differs_between_stages_given_identical_other_inputs():
    assert make_key(stage="dpo_training") != make_key(stage="model_evaluation")


# ------------------------------------------------------------------------- is_reusable


def make_completed_manifest(cache_key_value: str) -> StageManifest:
    return StageManifest(
        stage_name="dpo_training",
        stage_run_id="dpo_x",
        status="COMPLETED",
        code_version="0.12.0",
        cache_key=cache_key_value,
    )


def test_is_reusable_true_on_matching_key():
    manifest = make_completed_manifest("abc123")
    assert is_reusable(manifest, "abc123") is True


def test_is_reusable_false_on_mismatched_key():
    manifest = make_completed_manifest("abc123")
    assert is_reusable(manifest, "different") is False


def test_is_reusable_false_when_manifest_is_none():
    assert is_reusable(None, "abc123") is False


def test_is_reusable_false_when_stage_not_completed():
    manifest = StageManifest(
        stage_name="dpo_training",
        stage_run_id="dpo_x",
        status="FAILED",
        code_version="0.12.0",
        cache_key="abc123",
    )
    assert is_reusable(manifest, "abc123") is False


# -------------------------------------------------------------------------- invalidate


def test_invalidate_with_cascade_includes_every_dependent():
    # Spec section 22's literal example lists candidate-evaluation, preference-generation,
    # dpo-training and model-evaluation rerunning in that order; packaging and
    # error_analysis are graph-parallel siblings below dpo_training/model_evaluation with
    # no edge between them, so their relative order is not asserted here.
    result = invalidate("candidate_evaluation", cascade=True)
    assert set(result) == {
        "candidate_evaluation",
        "preference_generation",
        "dpo_training",
        "model_evaluation",
        "error_analysis",
        "packaging",
    }
    index = {name: i for i, name in enumerate(result)}
    assert index["candidate_evaluation"] < index["preference_generation"]
    assert index["preference_generation"] < index["dpo_training"]
    assert index["dpo_training"] < index["model_evaluation"]
    assert index["dpo_training"] < index["packaging"]
    assert index["model_evaluation"] < index["error_analysis"]


def test_invalidate_without_cascade_returns_only_the_named_stage():
    assert invalidate("candidate_evaluation", cascade=False) == ("candidate_evaluation",)


def test_invalidate_of_a_leaf_stage_returns_only_itself():
    assert invalidate("packaging", cascade=True) == ("packaging",)


def test_dpo_beta_change_leaves_problem_and_candidate_generation_cached():
    """Spec section 91's exact scenario: changing DPO beta invalidates training onward,
    but problem_dataset and candidate_generation are structurally unreachable."""
    invalidated = set(invalidate("dpo_training", cascade=True))
    assert "problem_dataset" not in invalidated
    assert "candidate_generation" not in invalidated
    assert "candidate_execution" not in invalidated
    assert "candidate_evaluation" not in invalidated
    assert "preference_generation" not in invalidated
    assert invalidated == {"dpo_training", "model_evaluation", "error_analysis", "packaging"}
