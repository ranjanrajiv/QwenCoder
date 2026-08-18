"""Tests for content-addressed cache keys (spec sections 92-94, 138, 139)."""

from __future__ import annotations

from python_dpo.model_evaluation.cache import (
    EvaluationCacheKey,
    GenerationCacheKey,
    JsonCacheStore,
)


def base_key(**overrides) -> GenerationCacheKey:
    fields = dict(
        model_variant="base",
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        model_revision=None,
        adapter_identity=None,
        prompt_sha256="abc123",
        generation_config={"temperature": 0.2, "top_p": 0.95},
        seed=1000,
    )
    fields.update(overrides)
    return GenerationCacheKey(**fields)


def test_same_prompt_different_variant_yields_different_key():
    """Spec section 94: model identity is always part of the key."""
    base = base_key(model_variant="base", adapter_identity=None)
    dpo = base_key(model_variant="dpo", adapter_identity="/data/adapter")
    assert base.digest != dpo.digest


def test_identical_inputs_yield_identical_digest():
    a = base_key()
    b = base_key()
    assert a.digest == b.digest


def test_digest_changes_on_model_revision():
    a = base_key(model_revision=None)
    b = base_key(model_revision="abc123def")
    assert a.digest != b.digest


def test_digest_changes_on_generation_config():
    a = base_key(generation_config={"temperature": 0.2})
    b = base_key(generation_config={"temperature": 0.8})
    assert a.digest != b.digest


def test_digest_changes_on_seed():
    a = base_key(seed=1000)
    b = base_key(seed=1001)
    assert a.digest != b.digest


def test_digest_changes_on_prompt():
    a = base_key(prompt_sha256="abc")
    b = base_key(prompt_sha256="def")
    assert a.digest != b.digest


def test_evaluation_cache_key_changes_on_evaluator_version():
    a = EvaluationCacheKey(
        candidate_code_sha256="codehash",
        problem_id="p001",
        test_suite_hash="testhash",
        evaluator_version="v1",
        sandbox_config={"image": "python-dpo-evaluator:1.0"},
    )
    b = EvaluationCacheKey(
        candidate_code_sha256="codehash",
        problem_id="p001",
        test_suite_hash="testhash",
        evaluator_version="v2",
        sandbox_config={"image": "python-dpo-evaluator:1.0"},
    )
    assert a.digest != b.digest


def test_json_cache_store_round_trips(tmp_path):
    store = JsonCacheStore(tmp_path / "cache.json")
    assert store.get("missing") is None
    store.put("key1", {"value": 42})
    assert store.get("key1") == {"value": 42}

    # A fresh store reading the same path sees the persisted value.
    reloaded = JsonCacheStore(tmp_path / "cache.json")
    assert reloaded.get("key1") == {"value": 42}
