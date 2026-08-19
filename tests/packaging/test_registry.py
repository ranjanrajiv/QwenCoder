"""Tests for ``models/registry.json`` (spec 12 sections 45-48)."""

from __future__ import annotations

import pytest

from python_dpo.packaging.errors import RegistryError
from python_dpo.packaging.registry import ModelRegistry, RegistryEntry


def make_entry(**overrides) -> RegistryEntry:
    fields = {
        "model_id": "exp_x",
        "status": "EXPERIMENTAL",
        "package_path": "data/experiments/runs/exp_x/model",
        "base_model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "training_run_id": "dpo_x",
        "created_at": "2026-08-19T10:00:00Z",
        "verification": {"ok": True},
    }
    fields.update(overrides)
    return RegistryEntry(**fields)


def test_register_persists_and_round_trips(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry())

    reloaded = ModelRegistry(tmp_path / "models" / "registry.json")
    entry = reloaded.get("exp_x")
    assert entry.status == "EXPERIMENTAL"
    assert entry.verification == {"ok": True}


def test_register_only_accepts_experimental(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    with pytest.raises(RegistryError, match="EXPERIMENTAL"):
        registry.register(make_entry(status="RECOMMENDED"))


def test_register_rejects_a_duplicate_model_id(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry())
    with pytest.raises(RegistryError, match="already registered"):
        registry.register(make_entry())


def test_get_raises_for_an_unknown_model(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    with pytest.raises(RegistryError, match="no registered model"):
        registry.get("does-not-exist")


def test_list_is_newest_first(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry(model_id="a", created_at="2026-08-19T09:00:00Z"))
    registry.register(make_entry(model_id="b", created_at="2026-08-19T11:00:00Z"))

    ids = [entry.model_id for entry in registry.list()]
    assert ids == ["b", "a"]


def test_promote_to_recommended_requires_an_evaluation_run(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry())
    registry.promote("exp_x", "VALIDATED")

    with pytest.raises(RegistryError, match="evaluation_run_id"):
        registry.promote("exp_x", "RECOMMENDED")


def test_promote_to_recommended_requires_passing_success_criteria(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry())
    registry.promote("exp_x", "VALIDATED")

    with pytest.raises(RegistryError, match="did not pass"):
        registry.promote(
            "exp_x", "RECOMMENDED",
            evaluation_run_id="eval_x", success_criteria_passed=False,
        )


def test_promote_to_recommended_succeeds_with_a_passing_record(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry())
    registry.promote("exp_x", "VALIDATED")

    entry = registry.promote(
        "exp_x", "RECOMMENDED",
        evaluation_run_id="eval_x", success_criteria_passed=True,
    )

    assert entry.status == "RECOMMENDED"
    assert entry.evaluation_run_id == "eval_x"


def test_promote_never_skips_validated_on_the_way_to_recommended(tmp_path):
    """Section 47: RECOMMENDED is reachable only via VALIDATED, so nothing is ever
    recommended sight-unseen straight from EXPERIMENTAL."""
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry())

    with pytest.raises(RegistryError, match="illegal registry status transition"):
        registry.promote(
            "exp_x", "RECOMMENDED",
            evaluation_run_id="eval_x", success_criteria_passed=True,
        )


def test_promote_from_a_terminal_status_is_rejected(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry())
    registry.promote("exp_x", "REJECTED")

    with pytest.raises(RegistryError, match="illegal registry status transition"):
        registry.promote("exp_x", "VALIDATED")


def test_registry_file_rejects_unknown_top_level_keys(tmp_path):
    path = tmp_path / "models" / "registry.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"registry_version": "registry_v1", "models": {}, "bogus": 1}', encoding="utf-8")

    with pytest.raises(RegistryError, match="unknown top-level key"):
        ModelRegistry(path).list()
