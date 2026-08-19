"""Tests for the stage dependency graph (spec 12 sections 5, 15, 16, 22)."""

from __future__ import annotations

import pytest

from python_dpo.pipeline.stages import (
    STAGE_NAMES,
    STAGES,
    StageGraphError,
    StageSpec,
    dependents_of,
    get_stage,
    resolve_adapter,
    topological_order,
    validate_graph,
)


def test_nine_stages_registered():
    assert len(STAGES) == 9
    assert len(STAGE_NAMES) == 9
    assert len(set(STAGE_NAMES)) == 9


def test_validate_graph_passes_for_the_real_graph():
    validate_graph()  # must not raise


def test_topological_order_respects_every_dependency_edge():
    order = topological_order()
    index = {name: i for i, name in enumerate(order)}
    for spec in STAGES:
        for dep in spec.requires:
            assert index[dep] < index[spec.name], f"{dep} must precede {spec.name}"


def test_topological_order_is_deterministic():
    assert topological_order() == topological_order()


def test_problem_dataset_has_no_dependencies():
    assert get_stage("problem_dataset").requires == ()


def test_packaging_requires_dpo_training_not_model_evaluation():
    # Spec section 15: packaging only needs a trained adapter.
    assert get_stage("packaging").requires == ("dpo_training",)


def test_error_analysis_requires_model_evaluation():
    assert get_stage("error_analysis").requires == ("model_evaluation",)


def test_get_stage_raises_on_unknown_name():
    with pytest.raises(StageGraphError, match="unknown stage"):
        get_stage("does_not_exist")


def test_topological_order_raises_on_cycle():
    cyclic = (
        StageSpec("a", ("b",), "mod:fn", "a"),
        StageSpec("b", ("a",), "mod:fn", "b"),
    )
    with _patched_stages(cyclic):
        with pytest.raises(StageGraphError, match="cycle"):
            topological_order(names=("a", "b"))


def test_validate_graph_rejects_unknown_dependency():
    bad = (StageSpec("a", ("missing",), "mod:fn", "a"),)
    with _patched_stages(bad):
        with pytest.raises(StageGraphError, match="unknown stage"):
            validate_graph()


def test_validate_graph_rejects_duplicate_name():
    bad = (
        StageSpec("a", (), "mod:fn", "a"),
        StageSpec("a", (), "mod:fn", "a"),
    )
    with _patched_stages(bad):
        with pytest.raises(StageGraphError, match="duplicate"):
            validate_graph()


def test_dependents_of_candidate_evaluation_matches_the_downstream_cascade():
    # Spec section 22's example: forcing candidate_evaluation invalidates everything
    # downstream, in order.
    assert dependents_of("candidate_evaluation") == (
        "preference_generation",
        "dpo_training",
        "model_evaluation",
        "error_analysis",
        "packaging",
    )


def test_dependents_of_dpo_training_includes_both_of_its_direct_dependents():
    assert dependents_of("dpo_training") == (
        "model_evaluation",
        "error_analysis",
        "packaging",
    )


def test_dependents_of_packaging_is_empty():
    assert dependents_of("packaging") == ()


def test_dependents_of_unknown_stage_raises():
    with pytest.raises(StageGraphError):
        dependents_of("does_not_exist")


def test_resolve_adapter_imports_the_named_function():
    spec = StageSpec(
        name="problem_dataset",
        requires=(),
        adapter="python_dpo.pipeline.stages:STAGE_NAMES",
        config_section="problem_dataset",
    )
    # STAGE_NAMES is not callable, but resolve_adapter only has to fetch the attribute --
    # this proves the module:function split and the getattr both work.
    from python_dpo.pipeline import stages as stages_module

    assert resolve_adapter(spec) is stages_module.STAGE_NAMES


def test_resolve_adapter_raises_on_malformed_reference():
    spec = StageSpec("x", (), "not-a-valid-reference", "x")
    with pytest.raises(StageGraphError, match="not a valid"):
        resolve_adapter(spec)


def test_resolve_adapter_raises_on_missing_attribute():
    spec = StageSpec("x", (), "python_dpo.pipeline.stages:does_not_exist", "x")
    with pytest.raises(StageGraphError, match="no attribute"):
        resolve_adapter(spec)


class _patched_stages:
    """Context manager swapping the module-level STAGES/lookup for one test."""

    def __init__(self, replacement):
        self._replacement = replacement

    def __enter__(self):
        import python_dpo.pipeline.stages as mod

        self._orig_stages = mod.STAGES
        self._orig_names = mod.STAGE_NAMES
        self._orig_by_name = mod._STAGES_BY_NAME
        mod.STAGES = self._replacement
        mod.STAGE_NAMES = tuple(s.name for s in self._replacement)
        mod._STAGES_BY_NAME = {s.name: s for s in self._replacement}
        return mod

    def __exit__(self, *exc_info):
        import python_dpo.pipeline.stages as mod

        mod.STAGES = self._orig_stages
        mod.STAGE_NAMES = self._orig_names
        mod._STAGES_BY_NAME = self._orig_by_name
