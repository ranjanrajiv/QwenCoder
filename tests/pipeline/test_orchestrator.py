"""Tests for PipelineOrchestrator (spec 12 sections 7, 16, 17, 22, 66, 67, 89, 90)."""

from __future__ import annotations

from python_dpo.pipeline.config import ExperimentConfig
from python_dpo.pipeline.errors import StageFailedError
from python_dpo.pipeline.orchestrator import PipelineOrchestrator
from python_dpo.pipeline.stages import STAGE_NAMES

from .conftest import (
    ALL_ADAPTER_STAGES,
    full_experiment_mapping,
    install_all_stub_adapters,
    install_stub_adapter,
)


def make_orchestrator(project_config, experiment_repo):
    return PipelineOrchestrator(project_config, experiment_repo)


def test_full_run_completes_every_enabled_stage_in_order(project_config, experiment_repo, monkeypatch):
    calls: list[str] = []
    install_all_stub_adapters(monkeypatch, calls)
    experiment = ExperimentConfig.from_mapping(full_experiment_mapping())
    orchestrator = make_orchestrator(project_config, experiment_repo)

    manifest = orchestrator.run(experiment)

    assert manifest.status == "completed"
    assert calls == list(ALL_ADAPTER_STAGES)  # topological order, error_analysis skipped
    for name in ALL_ADAPTER_STAGES:
        sm = experiment_repo.read_stage_manifest(manifest.experiment_run_id, name)
        assert sm.status == "COMPLETED"
    skipped = experiment_repo.read_stage_manifest(manifest.experiment_run_id, "error_analysis")
    assert skipped.status == "SKIPPED"


def test_enabled_order_excludes_disabled_stages(project_config, experiment_repo):
    experiment = ExperimentConfig.from_mapping(full_experiment_mapping())
    orchestrator = make_orchestrator(project_config, experiment_repo)
    order = orchestrator.enabled_order(experiment)
    assert "error_analysis" not in order
    assert order == tuple(n for n in STAGE_NAMES if n != "error_analysis")


def test_dependency_error_fails_the_stage_when_upstream_is_disabled(
    project_config, experiment_repo, monkeypatch
):
    # candidate_execution is enabled but candidate_generation (its only dependency) is not.
    calls: list[str] = []
    install_stub_adapter(monkeypatch, "problem_dataset", calls=calls)
    install_stub_adapter(monkeypatch, "candidate_execution", calls=calls)
    experiment = ExperimentConfig.from_mapping(
        full_experiment_mapping(enabled={"candidate_generation": False})
    )
    orchestrator = make_orchestrator(project_config, experiment_repo)

    manifest = orchestrator.run(experiment)

    assert manifest.status == "failed"
    sm = experiment_repo.read_stage_manifest(manifest.experiment_run_id, "candidate_execution")
    assert sm.status == "FAILED"
    assert sm.error.error_type == "DependencyError"
    # never called -- the missing-dependency check happens before the adapter runs
    assert "candidate_execution" not in calls


def test_stage_failure_blocks_every_downstream_stage(project_config, experiment_repo, monkeypatch):
    calls: list[str] = []
    install_stub_adapter(monkeypatch, "problem_dataset", calls=calls)
    install_stub_adapter(monkeypatch, "candidate_generation", calls=calls)
    install_stub_adapter(
        monkeypatch, "candidate_execution", raises=StageFailedError("sandbox exploded"), calls=calls
    )
    for stage_name in ("candidate_evaluation", "preference_generation", "dpo_training", "model_evaluation", "packaging"):
        install_stub_adapter(monkeypatch, stage_name, calls=calls)
    experiment = ExperimentConfig.from_mapping(full_experiment_mapping())
    orchestrator = make_orchestrator(project_config, experiment_repo)

    manifest = orchestrator.run(experiment)

    assert manifest.status == "failed"
    assert calls == ["problem_dataset", "candidate_generation", "candidate_execution"]

    assert experiment_repo.read_stage_manifest(manifest.experiment_run_id, "problem_dataset").status == "COMPLETED"
    assert experiment_repo.read_stage_manifest(manifest.experiment_run_id, "candidate_generation").status == "COMPLETED"
    assert experiment_repo.read_stage_manifest(manifest.experiment_run_id, "candidate_execution").status == "FAILED"
    for stage_name in ("candidate_evaluation", "preference_generation", "dpo_training", "model_evaluation", "packaging"):
        sm = experiment_repo.read_stage_manifest(manifest.experiment_run_id, stage_name)
        assert sm.status == "BLOCKED", stage_name


def test_keyboard_interrupt_cancels_the_in_flight_stage_and_interrupts_the_experiment(
    project_config, experiment_repo, monkeypatch
):
    calls: list[str] = []
    install_stub_adapter(monkeypatch, "problem_dataset", calls=calls)
    install_stub_adapter(monkeypatch, "candidate_generation", raises=KeyboardInterrupt(), calls=calls)
    experiment = ExperimentConfig.from_mapping(full_experiment_mapping())
    orchestrator = make_orchestrator(project_config, experiment_repo)

    manifest = orchestrator.run(experiment)

    assert manifest.status == "interrupted"
    assert experiment_repo.read_stage_manifest(manifest.experiment_run_id, "problem_dataset").status == "COMPLETED"
    cancelled = experiment_repo.read_stage_manifest(manifest.experiment_run_id, "candidate_generation")
    assert cancelled.status == "CANCELLED"
    # stages after the interrupted one are never touched
    assert experiment_repo.read_stage_manifest(manifest.experiment_run_id, "candidate_execution") is None


def test_second_identical_run_reuses_every_stage(project_config, experiment_repo, monkeypatch):
    """Spec section 90: run the same experiment twice; the second reuses compatible
    artifacts rather than recomputing them."""
    calls: list[str] = []
    install_all_stub_adapters(monkeypatch, calls)
    experiment = ExperimentConfig.from_mapping(full_experiment_mapping())
    orchestrator = make_orchestrator(project_config, experiment_repo)

    orchestrator.run(experiment)
    first_call_count = len(calls)
    assert first_call_count == len(ALL_ADAPTER_STAGES)

    second_manifest = orchestrator.run(experiment)

    assert second_manifest.status == "completed"
    assert len(calls) == first_call_count, "no adapter should have been called again"
    for name in ALL_ADAPTER_STAGES:
        sm = experiment_repo.read_stage_manifest(second_manifest.experiment_run_id, name)
        assert sm.status == "COMPLETED"
        assert sm.reused is True


def test_force_reruns_only_the_named_stage_and_its_dependents(project_config, experiment_repo, monkeypatch):
    calls: list[str] = []
    install_all_stub_adapters(monkeypatch, calls)
    experiment = ExperimentConfig.from_mapping(full_experiment_mapping())
    orchestrator = make_orchestrator(project_config, experiment_repo)

    orchestrator.run(experiment)
    calls.clear()

    orchestrator.run(experiment, force="candidate_evaluation")

    # candidate_evaluation and everything downstream of it reruns; everything upstream
    # (problem_dataset, candidate_generation, candidate_execution) is reused.
    assert set(calls) == {
        "candidate_evaluation",
        "preference_generation",
        "dpo_training",
        "model_evaluation",
        "packaging",
    }
    assert "problem_dataset" not in calls
    assert "candidate_generation" not in calls
    assert "candidate_execution" not in calls


def test_resume_reuses_completed_stages_and_reruns_only_the_failed_one(
    project_config, experiment_repo, monkeypatch
):
    """Spec section 89: interrupt (here: fail) after an early stage, resume, verify
    earlier stages are reused and the failed one is re-executed."""
    calls: list[str] = []
    install_stub_adapter(monkeypatch, "problem_dataset", calls=calls)
    install_stub_adapter(monkeypatch, "candidate_generation", calls=calls)
    install_stub_adapter(
        monkeypatch, "candidate_execution", raises=StageFailedError("docker was down"), calls=calls
    )
    experiment = ExperimentConfig.from_mapping(
        full_experiment_mapping(
            enabled={
                "candidate_evaluation": False,
                "preference_generation": False,
                "dpo_training": False,
                "model_evaluation": False,
                "packaging": False,
            }
        )
    )
    orchestrator = make_orchestrator(project_config, experiment_repo)

    first = orchestrator.run(experiment)
    assert first.status == "failed"
    assert calls == ["problem_dataset", "candidate_generation", "candidate_execution"]
    calls.clear()

    # Fix candidate_execution and resume.
    install_stub_adapter(monkeypatch, "candidate_execution", calls=calls)
    second = orchestrator.run(experiment, resume_run_id=first.experiment_run_id)

    assert second.experiment_run_id == first.experiment_run_id
    assert second.status == "completed"
    assert calls == ["candidate_execution"]  # earlier stages reused, not re-run
    problem_dataset_sm = experiment_repo.read_stage_manifest(second.experiment_run_id, "problem_dataset")
    assert problem_dataset_sm.reused is False  # never invalidated, so its own original run
    candidate_execution_sm = experiment_repo.read_stage_manifest(second.experiment_run_id, "candidate_execution")
    assert candidate_execution_sm.status == "COMPLETED"


def test_resume_of_a_completed_experiment_raises(project_config, experiment_repo, monkeypatch):
    from python_dpo.pipeline.errors import PipelineError

    calls: list[str] = []
    install_stub_adapter(monkeypatch, "problem_dataset", calls=calls)
    experiment = ExperimentConfig.from_mapping(
        full_experiment_mapping(
            enabled={n: False for n in STAGE_NAMES if n != "problem_dataset"}
        )
    )
    orchestrator = make_orchestrator(project_config, experiment_repo)
    manifest = orchestrator.run(experiment)
    assert manifest.status == "completed"

    import pytest

    with pytest.raises(PipelineError, match="cannot be resumed"):
        orchestrator.run(experiment, resume_run_id=manifest.experiment_run_id)


def test_experiment_manifest_records_git_commit(project_config, experiment_repo, monkeypatch):
    """Git capture is stubbed rather than read from the ambient checkout: `project_config`
    is rooted at a tmp_path that is not a repository, and asserting only that a `sha` key
    exists would pass even when its value is None. Stubbing pins the value that actually
    reaches the manifest."""
    calls: list[str] = []
    install_stub_adapter(monkeypatch, "problem_dataset", calls=calls)
    monkeypatch.setattr(
        "python_dpo.pipeline.orchestrator.capture_git_info",
        lambda root: {"sha": "abc1234", "branch": "main", "dirty": False},
    )
    experiment = ExperimentConfig.from_mapping(
        full_experiment_mapping(enabled={n: False for n in STAGE_NAMES if n != "problem_dataset"})
    )
    orchestrator = make_orchestrator(project_config, experiment_repo)
    manifest = orchestrator.run(experiment)
    assert manifest.git_commit == {"sha": "abc1234", "branch": "main", "dirty": False}


def test_run_requires_an_experiment_unless_resuming(project_config, experiment_repo):
    from python_dpo.pipeline.errors import PipelineError

    import pytest

    orchestrator = make_orchestrator(project_config, experiment_repo)
    with pytest.raises(PipelineError, match="requires an experiment"):
        orchestrator.run()


def test_fail_on_dirty_never_creates_a_run_directory(project_config, experiment_repo, monkeypatch):
    from python_dpo.pipeline.gitinfo import GitInfoError

    import pytest

    monkeypatch.setattr(
        "python_dpo.pipeline.orchestrator.capture_git_info",
        lambda root: {"sha": "abc", "branch": "main", "dirty": True},
    )
    experiment = ExperimentConfig.from_mapping(
        full_experiment_mapping(enabled={n: False for n in STAGE_NAMES})
    )
    experiment = ExperimentConfig.from_mapping(
        {**experiment.to_dict(), "git": {"on_dirty": "fail"}}
    )
    orchestrator = make_orchestrator(project_config, experiment_repo)

    with pytest.raises(GitInfoError):
        orchestrator.run(experiment)

    assert experiment_repo.list_runs() == []
