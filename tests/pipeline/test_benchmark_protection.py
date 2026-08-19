"""Spec 12 section 92: adding a held-out benchmark problem to a training split must FAIL.

The ``model_evaluation`` stage adapter derives the preference run to check leakage
against from the training run's own manifest (``training_manifest.preference_run_id``)
rather than a separate setting, so there is no way to silently skip the check by
omitting one. This runs entirely offline: leakage is checked before any GPU or Docker
resource is touched, so a leaking benchmark fails fast without either.
"""

from __future__ import annotations

import pytest

from python_dpo.model_evaluation import build_benchmark, save_benchmark
from python_dpo.pipeline.config import StageConfig
from python_dpo.pipeline.errors import StageFailedError
from python_dpo.pipeline.manifest import StageManifest
from python_dpo.pipeline.stages._context import StageContext
from python_dpo.pipeline.stages.model_evaluation import run as model_evaluation_run
from python_dpo.preferences import PreferenceRunRepository
from python_dpo.preferences.splitter import SplitManifest
from python_dpo.problems import build_catalog, dataset_path, load_problems, save_problems
from python_dpo.runs import RunRepository
from python_dpo.training.run_repository import TrainingRunRepository


def _seed_problems(project_config):
    problems = build_catalog()
    save_problems(problems, dataset_path(project_config.paths.problems))
    return problems


def _seed_preference_split(project_config, preference_run_id, leaked_problem_id):
    repo = PreferenceRunRepository(project_config.paths.preferences / "runs")
    repo.create_run(
        ranking_run_id="rank_x",
        evaluation_run_id="eval_x",
        candidate_run_id="run_x",
        preference_version="dpo_preference_v1",
        selection_policy="strict",
        selection_policy_version="1.0",
        minimum_score_margin=0.2,
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        split_seed=42,
        builder_version="1.0",
        max_pairs_per_problem=None,
        preference_run_id=preference_run_id,
    )
    split_manifest = SplitManifest(
        train_problem_ids=(leaked_problem_id,),
        validation_problem_ids=(),
        test_problem_ids=(),
        seed=42,
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
    )
    repo.write_split_manifest(preference_run_id, split_manifest)
    return split_manifest


def _seed_training_run(project_config, training_run_id, preference_run_id):
    repo = TrainingRunRepository(project_config.paths.training / "runs")
    repo.create_run(
        experiment_name="x",
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        model_revision=None,
        tokenizer_revision=None,
        preference_run_id=preference_run_id,
        ranking_run_id="rank_x",
        evaluation_run_id="eval_x",
        candidate_run_id="run_x",
        dataset_hashes={"train": "a" * 64, "validation": "b" * 64, "test": "c" * 64},
        hardware={},
        environment={},
        configuration={},
        seed=42,
        data_seed=42,
        trainer_version="1.0",
        training_run_id=training_run_id,
    )


def test_model_evaluation_fails_when_a_benchmark_problem_leaked_into_training(project_config):
    problems = _seed_problems(project_config)
    leaked_problem_id = problems[0].id

    preference_run_id = "pref_leak_test"
    _seed_preference_split(project_config, preference_run_id, leaked_problem_id)

    training_run_id = "dpo_leak_test"
    _seed_training_run(project_config, training_run_id, preference_run_id)

    benchmark_manifest = build_benchmark("leak_test_v1", problems, [leaked_problem_id])
    save_benchmark(project_config.project_root / "benchmarks", benchmark_manifest)

    context = StageContext(
        experiment_run_id="exp_leak_test",
        stage_config=StageConfig(
            name="model_evaluation", enabled=True, settings={"benchmark": "leak_test_v1"}
        ),
        project_config=project_config,
        experiment_repo=None,
        upstream={
            "dpo_training": StageManifest(
                stage_name="dpo_training",
                stage_run_id=training_run_id,
                status="COMPLETED",
                code_version="0.12.0",
            )
        },
    )

    with pytest.raises(StageFailedError, match="overlaps"):
        model_evaluation_run(context)


def test_model_evaluation_proceeds_past_the_leakage_check_when_benchmark_is_clean(
    project_config, monkeypatch
):
    """A clean benchmark passes the leakage gate and moves on to the next step.

    This machine may have a real GPU and Docker, so the next real step (hardware, then
    actual model loading) is forced to fail deterministically rather than relying on
    their absence -- otherwise this test would try to download and load the real Qwen
    model, which is exactly what an offline test must never do.
    """
    from python_dpo.training.hardware import HardwareCheck, HardwareInfo, HardwareReport

    monkeypatch.setattr(
        "python_dpo.pipeline.stages.model_evaluation.check_hardware",
        lambda **kwargs: HardwareReport(
            checks=(HardwareCheck("forced failure", False, "test forces this path to stop here"),),
            info=HardwareInfo(cuda_available=False),
        ),
    )

    problems = _seed_problems(project_config)
    trained_on = problems[0].id
    held_out = problems[1].id

    preference_run_id = "pref_clean_test"
    _seed_preference_split(project_config, preference_run_id, trained_on)

    training_run_id = "dpo_clean_test"
    _seed_training_run(project_config, training_run_id, preference_run_id)

    benchmark_manifest = build_benchmark("clean_test_v1", problems, [held_out])
    save_benchmark(project_config.project_root / "benchmarks", benchmark_manifest)

    context = StageContext(
        experiment_run_id="exp_clean_test",
        stage_config=StageConfig(
            name="model_evaluation", enabled=True, settings={"benchmark": "clean_test_v1"}
        ),
        project_config=project_config,
        experiment_repo=None,
        upstream={
            "dpo_training": StageManifest(
                stage_name="dpo_training",
                stage_run_id=training_run_id,
                status="COMPLETED",
                code_version="0.12.0",
            )
        },
    )

    # No leakage error; whatever failure occurs next is unrelated to leakage.
    with pytest.raises(StageFailedError) as excinfo:
        model_evaluation_run(context)
    assert "overlaps" not in str(excinfo.value)
