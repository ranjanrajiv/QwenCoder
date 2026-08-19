"""Stage 9 as a pipeline stage: DPO/QLoRA training (spec 12 section 5, item 6).

Mirrors ``python_dpo.cli._cmd_train_dpo``'s sequence, adapted from an ``argparse.Namespace``
to this stage's settings mapping. GPU/torch imports stay inside
:mod:`python_dpo.training`, exactly as they already were -- this module imports the
package eagerly, so it must never be imported at :mod:`python_dpo.pipeline` module scope
(only via :func:`python_dpo.pipeline.stages.resolve_adapter`, on demand).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...preferences import PreferenceRunRepository
from ...training import (
    TRAINER_VERSION,
    DatasetManifest,
    DpoTrainingJob,
    ExperimentConfig as TrainingExperimentConfig,
    FinalReport,
    TrainingError,
    check_hardware,
    format_exception,
    load_training_dataset,
)
from ...training.config import DEFAULT_CONFIG_PATH
from ...training.run_repository import TrainingRunRepository, run_log_file
from ...training.versions import capture_environment
from ..errors import StageFailedError
from ..hashing import sha256_tree
from ._context import StageContext, StageResult

logger = logging.getLogger("python_dpo.pipeline.stages.dpo_training")


def run(context: StageContext) -> StageResult:
    config = context.project_config
    settings = context.stage_config

    config_path = Path(settings.get("config")) if settings.get("config") else DEFAULT_CONFIG_PATH
    try:
        experiment = TrainingExperimentConfig.load(config_path)
    except TrainingError as exc:
        raise StageFailedError(str(exc)) from exc

    preference_run_id = context.upstream_run_id("preference_generation")
    try:
        experiment = experiment.with_overrides(
            preference_run_id=preference_run_id,
            experiment_name=settings.get("experiment_name"),
            learning_rate=settings.get("learning_rate"),
            beta=settings.get("beta"),
            num_train_epochs=settings.get("epochs"),
            max_steps=settings.get("max_steps"),
            seed=settings.get("seed", context.seed),
            lora_r=settings.get("lora_r"),
        )
    except TrainingError as exc:
        raise StageFailedError(str(exc)) from exc

    preference_run_dir = PreferenceRunRepository(config.paths.preferences / "runs").run_dir(
        preference_run_id
    )
    allow_small_dataset = bool(settings.get("allow_small_dataset", False))
    try:
        dataset = load_training_dataset(
            preference_run_dir,
            allow_small_dataset=allow_small_dataset,
            min_training_pairs=experiment.training.min_training_pairs,
        )
    except TrainingError as exc:
        raise StageFailedError(str(exc)) from exc

    mode = "smoke_test" if settings.get("smoke_test", False) else "train"

    training_run_repo = TrainingRunRepository(config.paths.training / "runs")
    hardware_report = check_hardware()
    environment = capture_environment()

    manifest = training_run_repo.create_run(
        experiment_name=experiment.experiment_name,
        model_name=experiment.model.name,
        model_revision=experiment.model.revision,
        tokenizer_revision=experiment.model.revision,
        preference_run_id=dataset.preference_run_id,
        ranking_run_id=dataset.provenance["ranking_run_id"],
        evaluation_run_id=dataset.provenance["evaluation_run_id"],
        candidate_run_id=dataset.provenance["candidate_run_id"],
        dataset_hashes=dataset.split_hashes,
        hardware=hardware_report.info.to_dict(),
        environment=environment,
        configuration=experiment.to_dict(),
        seed=experiment.training.seed,
        data_seed=experiment.training.data_seed,
        trainer_version=TRAINER_VERSION,
        mode=mode,
    )
    run_id = manifest.training_run_id

    training_run_repo.write_config(run_id, experiment.to_dict())
    training_run_repo.write_hardware(run_id, hardware_report.info.to_dict())
    training_run_repo.write_dataset_manifest(
        run_id,
        DatasetManifest(
            preference_run_id=dataset.preference_run_id,
            preference_version=dataset.provenance["preference_version"],
            selection_policy=dataset.provenance["selection_policy"],
            selection_policy_version=dataset.provenance["selection_policy_version"],
            dataset_schema_version=dataset.provenance["dataset_schema_version"],
            ranking_run_id=dataset.provenance["ranking_run_id"],
            evaluation_run_id=dataset.provenance["evaluation_run_id"],
            candidate_run_id=dataset.provenance["candidate_run_id"],
            split_hashes=dataset.split_hashes,
            split_counts=dataset.split_counts,
            split_problem_ids=dataset.split_problem_ids,
            statistics={"splits": dataset.statistics, "balance": dataset.balance.to_dict()},
        ),
    )
    training_run_repo.start_run(run_id)
    logger.info(
        "Training run %s created | mode=%s | model=%s | dataset=%s",
        run_id,
        mode,
        experiment.model.name,
        dataset.preference_run_id,
    )

    job = DpoTrainingJob(
        experiment,
        dataset,
        training_run_repo,
        run_id,
        mode=mode,
        allow_small_dataset=allow_small_dataset,
        override_truncation=bool(settings.get("override_truncation", False)),
    )

    try:
        with run_log_file(training_run_repo.log_path(run_id)):
            outcome = job.run()
    except TrainingError as exc:
        error_type, message, tb = format_exception(exc)
        training_run_repo.fail_run(run_id, error_type=error_type, error_message=message, traceback_text=tb)
        raise StageFailedError(f"training run {run_id} failed: {message}") from exc
    except Exception as exc:  # noqa: BLE001 - spec section 82: record, then re-report
        error_type, message, tb = format_exception(exc)
        training_run_repo.fail_run(run_id, error_type=error_type, error_message=message, traceback_text=tb)
        raise StageFailedError(f"training run {run_id} failed ({error_type}): {message}") from exc

    if mode == "dry_run":
        training_run_repo.complete_run(run_id)
        return StageResult(stage_run_id=run_id, output_artifacts={})

    preflight = outcome.preflight
    counts = preflight.parameter_counts
    report = FinalReport(
        training_run_id=run_id,
        experiment_name=experiment.experiment_name,
        status="completed",
        model_name=experiment.model.name,
        model_revision=experiment.model.revision,
        preference_run_id=dataset.preference_run_id,
        number_of_examples={
            "train": dataset.split_counts["train"],
            "validation": dataset.split_counts["validation"],
        },
        epochs=outcome.epochs,
        steps=outcome.steps,
        final_train_loss=outcome.final_train_loss,
        final_eval_loss=outcome.final_eval_loss,
        reward_metrics=outcome.reward_metrics,
        peak_gpu_memory_bytes=outcome.peak_gpu_memory_bytes,
        trainable_parameters=counts.trainable if counts else 0,
        total_parameters=counts.total if counts else 0,
        effective_batch_size=experiment.effective_batch_size,
        optimizer=preflight.optimizer_name,
        compute_dtype=preflight.compute_dtype,
        adapter_path=outcome.adapter_path,
        checkpoint_path=outcome.checkpoint_path,
        adapter_reload_ok=bool(outcome.reload and outcome.reload.ok),
        training_duration_seconds=outcome.duration_seconds,
    )
    training_run_repo.write_final_report(run_id, report)
    training_run_repo.complete_run(run_id)

    return StageResult(
        stage_run_id=run_id,
        output_artifacts={"adapter": sha256_tree(training_run_repo.adapter_dir(run_id))},
    )


__all__ = ["run"]
