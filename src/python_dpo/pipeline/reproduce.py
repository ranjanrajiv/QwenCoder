"""Reproducing an experiment (spec 12 sections 71, 72).

:func:`render_reproduce_commands` prints the command to recreate an experiment from its
own frozen ``resolved_config.yaml`` (spec section 10 -- that file never changes after the
run starts, so it is always the authoritative recreation source, not whatever the original
``--config`` source file has since become).

:func:`verify_reproducibility` (``--verify-only``) diffs the recorded manifest against
*only what is actually persisted*: the configured model (via the ``dpo_training`` stage's
own training-run manifest), the problem dataset's content hash (via the
``problem_dataset`` stage's recorded output hash), the resolved-config hash (if a current
``--config`` source is given to recompute it from), and the captured environment. It never
invents a comparison for data nothing recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .repository import ExperimentRunRepository

# The environment.json keys whose drift actually threatens reproducibility -- not every
# captured field (pytest version, for instance, does not affect a trained model).
_ENVIRONMENT_KEYS_TO_COMPARE = (
    "python_version",
    "torch_version",
    "transformers_version",
    "peft_version",
    "trl_version",
    "bitsandbytes_version",
    "accelerate_version",
    "datasets_version",
    "cuda_version",
    "os",
)


def render_reproduce_commands(experiment_run_id: str, resolved_config_path: Path) -> str:
    return (
        f"# Recreate experiment {experiment_run_id}\n\n"
        f"python -m python_dpo experiment run --config {resolved_config_path}\n"
    )


@dataclass(frozen=True)
class ReproducibilityReport:
    """``None`` for any field means "nothing was recorded to check this against",
    never "it matched"."""

    experiment_run_id: str
    config_hash_matches: bool | None = None
    model_matches: bool | None = None
    dataset_hash_matches: bool | None = None
    environment_diffs: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def reproducible(self) -> bool:
        return (
            self.config_hash_matches is not False
            and self.model_matches is not False
            and self.dataset_hash_matches is not False
            and not self.environment_diffs
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_run_id": self.experiment_run_id,
            "reproducible": self.reproducible,
            "config_hash_matches": self.config_hash_matches,
            "model_matches": self.model_matches,
            "dataset_hash_matches": self.dataset_hash_matches,
            "environment_diffs": {
                key: {"recorded": recorded, "current": current}
                for key, (recorded, current) in self.environment_diffs.items()
            },
        }


def verify_reproducibility(
    repo: ExperimentRunRepository,
    experiment_run_id: str,
    project_config: Any,
    *,
    config_path: Path | None = None,
) -> ReproducibilityReport:
    manifest = repo.get_run(experiment_run_id)

    config_hash_matches: bool | None = None
    if config_path is not None:
        from .config import ExperimentConfig, ExperimentConfigError
        from .hashing import config_hash as _config_hash

        try:
            reloaded = ExperimentConfig.load(Path(config_path))
            config_hash_matches = _config_hash(reloaded.to_dict()) == manifest.configuration_hash
        except ExperimentConfigError:
            config_hash_matches = False

    model_matches: bool | None = None
    training_stage = repo.read_stage_manifest(experiment_run_id, "dpo_training")
    if training_stage is not None and training_stage.status == "COMPLETED":
        from ..training.errors import TrainingRunNotFoundError
        from ..training.run_repository import TrainingRunRepository

        training_repo = TrainingRunRepository(project_config.paths.training / "runs")
        try:
            training_manifest = training_repo.get_run(training_stage.stage_run_id)
            model_matches = (
                training_manifest.model_name == project_config.model.name
                and training_manifest.model_revision == project_config.model.revision
            )
        except TrainingRunNotFoundError:
            model_matches = False

    dataset_hash_matches: bool | None = None
    dataset_stage = repo.read_stage_manifest(experiment_run_id, "problem_dataset")
    if dataset_stage is not None and dataset_stage.status == "COMPLETED":
        recorded_hash = dataset_stage.output_artifacts.get("problem_dataset")
        if recorded_hash is not None:
            from ..problems import dataset_path
            from .hashing import sha256_file

            current_path = dataset_path(project_config.paths.problems)
            dataset_hash_matches = (
                current_path.is_file() and sha256_file(current_path) == recorded_hash
            )

    recorded_env = repo.read_environment(experiment_run_id) or {}
    from .environment import capture_environment

    current_env = capture_environment()
    environment_diffs = {
        key: (recorded_env.get(key), current_env.get(key))
        for key in _ENVIRONMENT_KEYS_TO_COMPARE
        if recorded_env.get(key) != current_env.get(key)
    }

    return ReproducibilityReport(
        experiment_run_id=experiment_run_id,
        config_hash_matches=config_hash_matches,
        model_matches=model_matches,
        dataset_hash_matches=dataset_hash_matches,
        environment_diffs=environment_diffs,
    )


def format_reproducibility_report(report: ReproducibilityReport) -> str:
    def _fmt(value: bool | None) -> str:
        return {True: "MATCH", False: "MISMATCH", None: "not checked"}[value]

    lines = [
        f"Experiment {report.experiment_run_id}: "
        f"{'REPRODUCIBLE' if report.reproducible else 'NOT REPRODUCIBLE'}",
        f"  config hash:   {_fmt(report.config_hash_matches)}",
        f"  model:         {_fmt(report.model_matches)}",
        f"  dataset hash:  {_fmt(report.dataset_hash_matches)}",
    ]
    if report.environment_diffs:
        lines.append("  environment:   MISMATCH")
        for key, (recorded, current) in sorted(report.environment_diffs.items()):
            lines.append(f"    {key}: recorded={recorded!r} current={current!r}")
    else:
        lines.append("  environment:   MATCH")
    return "\n".join(lines) + "\n"


__all__ = [
    "ReproducibilityReport",
    "format_reproducibility_report",
    "render_reproduce_commands",
    "verify_reproducibility",
]
