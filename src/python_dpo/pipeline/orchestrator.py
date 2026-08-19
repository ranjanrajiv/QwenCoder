"""``PipelineOrchestrator`` (spec 12 section 7): resolve, validate, execute, persist.

The orchestrator adds a layer over the six existing per-stage run repositories; it does
not replace them (plan decision 2). Every stage's actual output still lives under its own
canonical ``data/<stage>/runs/<run_id>/`` directory. What this module owns is the
experiment-level bookkeeping: which stages run in which order, whether a stage's cached
output can be reused, what happens when one fails, and the final manifest tying it all
together.
"""

from __future__ import annotations

import dataclasses
import logging
import signal
from typing import Any

from .. import __version__
from .cache import cache_key, is_reusable
from .config import ExperimentConfig
from .errors import DependencyError, PipelineError, StageFailedError, StageNotImplementedError
from .gitinfo import capture_git_info, enforce_dirty_policy
from .hashing import config_hash
from .lineage import build_lineage
from .manifest import ExperimentManifest, ManifestError, StageError, StageManifest, utc_now_iso
from .repository import ExperimentRunRepository
from .stages import STAGE_NAMES, get_stage, resolve_adapter, topological_order, validate_graph
from .stages._context import StageContext

logger = logging.getLogger("python_dpo.pipeline.orchestrator")


def _model_version(project_config: Any) -> str:
    revision = project_config.model.revision or "default"
    return f"{project_config.model.name}@{revision}"


class PipelineOrchestrator:
    def __init__(self, project_config: Any, experiment_repo: ExperimentRunRepository | None = None):
        self.project_config = project_config
        self.repo = experiment_repo or ExperimentRunRepository(project_config.paths.experiments / "runs")

    # ------------------------------------------------------------------------- planning

    def enabled_order(self, experiment: ExperimentConfig) -> tuple[str, ...]:
        """The stages that would execute, in order -- section 5's numbering, restricted
        to what this experiment enables. Used by both ``--dry-run`` (section 23) and
        ``experiment graph`` (section 62)."""
        validate_graph()
        return tuple(name for name in topological_order() if experiment.stage(name).enabled)

    # -------------------------------------------------------------------------- running

    def run(
        self,
        experiment: ExperimentConfig | None = None,
        *,
        force: str | None = None,
        resume_run_id: str | None = None,
    ) -> ExperimentManifest:
        """Execute (or resume) one experiment end to end.

        ``experiment`` is required unless ``resume_run_id`` is given, in which case it is
        ignored: the authoritative configuration for a resumed run is always the
        immutable ``resolved_config.yaml`` written when it first started (spec section
        10), never whatever the caller happens to have loaded from a possibly-edited
        source file.
        """
        validate_graph()
        git_info = capture_git_info(self.project_config.project_root)

        if resume_run_id is not None:
            experiment_run_id, experiment, manifest = self._resume(resume_run_id)
            enforce_dirty_policy(git_info, on_dirty=experiment.on_dirty)
        else:
            if experiment is None:
                raise PipelineError("run() requires an experiment configuration unless resuming")
            # Checked before anything is written to disk: a fail-on-dirty policy must
            # prevent the run from being created at all, not create it and fail after.
            enforce_dirty_policy(git_info, on_dirty=experiment.on_dirty)
            experiment_run_id, manifest = self._start(experiment)

        manifest = dataclasses.replace(manifest, git_commit=git_info)
        self.repo.save(manifest)

        force_targets: set[str] = set()
        if force:
            from .cache import invalidate

            force_targets = set(invalidate(force, cascade=True))

        order = topological_order()
        # `upstream` is populated incrementally as the loop below processes stages in
        # topological order -- a resumed stage's own read-and-compare-cache-key logic in
        # `_run_one_stage` already recovers an already-COMPLETED stage from disk, so
        # nothing needs to be pre-seeded here before the loop starts.
        upstream: dict[str, StageManifest] = {}

        failed_stage: str | None = None
        interrupted = False

        with _signal_guard():
            for name in order:
                try:
                    result = self._run_one_stage(
                        experiment_run_id, experiment, name, upstream, force_targets
                    )
                except KeyboardInterrupt:
                    self._mark_cancelled(experiment_run_id, name)
                    interrupted = True
                    break
                except (StageFailedError, StageNotImplementedError, DependencyError) as exc:
                    self._mark_failed(experiment_run_id, name, exc)
                    failed_stage = name
                    break
                if result is not None:
                    upstream[name] = result

        if interrupted:
            self.repo.interrupt_run(experiment_run_id)
            return self.repo.get_run(experiment_run_id)

        if failed_stage is not None:
            self._block_remaining(experiment_run_id, experiment, order, upstream, failed_stage)
            self.repo.fail_run(experiment_run_id)
            return self.repo.get_run(experiment_run_id)

        self._finalize(experiment_run_id, upstream)
        return self.repo.get_run(experiment_run_id)

    # -------------------------------------------------------------------------- helpers

    def _start(self, experiment: ExperimentConfig) -> tuple[str, ExperimentManifest]:
        manifest = self.repo.create_run(
            experiment_name=experiment.name,
            configuration_hash=config_hash(experiment.to_dict()),
        )
        experiment_run_id = manifest.experiment_run_id
        self.repo.write_resolved_config(experiment_run_id, experiment.to_dict())

        from .environment import capture_environment

        self.repo.write_environment(experiment_run_id, capture_environment())
        manifest = self.repo.start_run(experiment_run_id)
        logger.info("Experiment %s created | %s", experiment_run_id, experiment.name)
        return experiment_run_id, manifest

    def _resume(self, resume_run_id: str) -> tuple[str, ExperimentConfig, ExperimentManifest]:
        manifest = self.repo.get_run(resume_run_id)
        if manifest.status not in ("created", "running", "failed", "interrupted"):
            raise PipelineError(
                f"experiment {resume_run_id!r} is {manifest.status!r} and cannot be resumed"
            )
        resolved = self.repo.read_resolved_config(resume_run_id)
        if resolved is None:
            raise PipelineError(f"no resolved_config.yaml for {resume_run_id!r}; cannot resume")
        experiment = ExperimentConfig.from_mapping(resolved)
        if manifest.status != "running":
            manifest = self.repo.start_run(resume_run_id)
        logger.info("Resuming experiment %s", resume_run_id)
        return resume_run_id, experiment, manifest

    def _run_one_stage(
        self,
        experiment_run_id: str,
        experiment: ExperimentConfig,
        name: str,
        upstream: dict[str, StageManifest],
        force_targets: set[str],
    ) -> StageManifest | None:
        spec = get_stage(name)
        stage_config = experiment.stage(name)

        if not stage_config.enabled:
            self.repo.write_stage_manifest(
                experiment_run_id,
                StageManifest(
                    stage_name=name,
                    stage_run_id=f"{experiment_run_id}_{name}",
                    status="SKIPPED",
                    code_version=__version__,
                ),
            )
            return None

        missing_deps = [dep for dep in spec.requires if dep not in upstream]
        if missing_deps:
            raise DependencyError(
                f"stage {name!r} requires completed output from {', '.join(missing_deps)}, "
                "which is not available. Run the missing stage(s) first; nothing is "
                "reconstructed silently (spec section 16)"
            )

        input_hashes: dict[str, str] = {}
        for dep in spec.requires:
            input_hashes.update(upstream[dep].output_artifacts)

        configuration_hash = config_hash(stage_config.to_dict())
        key = cache_key(
            stage=name,
            input_hashes=input_hashes,
            configuration_hash=configuration_hash,
            code_version=__version__,
            model_version=_model_version(self.project_config),
        )

        if name not in force_targets:
            existing = self.repo.read_stage_manifest(experiment_run_id, name)
            if is_reusable(existing, key):
                logger.info("Stage %s reused (already completed this run, cache hit)", name)
                return existing

            reused = self.repo.find_reusable_stage(name, key, exclude_run_id=experiment_run_id)
            if reused is not None:
                logger.info("Stage %s reused from experiment run's cache hit", name)
                manifest = StageManifest(
                    stage_name=name,
                    stage_run_id=reused.stage_run_id,
                    status="COMPLETED",
                    code_version=__version__,
                    start_time=utc_now_iso(),
                    end_time=utc_now_iso(),
                    input_artifacts=input_hashes,
                    output_artifacts=reused.output_artifacts,
                    configuration_hash=configuration_hash,
                    cache_key=key,
                    reused=True,
                )
                self.repo.write_stage_manifest(experiment_run_id, manifest)
                return manifest

        running = StageManifest(
            stage_name=name,
            stage_run_id=f"{experiment_run_id}_{name}",
            status="RUNNING",
            code_version=__version__,
            start_time=utc_now_iso(),
            input_artifacts=input_hashes,
            configuration_hash=configuration_hash,
            cache_key=key,
        )
        self.repo.write_stage_manifest(experiment_run_id, running)

        context = StageContext(
            experiment_run_id=experiment_run_id,
            stage_config=stage_config,
            project_config=self.project_config,
            experiment_repo=self.repo,
            upstream=upstream,
            seed=experiment.seed,
        )
        adapter = resolve_adapter(spec)
        logger.info("Stage %s starting", name)
        result = adapter(context)
        logger.info("Stage %s completed", name)

        completed = StageManifest(
            stage_name=name,
            stage_run_id=result.stage_run_id,
            status="COMPLETED",
            code_version=__version__,
            start_time=running.start_time,
            end_time=utc_now_iso(),
            input_artifacts=input_hashes,
            output_artifacts=result.output_artifacts,
            configuration_hash=configuration_hash,
            cache_key=key,
            reused=False,
        )
        self.repo.write_stage_manifest(experiment_run_id, completed)
        return completed

    def _mark_failed(self, experiment_run_id: str, name: str, exc: Exception) -> None:
        existing = self.repo.read_stage_manifest(experiment_run_id, name)
        error = StageError(
            stage=name, error_type=type(exc).__name__, message=str(exc), timestamp=utc_now_iso()
        )
        if existing is not None and existing.status == "RUNNING":
            failed = existing.with_status("FAILED", end_time=utc_now_iso(), error=error)
        else:
            failed = StageManifest(
                stage_name=name,
                stage_run_id=f"{experiment_run_id}_{name}",
                status="FAILED",
                code_version=__version__,
                end_time=utc_now_iso(),
                error=error,
            )
        self.repo.write_stage_manifest(experiment_run_id, failed)
        logger.error("Stage %s failed: %s", name, exc)

    def _mark_cancelled(self, experiment_run_id: str, name: str) -> None:
        """Section 67: the in-flight stage is marked CANCELLED, not left at RUNNING."""
        existing = self.repo.read_stage_manifest(experiment_run_id, name)
        if existing is not None and existing.status == "RUNNING":
            cancelled = existing.with_status("CANCELLED", end_time=utc_now_iso())
        else:
            cancelled = StageManifest(
                stage_name=name,
                stage_run_id=f"{experiment_run_id}_{name}",
                status="CANCELLED",
                code_version=__version__,
                end_time=utc_now_iso(),
            )
        self.repo.write_stage_manifest(experiment_run_id, cancelled)
        logger.warning("Stage %s cancelled (interrupted)", name)

    def _block_remaining(
        self,
        experiment_run_id: str,
        experiment: ExperimentConfig,
        order: tuple[str, ...],
        upstream: dict[str, StageManifest],
        failed_stage: str,
    ) -> None:
        """Section 66: downstream stages become BLOCKED, never executed against
        incomplete artifacts."""
        for name in order:
            if name == failed_stage or name in upstream:
                continue
            if not experiment.stage(name).enabled:
                continue
            self.repo.write_stage_manifest(
                experiment_run_id,
                StageManifest(
                    stage_name=name,
                    stage_run_id=f"{experiment_run_id}_{name}",
                    status="BLOCKED",
                    code_version=__version__,
                ),
            )

    def _finalize(self, experiment_run_id: str, upstream: dict[str, StageManifest]) -> None:
        from .artifacts import ArtifactError, make_artifact_ref, write_artifact_manifest

        run_dir = self.repo.run_dir(experiment_run_id)
        refs = {}
        for name, stage_manifest in upstream.items():
            if name == "packaging":
                # The one deliberate copy in the whole layout (plan decision 2): the
                # packaged model lives inside the experiment's own tree, not a stage's
                # canonical `data/<stage>/runs/` store.
                path = self.repo.model_dir(experiment_run_id)
            else:
                path = _artifact_path(self.project_config, name, stage_manifest.stage_run_id)
            if path is None:
                continue
            try:
                refs[name] = make_artifact_ref(
                    name, path, relative_to=self.project_config.project_root
                )
            except ArtifactError:
                continue
        write_artifact_manifest(run_dir / "artifacts.json", refs)

        self.repo.write_lineage(experiment_run_id, build_lineage(upstream))

        manifest = self.repo.get_run(experiment_run_id)
        stage_runs = {
            name: _summary(stage_manifest) for name, stage_manifest in upstream.items()
        }
        manifest = dataclasses.replace(manifest, stage_runs=stage_runs)
        self.repo.save(manifest)

        # Complete the run before building reports -- experiment_summary.md must show the
        # real final status and end_time, not a "running" snapshot taken before the
        # transition (spec section 50).
        manifest = self.repo.complete_run(experiment_run_id)

        from .report import write_experiment_reports

        write_experiment_reports(
            self.repo, experiment_run_id, manifest, upstream, self.project_config
        )


def _artifact_path(project_config: Any, stage_name: str, stage_run_id: str):
    """Where each stage's canonical output actually lives on disk, for ``artifacts.json``
    (spec sections 69, 70). Mirrors exactly what each stage adapter hashed."""
    from ..problems import dataset_path

    paths = project_config.paths
    builders: dict[str, Any] = {
        "problem_dataset": lambda: dataset_path(paths.problems),
        "candidate_generation": lambda: paths.candidates / "runs" / stage_run_id,
        "candidate_execution": lambda: paths.evaluations / "runs" / stage_run_id,
        "candidate_evaluation": lambda: paths.rankings / "runs" / stage_run_id,
        "preference_generation": lambda: paths.preferences / "runs" / stage_run_id,
        "dpo_training": lambda: paths.training / "runs" / stage_run_id / "adapter",
        "model_evaluation": lambda: paths.model_evaluations / "runs" / stage_run_id,
        "error_analysis": lambda: paths.analysis / "runs" / stage_run_id,
    }
    builder = builders.get(stage_name)
    return builder() if builder else None


def _summary(stage_manifest: StageManifest):
    from .manifest import StageRunSummary

    return StageRunSummary(
        status=stage_manifest.status,
        stage_run_id=stage_manifest.stage_run_id,
        reused=stage_manifest.reused,
    )


class _signal_guard:
    """Section 67: SIGINT already raises ``KeyboardInterrupt`` via Python's default
    handler; this makes SIGTERM raise the same exception, so both signals take one path
    -- through whichever ``except KeyboardInterrupt`` handling the current stage adapter
    already has (persisting *its own* underlying run's state and flushing logs), and then
    up to :meth:`PipelineOrchestrator.run`'s own catch, which persists the experiment-level
    interruption and preserves every already-completed stage's artifacts untouched.
    """

    def __enter__(self) -> "_signal_guard":
        self._previous_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle)
        return self

    def _handle(self, signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    def __exit__(self, *exc_info: object) -> None:
        signal.signal(signal.SIGTERM, self._previous_term)


__all__ = ["PipelineOrchestrator"]
