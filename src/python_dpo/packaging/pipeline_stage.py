"""Stage 12 packaging, assembled from the rest of :mod:`python_dpo.packaging` (spec 12
section 5, item 9).

Package the trained adapter -> verify it end to end through the sandbox -> register it as
``EXPERIMENTAL`` (spec section 48: packaging never registers anything at a higher trust
level). A verification failure fails the stage outright; there is no way to package
without verifying.

:func:`package_and_verify` is the one body both the pipeline stage adapter (:func:`run`,
below) and the standalone ``model package`` CLI command call -- one code path per stage,
driven by both entry points, matching how Stages 2-10's bodies moved into
``pipeline/stages/*`` (Stage 12 plan, Phase 2). Nothing here is called except through
those two entry points, both of which build every collaborator (the sandbox executor, the
adapter runner) explicitly, keeping this module itself torch-free at import time.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..evaluation import CandidateEvaluator, EvaluationRepository, PytestRunner, build_evaluation_sandbox_config
from ..model_evaluation.config import GenerationSettings, QuantizationSettings
from ..model_evaluation.runners import AdapterModelRunner
from ..pipeline.errors import StageFailedError
from ..pipeline.hashing import sha256_tree
from ..pipeline.stages._context import StageContext, StageResult
from ..sandbox import SandboxExecutor
from ..sandbox.config import SandboxConfig
from ..training.errors import TrainingRunNotFoundError
from ..training.run_repository import TrainingRunRepository
from .errors import PackagingError, VerificationError
from .package import ModelPackage, build_package
from .registry import ModelRegistry, RegistryEntry
from .verify import VerificationResult, verify_package

logger = logging.getLogger("python_dpo.packaging.pipeline_stage")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class PackageAndVerifyResult:
    package: ModelPackage
    verification: VerificationResult
    registry_entry: RegistryEntry


def package_and_verify(
    *,
    model_id: str,
    training_run_id: str,
    training_run_repo: TrainingRunRepository,
    dest_dir: Path,
    verification_dir: Path,
    sandbox_config: SandboxConfig,
    registry: ModelRegistry,
    experiment_run_id: str | None = None,
    seed: int = 42,
) -> PackageAndVerifyResult:
    """Package ``training_run_id``'s adapter into ``dest_dir``, verify it through the
    sandbox, and register it as ``EXPERIMENTAL``. Raises :class:`PackagingError` /
    :class:`VerificationError` on any failure -- there is no partial success: a package
    that fails verification is never registered.
    """
    try:
        training_manifest = training_run_repo.get_run(training_run_id)
    except TrainingRunNotFoundError as exc:
        raise PackagingError(str(exc)) from exc

    quantization_dict: dict[str, Any] = training_manifest.configuration.get("quantization", {}) or {}

    package = build_package(
        dest_dir=dest_dir,
        training_run_id=training_run_id,
        base_model_name=training_manifest.model_name,
        base_model_revision=training_manifest.model_revision,
        adapter_source=training_run_repo.adapter_dir(training_run_id),
        tokenizer_source=training_run_repo.tokenizer_dir(training_run_id),
        quantization=quantization_dict,
        created_at=_utc_now_iso(),
        experiment_run_id=experiment_run_id,
    )

    try:
        quantization = QuantizationSettings.from_mapping(quantization_dict)
    except Exception as exc:  # noqa: BLE001 - a malformed recorded config fails packaging
        raise PackagingError(f"could not resolve quantization settings: {exc}") from exc
    # Greedy, not sampled: verification asks a fixed trivial question and needs a
    # deterministic answer, not decoding-parameter noise (spec section 38's intent).
    generation = dataclasses.replace(GenerationSettings(), do_sample=False)

    def _generate(prompt: str) -> str:
        runner = AdapterModelRunner(
            model_name=package.base_model_name,
            model_revision=package.base_model_revision,
            adapter_dir=package.adapter_dir,
            quantization=quantization,
            generation=generation,
        )
        runner.ensure_loaded()
        try:
            return runner.generate(prompt, seed=seed).text
        finally:
            runner.unload()

    evaluator = CandidateEvaluator(
        runner=PytestRunner(SandboxExecutor(config=sandbox_config)),
        repository=EvaluationRepository(verification_dir),
    )
    verification = verify_package(
        package_id=model_id,
        model_name=package.base_model_name,
        generate=_generate,
        evaluator=evaluator,
        evaluation_run_id=f"{model_id}_verify",
        seed=seed,
    )

    entry = registry.register(
        RegistryEntry(
            model_id=model_id,
            status="EXPERIMENTAL",
            package_path=str(dest_dir),
            base_model_name=package.base_model_name,
            base_model_revision=package.base_model_revision,
            training_run_id=training_run_id,
            experiment_run_id=experiment_run_id,
            created_at=package.created_at,
            verification=verification.to_dict(),
        )
    )
    logger.info("Model %s packaged and registered EXPERIMENTAL", model_id)
    return PackageAndVerifyResult(package=package, verification=verification, registry_entry=entry)


def run(context: StageContext) -> StageResult:
    config = context.project_config
    experiment_run_id = context.experiment_run_id

    training_run_id = context.upstream_run_id("dpo_training")
    training_run_repo = TrainingRunRepository(config.paths.training / "runs")
    dest_dir = context.experiment_repo.model_dir(experiment_run_id)
    verification_dir = context.experiment_repo.stage_dir(experiment_run_id, "packaging") / "verification"
    sandbox_config = build_evaluation_sandbox_config(config.sandbox, config.evaluation)
    registry = ModelRegistry(config.project_root / "models" / "registry.json")

    try:
        package_and_verify(
            model_id=experiment_run_id,
            training_run_id=training_run_id,
            training_run_repo=training_run_repo,
            dest_dir=dest_dir,
            verification_dir=verification_dir,
            sandbox_config=sandbox_config,
            registry=registry,
            experiment_run_id=experiment_run_id,
            seed=context.seed,
        )
    except (PackagingError, VerificationError) as exc:
        raise StageFailedError(f"packaging failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - never crash the orchestrator; always FAIL
        raise StageFailedError(f"packaging failed ({type(exc).__name__}): {exc}") from exc

    return StageResult(
        stage_run_id=f"{experiment_run_id}_packaging",
        output_artifacts={"packaging": sha256_tree(dest_dir)},
    )


__all__ = ["PackageAndVerifyResult", "package_and_verify", "run"]
