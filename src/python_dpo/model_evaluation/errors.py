"""The Stage 10 exception hierarchy (spec 10).

Mirrors the shape of :mod:`python_dpo.training.errors`: one base class, and a handful of
narrow subclasses for the failure modes the spec calls out explicitly. Every one of these
is a "stop the run" condition — spec sections 7, 15, 89, 121 are all explicit that a
contaminated benchmark, a broken adapter, or a mismatched candidate set must fail loudly
rather than degrade into a silently weaker comparison.
"""

from __future__ import annotations


class ModelEvaluationError(Exception):
    """Base class for every Stage 10 failure."""


class ModelEvaluationConfigError(ModelEvaluationError):
    """Raised when the evaluation configuration (python_eval.yaml) is invalid."""


class BenchmarkError(ModelEvaluationError):
    """Raised when a benchmark manifest is missing, malformed, or has drifted (§10, §91)."""


class BenchmarkLeakageError(ModelEvaluationError):
    """Raised when the benchmark overlaps a training/validation split (§5, §7).

    A contaminated benchmark must stop the run, not warn.
    """


class ModelIdentityError(ModelEvaluationError):
    """Raised when the base model used for DPO does not match Step 9's base model (§14)."""


class AdapterIntegrityError(ModelEvaluationError):
    """Raised when the DPO adapter cannot be verified or loaded (§15).

    Never falls back to the base model silently — §118 requires this to be an error.
    """


class PairingError(ModelEvaluationError):
    """Raised when base and DPO generations do not share identical prompts/seeds (§27, §114-116)."""


class IncompleteBenchmarkError(ModelEvaluationError):
    """Raised when base and DPO cover different problem sets without an explicit paired
    subset having been requested (§121).
    """


class EvaluationRunNotFoundError(ModelEvaluationError):
    """Raised when an evaluation run id has no corresponding directory."""


class EvaluationRunError(ModelEvaluationError):
    """Raised for evaluation-run-level problems (bad manifest, bad status transition)."""


class EvaluationStoreError(ModelEvaluationError):
    """Raised when a persisted Stage 10 artifact (generation/evaluation record) is malformed."""


__all__ = [
    "AdapterIntegrityError",
    "BenchmarkError",
    "BenchmarkLeakageError",
    "EvaluationRunError",
    "EvaluationRunNotFoundError",
    "EvaluationStoreError",
    "IncompleteBenchmarkError",
    "ModelEvaluationConfigError",
    "ModelEvaluationError",
    "ModelIdentityError",
    "PairingError",
]
