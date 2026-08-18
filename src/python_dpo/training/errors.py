"""The training exception hierarchy (spec 09).

Unlike every earlier stage, this one depends on heavyweight optional packages (torch,
transformers, trl, peft, bitsandbytes) and on physical hardware. Its failure modes divide
into three kinds, and the hierarchy keeps them distinguishable:

* **Environment** — :class:`TrainingDependencyError`, :class:`HardwareError`: the machine
  or the install cannot support training at all.
* **Preflight** — :class:`DatasetValidationError`, :class:`TargetModuleError`,
  :class:`TruncationThresholdError`, :class:`FullFineTuneError`: the *configuration* is
  wrong, caught before any artifact is written (spec sections 6, 16, 20, 24, 36).
* **Run** — :class:`TrainingRunNotFoundError`, :class:`TrainingStoreError`,
  :class:`CheckpointCompatibilityError`, :class:`AdapterReloadError`: the machinery of a
  particular training run.

:class:`FullFineTuneError` deserves special mention: it is the spec section 20 safety trip
that fires when ``trainable_parameters == total_parameters``. Reaching it means the LoRA
configuration silently failed to freeze the base model, which would burn hours of GPU time
producing an artifact this stage explicitly must not produce. It is an error, never a
warning.
"""

from __future__ import annotations


class TrainingError(Exception):
    """Base class for every training failure."""


class TrainingDependencyError(TrainingError):
    """Raised when the optional training backend is not installed.

    Deliberately not ``ImportError``: the caller wants one actionable message naming the
    extra to install, not a traceback through a deferred import.
    """


class TrainingConfigError(TrainingError):
    """Raised when an experiment configuration file is invalid."""


class HardwareError(TrainingError):
    """Raised when the available hardware cannot support the requested configuration."""


class DatasetValidationError(TrainingError):
    """Raised when the preference dataset fails spec section 24's checks.

    Per section 24, this is raised *before* the model is loaded — there is no point
    spending minutes loading 4-bit weights for a dataset that was never trainable.
    """


class TargetModuleError(TrainingError):
    """Raised when no configured LoRA target module exists on the model (section 16).

    Training with zero LoRA targets would produce an adapter that changes nothing while
    reporting success, so this is fatal rather than a warning.
    """


class FullFineTuneError(TrainingError):
    """Raised when every parameter is trainable under a QLoRA configuration (section 20).

    The base model must stay frozen (section 3). If the trainable count equals the total,
    LoRA did not apply and this would be a full fine-tune — which Step 9 must never
    perform accidentally.
    """


class TruncationThresholdError(TrainingError):
    """Raised when too many examples would be truncated (sections 35, 36)."""


class TrainingRunNotFoundError(TrainingError):
    """Raised when a training run id has no corresponding directory."""


class TrainingRunError(TrainingError):
    """Raised for training-run-level problems (bad manifest, bad status transition)."""


class TrainingStoreError(TrainingError):
    """Raised when a persisted training file is malformed."""


class CheckpointCompatibilityError(TrainingError):
    """Raised when a checkpoint cannot be resumed against the current inputs (91, 92)."""


class AdapterReloadError(TrainingError):
    """Raised when a saved adapter cannot be reloaded and generated from (section 74).

    Section 82 makes this decisive: an adapter that does not reload is not a final
    artifact, and the run is not marked successful.
    """


__all__ = [
    "AdapterReloadError",
    "CheckpointCompatibilityError",
    "DatasetValidationError",
    "FullFineTuneError",
    "HardwareError",
    "TargetModuleError",
    "TrainingConfigError",
    "TrainingDependencyError",
    "TrainingError",
    "TrainingRunError",
    "TrainingRunNotFoundError",
    "TrainingStoreError",
    "TruncationThresholdError",
]
