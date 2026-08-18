"""The training experiment configuration (spec 09 sections 62, 63).

Deliberately a **standalone YAML file** (``configs/training/dpo_qlora.yaml``), not a
section of the root ``config.yaml``, for two reasons. The spec asks for it (section 62);
and these are the values an experimenter iterates on — learning rate, beta, LoRA rank —
which do not belong in the pipeline's own configuration. A second experiment is a second
file, not an edit.

It also sidesteps a real obstacle: ``python_dpo.models.base.ModelConfig`` actively
*rejects* a non-null ``quantization`` ("not implemented in Stage 3"), so the root config's
``model:`` section cannot express 4-bit NF4 at all.

Nothing here imports torch. The dtype and optimizer names are validated as strings against
closed sets and only resolved to real objects inside :mod:`python_dpo.training.loader`.

**Two fields are deliberately not passed through to TRL.** ``max_prompt_length`` and
``warmup_ratio`` are mandated by spec sections 33 and 53, but TRL 1.10's ``DPOConfig``
has neither: prompt and completion are truncated together against ``max_length``, and
warmup is expressed as ``warmup_steps``. Both are kept here — ``max_prompt_length`` drives
this stage's own length analysis (sections 34, 35) and ``warmup_ratio`` is converted to a
step count once the schedule length is known. See ``trainer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import TrainingConfigError

QUANT_TYPES = frozenset({"nf4", "fp4"})
COMPUTE_DTYPES = frozenset({"bfloat16", "float16", "float32", "auto"})
LORA_BIASES = frozenset({"none", "all", "lora_only"})
TASK_TYPES = frozenset({"CAUSAL_LM"})
DPO_LOSS_TYPES = frozenset({"sigmoid"})
SCHEDULERS = frozenset({"cosine", "linear", "constant", "constant_with_warmup"})
EVAL_STRATEGIES = frozenset({"no", "steps", "epoch"})

# Spec section 50: the memory-efficient preference, with a recorded fallback.
DEFAULT_OPTIMIZER = "paged_adamw_8bit"
FALLBACK_OPTIMIZER = "adamw_torch"

DEFAULT_CONFIG_PATH = Path("configs/training/dpo_qlora.yaml")

_TOP_LEVEL_KEYS = frozenset(
    {"experiment_name", "model", "quantization", "lora", "dpo", "training", "optimizer",
     "dataset", "distributed"}
)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingConfigError(f"{label} must be an integer of 1 or greater")
    return value


def _require_nonneg_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingConfigError(f"{label} must be an integer of 0 or greater")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(f"{label} must be a number")
    return float(value)


def _require_flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingConfigError(f"{label} must be true or false")
    return value


def _reject_unknown(data: Any, allowed: frozenset[str], label: str) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TrainingConfigError(f"{label} must be a mapping")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise TrainingConfigError(f"{label}: unknown key(s): {', '.join(unknown)}")
    return dict(data)


# ------------------------------------------------------------------------------- sections


@dataclass(frozen=True)
class ModelSpec:
    """Which base model to train against (spec sections 5, 7)."""

    name: str
    revision: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "model.name")
        if self.revision is not None:
            _require_text(self.revision, "model.revision")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "revision": self.revision}

    @classmethod
    def from_mapping(cls, data: Any) -> ModelSpec:
        raw = _reject_unknown(data, frozenset({"name", "revision"}), "model")
        if "name" not in raw:
            raise TrainingConfigError("model.name is required")
        return cls(name=raw["name"], revision=raw.get("revision"))


@dataclass(frozen=True)
class QuantizationSettings:
    """The spec section 8/9 4-bit configuration, every option configurable."""

    enabled: bool = True
    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        _require_flag(self.enabled, "quantization.enabled")
        _require_flag(self.double_quant, "quantization.double_quant")
        if self.bits not in (4, 8):
            raise TrainingConfigError("quantization.bits must be 4 or 8")
        if self.quant_type not in QUANT_TYPES:
            raise TrainingConfigError(
                f"quantization.quant_type must be one of {', '.join(sorted(QUANT_TYPES))}"
            )
        if self.compute_dtype not in COMPUTE_DTYPES:
            raise TrainingConfigError(
                f"quantization.compute_dtype must be one of {', '.join(sorted(COMPUTE_DTYPES))}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "bits": self.bits,
            "quant_type": self.quant_type,
            "double_quant": self.double_quant,
            "compute_dtype": self.compute_dtype,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> QuantizationSettings:
        raw = _reject_unknown(
            data,
            frozenset({"enabled", "bits", "quant_type", "double_quant", "compute_dtype"}),
            "quantization",
        )
        return cls(**raw) if raw else cls()


@dataclass(frozen=True)
class LoraSettings:
    """PEFT LoRA configuration (spec sections 14-18)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    task_type: str = "CAUSAL_LM"

    def __post_init__(self) -> None:
        _require_positive_int(self.r, "lora.r")
        _require_positive_int(self.alpha, "lora.alpha")
        dropout = _require_number(self.dropout, "lora.dropout")
        if not 0.0 <= dropout < 1.0:
            raise TrainingConfigError("lora.dropout must be in [0, 1)")
        if self.bias not in LORA_BIASES:
            raise TrainingConfigError(
                f"lora.bias must be one of {', '.join(sorted(LORA_BIASES))}"
            )
        if self.task_type not in TASK_TYPES:
            raise TrainingConfigError(
                f"lora.task_type must be one of {', '.join(sorted(TASK_TYPES))}"
            )
        if isinstance(self.target_modules, (str, bytes)) or not isinstance(
            self.target_modules, (list, tuple)
        ):
            raise TrainingConfigError("lora.target_modules must be a list of module names")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if not self.target_modules:
            # Spec section 16: never train with zero LoRA targets. Catching an empty list
            # here means the model never even loads for a configuration that cannot work.
            raise TrainingConfigError("lora.target_modules must not be empty")
        for name in self.target_modules:
            _require_text(name, "lora.target_modules entry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "target_modules": list(self.target_modules),
            "task_type": self.task_type,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> LoraSettings:
        raw = _reject_unknown(
            data,
            frozenset({"r", "alpha", "dropout", "bias", "target_modules", "task_type"}),
            "lora",
        )
        return cls(**raw) if raw else cls()


@dataclass(frozen=True)
class DpoSettings:
    """DPO objective settings (spec sections 44, 45)."""

    beta: float = 0.1
    loss_type: str = "sigmoid"

    def __post_init__(self) -> None:
        beta = _require_number(self.beta, "dpo.beta")
        if beta <= 0:
            raise TrainingConfigError("dpo.beta must be greater than 0")
        if self.loss_type not in DPO_LOSS_TYPES:
            # Spec section 45: establish the plain sigmoid baseline first. Alternative
            # losses are a later experiment, not a Step 9 option.
            raise TrainingConfigError(
                f"dpo.loss_type must be one of {', '.join(sorted(DPO_LOSS_TYPES))}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"beta": self.beta, "loss_type": self.loss_type}

    @classmethod
    def from_mapping(cls, data: Any) -> DpoSettings:
        raw = _reject_unknown(data, frozenset({"beta", "loss_type"}), "dpo")
        return cls(**raw) if raw else cls()


_TRAINING_KEYS = frozenset(
    {
        "max_length", "max_prompt_length", "learning_rate", "num_train_epochs", "max_steps",
        "per_device_train_batch_size", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "gradient_checkpointing", "warmup_ratio",
        "max_grad_norm", "lr_scheduler_type", "seed", "data_seed", "save_steps",
        "eval_steps", "logging_steps", "eval_strategy", "max_truncation_rate",
        "min_training_pairs", "apply_chat_template",
    }
)


@dataclass(frozen=True)
class TrainingSettings:
    """Optimization and sequence settings (spec sections 33-43, 48-60)."""

    max_length: int = 1024
    max_prompt_length: int = 512
    learning_rate: float = 1.0e-5
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    data_seed: int = 42
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 1
    eval_strategy: str = "steps"
    max_truncation_rate: float = 0.05
    min_training_pairs: int = 500
    apply_chat_template: bool = True

    def __post_init__(self) -> None:
        _require_positive_int(self.max_length, "training.max_length")
        _require_positive_int(self.max_prompt_length, "training.max_prompt_length")
        _require_positive_int(self.num_train_epochs, "training.num_train_epochs")
        _require_positive_int(
            self.per_device_train_batch_size, "training.per_device_train_batch_size"
        )
        _require_positive_int(
            self.per_device_eval_batch_size, "training.per_device_eval_batch_size"
        )
        _require_positive_int(
            self.gradient_accumulation_steps, "training.gradient_accumulation_steps"
        )
        _require_positive_int(self.save_steps, "training.save_steps")
        _require_positive_int(self.eval_steps, "training.eval_steps")
        _require_positive_int(self.logging_steps, "training.logging_steps")
        _require_nonneg_int(self.seed, "training.seed")
        _require_nonneg_int(self.data_seed, "training.data_seed")
        _require_nonneg_int(self.min_training_pairs, "training.min_training_pairs")
        _require_flag(self.gradient_checkpointing, "training.gradient_checkpointing")
        _require_flag(self.apply_chat_template, "training.apply_chat_template")

        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise TrainingConfigError("training.max_steps must be an integer")
        if self.max_steps == 0 or self.max_steps < -1:
            raise TrainingConfigError(
                "training.max_steps must be -1 (use epochs) or a positive step count"
            )

        lr = _require_number(self.learning_rate, "training.learning_rate")
        if lr <= 0:
            raise TrainingConfigError("training.learning_rate must be greater than 0")
        warmup = _require_number(self.warmup_ratio, "training.warmup_ratio")
        if not 0.0 <= warmup < 1.0:
            raise TrainingConfigError("training.warmup_ratio must be in [0, 1)")
        grad_norm = _require_number(self.max_grad_norm, "training.max_grad_norm")
        if grad_norm <= 0:
            raise TrainingConfigError("training.max_grad_norm must be greater than 0")
        rate = _require_number(self.max_truncation_rate, "training.max_truncation_rate")
        if not 0.0 <= rate <= 1.0:
            raise TrainingConfigError("training.max_truncation_rate must be in [0, 1]")

        if self.lr_scheduler_type not in SCHEDULERS:
            raise TrainingConfigError(
                f"training.lr_scheduler_type must be one of {', '.join(sorted(SCHEDULERS))}"
            )
        if self.eval_strategy not in EVAL_STRATEGIES:
            raise TrainingConfigError(
                f"training.eval_strategy must be one of {', '.join(sorted(EVAL_STRATEGIES))}"
            )
        if self.max_prompt_length >= self.max_length:
            raise TrainingConfigError(
                "training.max_prompt_length must be less than training.max_length; "
                "otherwise no room remains for the response"
            )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(_TRAINING_KEYS)}

    @classmethod
    def from_mapping(cls, data: Any) -> TrainingSettings:
        raw = _reject_unknown(data, _TRAINING_KEYS, "training")
        return cls(**raw) if raw else cls()


@dataclass(frozen=True)
class OptimizerSettings:
    """Optimizer selection (spec section 50)."""

    name: str = DEFAULT_OPTIMIZER

    def __post_init__(self) -> None:
        _require_text(self.name, "optimizer.name")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}

    @classmethod
    def from_mapping(cls, data: Any) -> OptimizerSettings:
        raw = _reject_unknown(data, frozenset({"name"}), "optimizer")
        return cls(**raw) if raw else cls()


@dataclass(frozen=True)
class DatasetSpec:
    """Which preference dataset to train on (spec sections 21, 25, 96)."""

    preference_run_id: str | None = None

    def __post_init__(self) -> None:
        if self.preference_run_id is not None:
            _require_text(self.preference_run_id, "dataset.preference_run_id")

    def to_dict(self) -> dict[str, Any]:
        return {"preference_run_id": self.preference_run_id}

    @classmethod
    def from_mapping(cls, data: Any) -> DatasetSpec:
        raw = _reject_unknown(data, frozenset({"preference_run_id"}), "dataset")
        return cls(**raw) if raw else cls()


@dataclass(frozen=True)
class DistributedSettings:
    """Accepted, and rejected when enabled (spec sections 86, 87).

    The spec allows the *key* to exist so a later stage can turn it on, but is explicit
    that Step 9 must not implement distributed training. Accepting ``true`` and quietly
    training on one GPU anyway would be the worst outcome, so it fails loudly.
    """

    enabled: bool = False

    def __post_init__(self) -> None:
        _require_flag(self.enabled, "distributed.enabled")
        if self.enabled:
            raise TrainingConfigError(
                "distributed.enabled must be false; Step 9 supports a single GPU only "
                "and does not implement distributed training"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    @classmethod
    def from_mapping(cls, data: Any) -> DistributedSettings:
        raw = _reject_unknown(data, frozenset({"enabled"}), "distributed")
        return cls(**raw) if raw else cls()


# ---------------------------------------------------------------------------- experiment


@dataclass(frozen=True)
class ExperimentConfig:
    """One training experiment, as loaded from a YAML file (spec section 62)."""

    model: ModelSpec
    experiment_name: str = "qwen-python-dpo"
    quantization: QuantizationSettings = field(default_factory=QuantizationSettings)
    lora: LoraSettings = field(default_factory=LoraSettings)
    dpo: DpoSettings = field(default_factory=DpoSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    distributed: DistributedSettings = field(default_factory=DistributedSettings)

    def __post_init__(self) -> None:
        _require_text(self.experiment_name, "experiment_name")

    @property
    def effective_batch_size(self) -> int:
        """Spec section 38. Single-GPU only (section 86), so device count is always 1."""
        return (
            self.training.per_device_train_batch_size
            * self.training.gradient_accumulation_steps
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "model": self.model.to_dict(),
            "quantization": self.quantization.to_dict(),
            "lora": self.lora.to_dict(),
            "dpo": self.dpo.to_dict(),
            "training": self.training.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "dataset": self.dataset.to_dict(),
            "distributed": self.distributed.to_dict(),
        }

    @classmethod
    def from_mapping(cls, data: Any) -> ExperimentConfig:
        raw = _reject_unknown(data, _TOP_LEVEL_KEYS, "experiment config")
        if "model" not in raw:
            raise TrainingConfigError("experiment config: 'model' section is required")
        return cls(
            experiment_name=raw.get("experiment_name", "qwen-python-dpo"),
            model=ModelSpec.from_mapping(raw["model"]),
            quantization=QuantizationSettings.from_mapping(raw.get("quantization")),
            lora=LoraSettings.from_mapping(raw.get("lora")),
            dpo=DpoSettings.from_mapping(raw.get("dpo")),
            training=TrainingSettings.from_mapping(raw.get("training")),
            optimizer=OptimizerSettings.from_mapping(raw.get("optimizer")),
            dataset=DatasetSpec.from_mapping(raw.get("dataset")),
            distributed=DistributedSettings.from_mapping(raw.get("distributed")),
        )

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        path = Path(path)
        if not path.is_file():
            raise TrainingConfigError(f"training config not found at {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise TrainingConfigError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise TrainingConfigError(f"{path}: root must be a mapping")
        try:
            return cls.from_mapping(raw)
        except TrainingConfigError as exc:
            raise TrainingConfigError(f"{path}: {exc}") from exc

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        """Apply CLI overrides (spec section 63), ignoring ``None`` values.

        Every override maps onto exactly one configuration field, so a flag can never
        change something the user did not name. Re-validates by reconstructing through
        :meth:`from_mapping`, so an override cannot bypass a rule the YAML must obey.
        """
        supplied = {key: value for key, value in overrides.items() if value is not None}
        if not supplied:
            return self

        data = self.to_dict()
        routes: dict[str, tuple[str, str]] = {
            "experiment_name": ("", "experiment_name"),
            "preference_run_id": ("dataset", "preference_run_id"),
            "model_name": ("model", "name"),
            "model_revision": ("model", "revision"),
            "beta": ("dpo", "beta"),
            "learning_rate": ("training", "learning_rate"),
            "num_train_epochs": ("training", "num_train_epochs"),
            "max_steps": ("training", "max_steps"),
            "max_length": ("training", "max_length"),
            "max_prompt_length": ("training", "max_prompt_length"),
            "seed": ("training", "seed"),
            "lora_r": ("lora", "r"),
            "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
            "eval_strategy": ("training", "eval_strategy"),
        }
        for key, value in supplied.items():
            if key not in routes:
                raise TrainingConfigError(f"unknown configuration override: {key!r}")
            section, name = routes[key]
            if section:
                data[section][name] = value
            else:
                data[name] = value
        return ExperimentConfig.from_mapping(data)


__all__ = [
    "COMPUTE_DTYPES",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OPTIMIZER",
    "DPO_LOSS_TYPES",
    "EVAL_STRATEGIES",
    "FALLBACK_OPTIMIZER",
    "LORA_BIASES",
    "QUANT_TYPES",
    "SCHEDULERS",
    "TASK_TYPES",
    "DatasetSpec",
    "DistributedSettings",
    "DpoSettings",
    "ExperimentConfig",
    "LoraSettings",
    "ModelSpec",
    "OptimizerSettings",
    "QuantizationSettings",
    "TrainingSettings",
]
