"""The Stage 10 evaluation configuration (spec section 140).

Deliberately a **standalone YAML file** (``configs/evaluation/python_eval.yaml``), not a
section of the root ``config.yaml`` — matching Stage 9's ``configs/training/dpo_qlora.yaml``
rationale. ``python_dpo.models.base.ModelConfig`` rejects a non-null ``quantization``
outright, so the root config's ``model:`` section still cannot express 4-bit inference;
this file is where Stage 10's own quantized, seeded, k-sample generation is configured.

Nothing here imports torch. Dtype/quant-type names are validated as strings against
closed sets and only resolved to real objects inside :mod:`python_dpo.model_evaluation.runners`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ModelEvaluationConfigError

QUANT_TYPES = frozenset({"nf4", "fp4"})
COMPUTE_DTYPES = frozenset({"bfloat16", "float16", "float32", "auto"})

DEFAULT_CONFIG_PATH = Path("configs/evaluation/python_eval.yaml")
DEFAULT_BENCHMARK_NAME = "python_eval_v1"

_TOP_LEVEL_KEYS = frozenset(
    {"benchmark", "generation", "quantization", "statistics", "success_criteria"}
)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelEvaluationConfigError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelEvaluationConfigError(f"{label} must be an integer of 1 or greater")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelEvaluationConfigError(f"{label} must be a number")
    return float(value)


def _require_flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ModelEvaluationConfigError(f"{label} must be true or false")
    return value


def _reject_unknown(data: Any, allowed: frozenset[str], label: str) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ModelEvaluationConfigError(f"{label} must be a mapping")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ModelEvaluationConfigError(f"{label}: unknown key(s): {', '.join(unknown)}")
    return dict(data)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Which benchmark to evaluate against (spec section 8)."""

    name: str = DEFAULT_BENCHMARK_NAME

    def __post_init__(self) -> None:
        _require_text(self.name, "benchmark.name")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}

    @classmethod
    def from_mapping(cls, data: Any) -> BenchmarkSpec:
        raw = _reject_unknown(data, frozenset({"name"}), "benchmark")
        return cls(**raw) if raw else cls()


@dataclass(frozen=True)
class GenerationSettings:
    """Decoding parameters, identical for base and DPO by construction (spec sections
    20-26, 140).

    ``base_seed`` drives the deterministic per-(problem, sample) seed schedule in
    :mod:`python_dpo.model_evaluation.generation` (spec sections 25, 26).
    """

    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 512
    num_samples: int = 10
    do_sample: bool = True
    repetition_penalty: float = 1.0
    base_seed: int = 1000

    def __post_init__(self) -> None:
        temperature = _require_number(self.temperature, "generation.temperature")
        top_p = _require_number(self.top_p, "generation.top_p")
        _require_positive_int(self.max_new_tokens, "generation.max_new_tokens")
        _require_positive_int(self.num_samples, "generation.num_samples")
        _require_flag(self.do_sample, "generation.do_sample")
        repetition_penalty = _require_number(
            self.repetition_penalty, "generation.repetition_penalty"
        )
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise ModelEvaluationConfigError("generation.base_seed must be an integer")

        if temperature < 0:
            raise ModelEvaluationConfigError("generation.temperature must not be negative")
        if self.do_sample and temperature <= 0:
            raise ModelEvaluationConfigError(
                "generation.temperature must be greater than 0 when do_sample is true"
            )
        if not 0 < top_p <= 1:
            raise ModelEvaluationConfigError("generation.top_p must be in the range (0, 1]")
        if repetition_penalty <= 0:
            raise ModelEvaluationConfigError(
                "generation.repetition_penalty must be greater than 0"
            )
        if self.base_seed < 0:
            raise ModelEvaluationConfigError("generation.base_seed must not be negative")

        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "repetition_penalty", repetition_penalty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "num_samples": self.num_samples,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "base_seed": self.base_seed,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> GenerationSettings:
        raw = _reject_unknown(
            data,
            frozenset(
                {
                    "temperature",
                    "top_p",
                    "max_new_tokens",
                    "num_samples",
                    "do_sample",
                    "repetition_penalty",
                    "base_seed",
                }
            ),
            "generation",
        )
        return cls(**raw) if raw else cls()


@dataclass(frozen=True)
class QuantizationSettings:
    """The spec section 19 identical-quantization requirement, structural rather than a
    thing to remember: one config feeds both :class:`~python_dpo.model_evaluation.runners.BaseModelRunner`
    and :class:`~python_dpo.model_evaluation.runners.AdapterModelRunner`.
    """

    enabled: bool = True
    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        _require_flag(self.enabled, "quantization.enabled")
        _require_flag(self.double_quant, "quantization.double_quant")
        if self.bits not in (4, 8):
            raise ModelEvaluationConfigError("quantization.bits must be 4 or 8")
        if self.quant_type not in QUANT_TYPES:
            raise ModelEvaluationConfigError(
                f"quantization.quant_type must be one of {', '.join(sorted(QUANT_TYPES))}"
            )
        if self.compute_dtype not in COMPUTE_DTYPES:
            raise ModelEvaluationConfigError(
                "quantization.compute_dtype must be one of "
                f"{', '.join(sorted(COMPUTE_DTYPES))}"
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
class StatisticsSettings:
    """Bootstrap and pass@k settings (spec sections 59-66, 105, 106)."""

    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 42
    confidence_level: float = 0.95
    pass_at_k: tuple[int, ...] = (1, 5, 10)

    def __post_init__(self) -> None:
        _require_positive_int(self.bootstrap_iterations, "statistics.bootstrap_iterations")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(self.bootstrap_seed, int):
            raise ModelEvaluationConfigError("statistics.bootstrap_seed must be an integer")
        confidence = _require_number(self.confidence_level, "statistics.confidence_level")
        if not 0 < confidence < 1:
            raise ModelEvaluationConfigError(
                "statistics.confidence_level must be in the range (0, 1)"
            )
        object.__setattr__(self, "confidence_level", confidence)

        if isinstance(self.pass_at_k, (str, bytes)) or not isinstance(
            self.pass_at_k, (list, tuple)
        ):
            raise ModelEvaluationConfigError("statistics.pass_at_k must be a list of integers")
        object.__setattr__(self, "pass_at_k", tuple(self.pass_at_k))
        if not self.pass_at_k:
            raise ModelEvaluationConfigError("statistics.pass_at_k must not be empty")
        for k in self.pass_at_k:
            _require_positive_int(k, "statistics.pass_at_k entry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "pass_at_k": list(self.pass_at_k),
        }

    @classmethod
    def from_mapping(cls, data: Any) -> StatisticsSettings:
        raw = _reject_unknown(
            data,
            frozenset(
                {"bootstrap_iterations", "bootstrap_seed", "confidence_level", "pass_at_k"}
            ),
            "statistics",
        )
        return cls(**raw) if raw else cls()


@dataclass(frozen=True)
class SuccessCriteria:
    """The spec section 86, 87, 88, 143 experimental gates. Configurable, not universal."""

    minimum_pass_at_1_improvement: float = 0.02
    maximum_allowed_regression: float = 0.02

    def __post_init__(self) -> None:
        minimum = _require_number(
            self.minimum_pass_at_1_improvement, "success_criteria.minimum_pass_at_1_improvement"
        )
        maximum = _require_number(
            self.maximum_allowed_regression, "success_criteria.maximum_allowed_regression"
        )
        if minimum < 0:
            raise ModelEvaluationConfigError(
                "success_criteria.minimum_pass_at_1_improvement must not be negative"
            )
        if maximum < 0:
            raise ModelEvaluationConfigError(
                "success_criteria.maximum_allowed_regression must not be negative"
            )
        object.__setattr__(self, "minimum_pass_at_1_improvement", minimum)
        object.__setattr__(self, "maximum_allowed_regression", maximum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_pass_at_1_improvement": self.minimum_pass_at_1_improvement,
            "maximum_allowed_regression": self.maximum_allowed_regression,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> SuccessCriteria:
        raw = _reject_unknown(
            data,
            frozenset({"minimum_pass_at_1_improvement", "maximum_allowed_regression"}),
            "success_criteria",
        )
        return cls(**raw) if raw else cls()


@dataclass(frozen=True)
class EvaluationExperimentConfig:
    """One Stage 10 evaluation experiment, as loaded from a YAML file (spec section 140)."""

    benchmark: BenchmarkSpec = field(default_factory=BenchmarkSpec)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    quantization: QuantizationSettings = field(default_factory=QuantizationSettings)
    statistics: StatisticsSettings = field(default_factory=StatisticsSettings)
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)

    def __post_init__(self) -> None:
        # Spec section 43: pass@k for k > num_samples cannot be estimated correctly, so
        # requesting it is a configuration error rather than a silently degraded metric.
        for k in self.statistics.pass_at_k:
            if k > self.generation.num_samples:
                raise ModelEvaluationConfigError(
                    f"statistics.pass_at_k includes {k}, but generation.num_samples is "
                    f"{self.generation.num_samples}; pass@{k} requires at least {k} "
                    "sample(s) per problem"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark.to_dict(),
            "generation": self.generation.to_dict(),
            "quantization": self.quantization.to_dict(),
            "statistics": self.statistics.to_dict(),
            "success_criteria": self.success_criteria.to_dict(),
        }

    @classmethod
    def from_mapping(cls, data: Any) -> EvaluationExperimentConfig:
        raw = _reject_unknown(data, _TOP_LEVEL_KEYS, "evaluation config")
        return cls(
            benchmark=BenchmarkSpec.from_mapping(raw.get("benchmark")),
            generation=GenerationSettings.from_mapping(raw.get("generation")),
            quantization=QuantizationSettings.from_mapping(raw.get("quantization")),
            statistics=StatisticsSettings.from_mapping(raw.get("statistics")),
            success_criteria=SuccessCriteria.from_mapping(raw.get("success_criteria")),
        )

    @classmethod
    def load(cls, path: Path) -> EvaluationExperimentConfig:
        path = Path(path)
        if not path.is_file():
            raise ModelEvaluationConfigError(f"evaluation config not found at {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ModelEvaluationConfigError(f"{path}: invalid YAML: {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ModelEvaluationConfigError(f"{path}: root must be a mapping")
        try:
            return cls.from_mapping(raw)
        except ModelEvaluationConfigError as exc:
            raise ModelEvaluationConfigError(f"{path}: {exc}") from exc

    def with_overrides(self, **overrides: Any) -> EvaluationExperimentConfig:
        """Apply CLI overrides (mirrors Stage 9's ``ExperimentConfig.with_overrides``)."""
        supplied = {key: value for key, value in overrides.items() if value is not None}
        if not supplied:
            return self

        data = self.to_dict()
        routes: dict[str, tuple[str, str]] = {
            "benchmark_name": ("benchmark", "name"),
            "num_samples": ("generation", "num_samples"),
            "temperature": ("generation", "temperature"),
            "max_new_tokens": ("generation", "max_new_tokens"),
            "bootstrap_iterations": ("statistics", "bootstrap_iterations"),
        }
        for key, value in supplied.items():
            if key not in routes:
                raise ModelEvaluationConfigError(f"unknown configuration override: {key!r}")
            section, name = routes[key]
            data[section][name] = value
        return EvaluationExperimentConfig.from_mapping(data)


__all__ = [
    "COMPUTE_DTYPES",
    "DEFAULT_BENCHMARK_NAME",
    "DEFAULT_CONFIG_PATH",
    "QUANT_TYPES",
    "BenchmarkSpec",
    "EvaluationExperimentConfig",
    "GenerationSettings",
    "QuantizationSettings",
    "StatisticsSettings",
    "SuccessCriteria",
]
