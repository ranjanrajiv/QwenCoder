"""The experiment configuration and its four-level hierarchy (spec 12 sections 8, 9, 10).

``ExperimentConfig.load`` resolves:

```
root config.yaml   (untouched -- individual stage adapters already read it directly,
                     per the plan's "the orchestrator adds a layer, it does not rewrite
                     one" boundary; this module does not re-parse it)
        v
experiment file     (--config PATH)
        v
stage section       (each top-level block inside the experiment file)
        v
CLI --set overrides (highest priority, applied last)
```

Only the shape every stage shares is validated here: an ``enabled`` flag per stage
section, and the top-level ``experiment``/``git``/``hypothesis``/``success_criteria``
blocks. Everything else inside a stage's section is passed through opaquely as
``StageConfig.settings`` -- deep validation of, say, ``dpo_training``'s beta or
``model_evaluation``'s benchmark name is each stage adapter's own job (several already
have a dedicated config module, e.g. :mod:`python_dpo.training.config`), matching
CLAUDE.md's rule against building ahead of what the current stage needs.

``problem_generation`` from the spec's literal example is named ``problem_dataset`` here,
matching :mod:`python_dpo.pipeline.stages` -- Stage 2's catalog is ten curated problems
with reference solutions, not an LLM generator, so ``problem_count: 1000`` from the spec's
own example is unimplementable; the key that exists instead only *selects a subset*.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ExperimentConfigError
from .stages import STAGE_NAMES

_TOP_LEVEL_KEYS = frozenset({"experiment", "git", "hypothesis", "success_criteria", *STAGE_NAMES})

# Spec section 24, 25: reduce scale while preserving the complete pipeline. Applied when
# `--smoke-test` is passed, before `--set` (which must still win, per section 9).
SMOKE_TEST_OVERRIDES: dict[str, dict[str, Any]] = {
    "problem_dataset": {"problem_count": 3},
    "candidate_generation": {"candidates_per_problem": 2},
    "dpo_training": {"max_steps": 1},
    "model_evaluation": {"num_samples": 2, "limit": 2},
}


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class StageConfig:
    """One stage's section of the experiment file: ``enabled`` plus opaque settings."""

    name: str
    enabled: bool
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in STAGE_NAMES:
            raise ExperimentConfigError(f"unknown stage {self.name!r}")
        if not isinstance(self.enabled, bool):
            raise ExperimentConfigError(f"{self.name}.enabled must be true or false")
        if not isinstance(self.settings, dict):
            raise ExperimentConfigError(f"{self.name}: settings must be a mapping")

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, **self.settings}

    @classmethod
    def from_mapping(cls, name: str, data: Any) -> StageConfig:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ExperimentConfigError(f"{name}: must be a mapping")
        if "enabled" not in data:
            raise ExperimentConfigError(f"{name}.enabled is required")
        settings = {key: value for key, value in data.items() if key != "enabled"}
        return cls(name=name, enabled=data["enabled"], settings=settings)


@dataclass(frozen=True)
class ExperimentConfig:
    """The fully resolved experiment configuration (spec section 8)."""

    name: str
    stages: dict[str, StageConfig]
    seed: int = 42
    on_dirty: str = "warn"
    hypothesis: str | None = None
    success_criteria: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.name, "experiment.name")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ExperimentConfigError("experiment.seed must be an integer")
        if self.on_dirty not in ("warn", "fail"):
            raise ExperimentConfigError(
                f"git.on_dirty must be 'warn' or 'fail', got {self.on_dirty!r}"
            )
        if self.hypothesis is not None:
            _require_text(self.hypothesis, "hypothesis.description")
        if not isinstance(self.success_criteria, dict):
            raise ExperimentConfigError("success_criteria must be a mapping")

        if not isinstance(self.stages, dict) or any(
            not isinstance(v, StageConfig) for v in self.stages.values()
        ):
            raise ExperimentConfigError("stages must map stage names to StageConfig")
        missing = sorted(set(STAGE_NAMES) - set(self.stages))
        if missing:
            raise ExperimentConfigError(f"missing stage section(s): {', '.join(missing)}")
        extra = sorted(set(self.stages) - set(STAGE_NAMES))
        if extra:
            raise ExperimentConfigError(f"unknown stage section(s): {', '.join(extra)}")

    def stage(self, name: str) -> StageConfig:
        try:
            return self.stages[name]
        except KeyError:
            raise ExperimentConfigError(f"unknown stage {name!r}") from None

    def enabled_stages(self) -> tuple[str, ...]:
        return tuple(name for name in STAGE_NAMES if self.stages[name].enabled)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "experiment": {"name": self.name, "seed": self.seed},
            "git": {"on_dirty": self.on_dirty},
            "success_criteria": dict(self.success_criteria),
        }
        if self.hypothesis is not None:
            data["hypothesis"] = {"description": self.hypothesis}
        for stage_name in STAGE_NAMES:
            data[stage_name] = self.stages[stage_name].to_dict()
        return data

    @classmethod
    def from_mapping(cls, data: Any) -> ExperimentConfig:
        if not isinstance(data, dict):
            raise ExperimentConfigError("experiment config: root must be a mapping")
        unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
        if unknown:
            raise ExperimentConfigError(
                f"experiment config: unknown top-level key(s): {', '.join(unknown)}"
            )

        experiment_section = data.get("experiment")
        if not isinstance(experiment_section, dict) or "name" not in experiment_section:
            raise ExperimentConfigError(
                "experiment config: missing required key 'experiment.name'"
            )
        name = experiment_section["name"]
        seed = experiment_section.get("seed", 42)

        git_section = data.get("git") or {}
        if not isinstance(git_section, dict):
            raise ExperimentConfigError("experiment config: 'git' must be a mapping")
        on_dirty = git_section.get("on_dirty", "warn")

        hypothesis_section = data.get("hypothesis")
        hypothesis: str | None = None
        if hypothesis_section is not None:
            if not isinstance(hypothesis_section, dict) or "description" not in hypothesis_section:
                raise ExperimentConfigError(
                    "experiment config: 'hypothesis' must be a mapping with 'description'"
                )
            hypothesis = hypothesis_section["description"]

        success_criteria = data.get("success_criteria") or {}
        if not isinstance(success_criteria, dict):
            raise ExperimentConfigError("experiment config: 'success_criteria' must be a mapping")

        stages = {
            stage_name: StageConfig.from_mapping(stage_name, data.get(stage_name))
            for stage_name in STAGE_NAMES
        }

        return cls(
            name=name,
            stages=stages,
            seed=seed,
            on_dirty=on_dirty,
            hypothesis=hypothesis,
            success_criteria=success_criteria,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        overrides: Sequence[str] = (),
        smoke_test: bool = False,
    ) -> ExperimentConfig:
        path = Path(path)
        if not path.is_file():
            raise ExperimentConfigError(f"experiment config not found at {path}")
        with path.open("r", encoding="utf-8") as handle:
            try:
                raw: Any = yaml.safe_load(handle)
            except yaml.YAMLError as exc:
                raise ExperimentConfigError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ExperimentConfigError(f"{path}: root must be a mapping")

        merged = apply_smoke_test(raw) if smoke_test else copy.deepcopy(raw)
        merged = apply_overrides(merged, overrides)
        return cls.from_mapping(merged)


def apply_smoke_test(data: dict[str, Any]) -> dict[str, Any]:
    """Section 24/25: reduce scale in every stage that has a scale knob, in place on a copy."""
    result = copy.deepcopy(data)
    for stage_name, stage_overrides in SMOKE_TEST_OVERRIDES.items():
        section = result.get(stage_name)
        if isinstance(section, dict):
            section.update(stage_overrides)
    return result


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply repeated ``--set stage.key=value`` flags -- the highest-priority layer
    (spec section 9). Each value is parsed with ``yaml.safe_load`` so ``2`` becomes an
    int, ``0.2`` a float, ``true`` a bool, and anything else a string, matching how the
    value would have been typed directly into the YAML file.
    """
    result = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ExperimentConfigError(f"--set {item!r} must be in the form key.path=value")
        dotted_path, _, raw_value = item.partition("=")
        keys = dotted_path.split(".")
        if not dotted_path or any(not key for key in keys):
            raise ExperimentConfigError(f"--set {item!r}: invalid key path {dotted_path!r}")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ExperimentConfigError(f"--set {item!r}: invalid value: {exc}") from exc

        cursor = result
        for key in keys[:-1]:
            existing = cursor.get(key)
            if not isinstance(existing, dict):
                existing = {}
                cursor[key] = existing
            cursor = existing
        cursor[keys[-1]] = value
    return result


__all__ = [
    "SMOKE_TEST_OVERRIDES",
    "ExperimentConfig",
    "StageConfig",
    "apply_overrides",
    "apply_smoke_test",
]
