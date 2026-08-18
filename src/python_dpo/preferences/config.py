"""The ``preferences:`` configuration section (spec 08 sections 21, 22, 66).

Unlike Stage 7 (no tunable scoring parameters), preference generation has four genuinely
tunable knobs — the default policy, the minimum score margin, the pair cap per problem,
and the split ratios/seed — and spec section 22 explicitly forbids hard-coding the margin
in the pair builder. This module owns the section's own error type
(:class:`PreferenceConfigError`) so ``python_dpo.config`` can translate it, keeping the
dependency one-way, exactly as ``evaluation.config``/``sandbox.config`` already do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import PreferenceConfigError
from .models import POLICIES
from .policies import DEFAULT_MINIMUM_SCORE_MARGIN
from .splitter import DEFAULT_SPLIT_RATIOS, DEFAULT_SPLIT_SEED

_PREFERENCES_KEYS = frozenset(
    {"policy", "minimum_score_margin", "max_pairs_per_problem", "split"}
)
_SPLIT_KEYS = frozenset({"train", "validation", "test", "seed"})

DEFAULT_POLICY = "strict"


@dataclass(frozen=True)
class SplitSettings:
    train: float = DEFAULT_SPLIT_RATIOS["train"]
    validation: float = DEFAULT_SPLIT_RATIOS["validation"]
    test: float = DEFAULT_SPLIT_RATIOS["test"]
    seed: int = DEFAULT_SPLIT_SEED

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise PreferenceConfigError(f"preferences.split.{name} must be a number >= 0")
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise PreferenceConfigError(
                f"preferences.split ratios must sum to 1.0, got {total}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise PreferenceConfigError("preferences.split.seed must be an integer")

    def as_ratios(self) -> dict[str, float]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "seed": self.seed,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> SplitSettings:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise PreferenceConfigError("preferences.split must be a mapping")
        unknown = sorted(set(data) - _SPLIT_KEYS)
        if unknown:
            raise PreferenceConfigError(
                f"preferences.split: unknown key(s): {', '.join(unknown)}"
            )
        return cls(**data)


@dataclass(frozen=True)
class PreferenceConfig:
    """Default settings for ``preferences generate`` (spec sections 78, 84) — every field
    here is overridable per-invocation by the matching CLI flag.
    """

    policy: str = DEFAULT_POLICY
    minimum_score_margin: float = DEFAULT_MINIMUM_SCORE_MARGIN
    max_pairs_per_problem: int | None = None
    split: SplitSettings = SplitSettings()

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise PreferenceConfigError(
                f"preferences.policy must be one of {', '.join(sorted(POLICIES))}, "
                f"got {self.policy!r}"
            )
        if isinstance(self.minimum_score_margin, bool) or not isinstance(
            self.minimum_score_margin, (int, float)
        ):
            raise PreferenceConfigError("preferences.minimum_score_margin must be a number")
        if self.minimum_score_margin < 0:
            raise PreferenceConfigError("preferences.minimum_score_margin must be 0 or greater")
        if self.max_pairs_per_problem is not None:
            if (
                isinstance(self.max_pairs_per_problem, bool)
                or not isinstance(self.max_pairs_per_problem, int)
                or self.max_pairs_per_problem < 1
            ):
                raise PreferenceConfigError(
                    "preferences.max_pairs_per_problem must be a positive integer or null"
                )
        if not isinstance(self.split, SplitSettings):
            raise PreferenceConfigError("preferences.split must resolve to SplitSettings")

    @classmethod
    def from_mapping(cls, data: Any) -> PreferenceConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise PreferenceConfigError("preferences: expected a mapping")
        unknown = sorted(set(data) - _PREFERENCES_KEYS)
        if unknown:
            raise PreferenceConfigError(f"preferences: unknown key(s): {', '.join(unknown)}")
        return cls(
            policy=data.get("policy", DEFAULT_POLICY),
            minimum_score_margin=data.get(
                "minimum_score_margin", DEFAULT_MINIMUM_SCORE_MARGIN
            ),
            max_pairs_per_problem=data.get("max_pairs_per_problem"),
            split=SplitSettings.from_mapping(data.get("split")),
        )


__all__ = ["DEFAULT_POLICY", "PreferenceConfig", "SplitSettings"]
