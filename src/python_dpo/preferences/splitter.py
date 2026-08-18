"""Deterministic, problem-level train/validation/test splitting (spec 08 sections 62-68).

The split unit is ``problem_id``, never a preference pair (spec section 63): every pair
belonging to one problem lands in exactly one split, which is what makes it structurally
impossible for the same prompt to appear in both train and validation (spec section 62).

The pool being split is the set of problems that produced at least one preference pair
(decision 4) — splitting the *entire* problem set would spend validation/test budget on
problems contributing nothing, and at ten problems that is measured to leave a split empty
21% of the time (seed sweep in the Stage 8 plan). A floor rule guarantees ``train`` is
non-empty whenever the pool is non-empty, since a training split of size zero is never
useful even when the arithmetic floor rounds it there.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .errors import PreferenceError

SPLITTER_VERSION = "v1"

DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}
DEFAULT_SPLIT_SEED = 42

_SPLIT_MANIFEST_FIELDS = frozenset(
    {
        "splitter_version",
        "train_problem_ids",
        "validation_problem_ids",
        "test_problem_ids",
        "seed",
        "split_ratios",
    }
)


class SplitConfigError(PreferenceError):
    """Raised when the requested split ratios are invalid."""


@dataclass(frozen=True)
class SplitManifest:
    """Split membership (spec section 68) — auditable and independently reproducible from
    ``seed``/``split_ratios`` alone (spec section 67).
    """

    train_problem_ids: tuple[str, ...]
    validation_problem_ids: tuple[str, ...]
    test_problem_ids: tuple[str, ...]
    seed: int
    split_ratios: dict[str, float]
    splitter_version: str = SPLITTER_VERSION

    def __post_init__(self) -> None:
        for name in ("train_problem_ids", "validation_problem_ids", "test_problem_ids"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
                raise PreferenceError(f"{name} must be a sequence of problem ids")
            object.__setattr__(self, name, tuple(value))

        all_ids = (
            self.train_problem_ids + self.validation_problem_ids + self.test_problem_ids
        )
        # Spec section 62: the check that directly protects against data leakage.
        if len(set(all_ids)) != len(all_ids):
            raise PreferenceError(
                "a problem id must not appear in more than one split (spec section 62)"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise PreferenceError("seed must be an integer")
        if not isinstance(self.split_ratios, dict):
            raise PreferenceError("split_ratios must be a mapping")

    def split_of(self, problem_id: str) -> str | None:
        if problem_id in self.train_problem_ids:
            return "train"
        if problem_id in self.validation_problem_ids:
            return "validation"
        if problem_id in self.test_problem_ids:
            return "test"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "splitter_version": self.splitter_version,
            "train_problem_ids": list(self.train_problem_ids),
            "validation_problem_ids": list(self.validation_problem_ids),
            "test_problem_ids": list(self.test_problem_ids),
            "seed": self.seed,
            "split_ratios": self.split_ratios,
        }

    @classmethod
    def from_dict(cls, data: Any) -> SplitManifest:
        if not isinstance(data, dict):
            raise PreferenceError("split manifest: expected a JSON object")
        unknown = sorted(set(data) - _SPLIT_MANIFEST_FIELDS)
        if unknown:
            raise PreferenceError(f"split manifest: unknown field(s): {', '.join(unknown)}")
        required = _SPLIT_MANIFEST_FIELDS - {"splitter_version"}
        missing = sorted(required - set(data))
        if missing:
            raise PreferenceError(f"split manifest: missing required field(s): {', '.join(missing)}")
        return cls(
            train_problem_ids=tuple(data["train_problem_ids"]),
            validation_problem_ids=tuple(data["validation_problem_ids"]),
            test_problem_ids=tuple(data["test_problem_ids"]),
            seed=data["seed"],
            split_ratios=dict(data["split_ratios"]),
            splitter_version=data.get("splitter_version", SPLITTER_VERSION),
        )


class ProblemSplitter:
    """Splits a pool of problem ids into train/validation/test (spec sections 64-67)."""

    def __init__(
        self,
        *,
        ratios: dict[str, float] | None = None,
        seed: int = DEFAULT_SPLIT_SEED,
    ) -> None:
        self.ratios = dict(ratios) if ratios is not None else dict(DEFAULT_SPLIT_RATIOS)
        self._validate_ratios(self.ratios)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SplitConfigError("split seed must be an integer")
        self.seed = seed

    @staticmethod
    def _validate_ratios(ratios: dict[str, float]) -> None:
        missing = sorted({"train", "validation", "test"} - set(ratios))
        if missing:
            raise SplitConfigError(f"split ratios: missing split(s): {', '.join(missing)}")
        unknown = sorted(set(ratios) - {"train", "validation", "test"})
        if unknown:
            raise SplitConfigError(f"split ratios: unknown split(s): {', '.join(unknown)}")
        for name, value in ratios.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise SplitConfigError(f"split ratios[{name!r}] must be a number >= 0")
        total = sum(ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise SplitConfigError(f"split ratios must sum to 1.0, got {total}")

    def split(self, problem_ids: Iterable[str]) -> SplitManifest:
        """Deterministically split ``problem_ids`` (spec section 67): a sorted pool is
        shuffled with ``random.Random(seed)`` — the input's own ordering never affects the
        result, and no timestamp or UUID is ever consulted.
        """
        pool = sorted(set(problem_ids))
        if not pool:
            return SplitManifest(
                train_problem_ids=(),
                validation_problem_ids=(),
                test_problem_ids=(),
                seed=self.seed,
                split_ratios=dict(self.ratios),
            )

        shuffled = list(pool)
        random.Random(self.seed).shuffle(shuffled)

        n = len(shuffled)
        n_train = math.floor(n * self.ratios["train"])
        # Floor rule: a non-empty pool must never produce an empty training split, even
        # when the arithmetic floor rounds it to zero (spec section 65's "acceptable for
        # pipeline development" does not mean "acceptable to be useless").
        if n_train == 0:
            n_train = 1
        n_train = min(n_train, n)

        train = shuffled[:n_train]
        remaining = shuffled[n_train:]
        n_validation = min(math.floor(n * self.ratios["validation"]), len(remaining))
        validation = remaining[:n_validation]
        test = remaining[n_validation:]

        return SplitManifest(
            train_problem_ids=tuple(sorted(train)),
            validation_problem_ids=tuple(sorted(validation)),
            test_problem_ids=tuple(sorted(test)),
            seed=self.seed,
            split_ratios=dict(self.ratios),
        )


__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_SPLIT_SEED",
    "SPLITTER_VERSION",
    "ProblemSplitter",
    "SplitConfigError",
    "SplitManifest",
]
