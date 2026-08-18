"""Token-length and truncation analysis (spec 09 sections 34, 35, 36).

Run during preflight, before training starts, because ``max_length`` is a guess until you
have measured the data against the real tokenizer. Silently truncating examples would
quietly change what the model learns from — a rejected response cut off mid-function is a
different training signal than the one the dataset recorded.

The tokenizer is a parameter rather than a global, so this module needs no torch and the
whole analysis is unit-testable with a trivial fake.

**Note on ``max_prompt_length``.** TRL 1.10's ``DPOConfig`` has no such field — prompt and
completion are truncated together against ``max_length``. The spec (section 33) still asks
for it, and it remains genuinely useful as a *diagnostic*: a prompt that alone consumes
most of ``max_length`` leaves no room for the response, which is worth reporting even
though it is not a value passed to the trainer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from .dataset import PreferenceRecord
from .errors import TruncationThresholdError

logger = logging.getLogger("python_dpo.training.lengths")

PERCENTILES = (50, 90, 95, 99)


class Tokenizer(Protocol):
    """The minimal surface this module needs — deliberately not a transformers type."""

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class LengthDistribution:
    """Percentiles for one measured quantity (spec section 34)."""

    name: str
    p50: int
    p90: int
    p95: int
    p99: int
    maximum: int
    mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.maximum,
            "mean": round(self.mean, 2),
        }


@dataclass(frozen=True)
class LengthAnalysis:
    """The full section 34/35 report for one split."""

    distributions: dict[str, LengthDistribution]
    examples: int
    truncated_examples: int
    prompt_overflow_examples: int
    max_length: int
    max_prompt_length: int

    @property
    def truncation_rate(self) -> float:
        if self.examples == 0:
            return 0.0
        return self.truncated_examples / self.examples

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "truncated_examples": self.truncated_examples,
            "truncation_rate": round(self.truncation_rate, 6),
            "prompt_overflow_examples": self.prompt_overflow_examples,
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
            "distributions": {
                name: dist.to_dict() for name, dist in sorted(self.distributions.items())
            },
        }


def _percentile(sorted_values: list[int], percentile: int) -> int:
    """Nearest-rank percentile — no interpolation, so every result is a real observed
    token count rather than a fractional value no example actually has."""
    if not sorted_values:
        return 0
    rank = max(1, min(len(sorted_values), round(percentile / 100 * len(sorted_values))))
    return sorted_values[rank - 1]


def _distribution(name: str, values: list[int]) -> LengthDistribution:
    ordered = sorted(values)
    return LengthDistribution(
        name=name,
        p50=_percentile(ordered, 50),
        p90=_percentile(ordered, 90),
        p95=_percentile(ordered, 95),
        p99=_percentile(ordered, 99),
        maximum=ordered[-1] if ordered else 0,
        mean=(sum(ordered) / len(ordered)) if ordered else 0.0,
    )


def analyze_lengths(
    records: list[PreferenceRecord],
    tokenizer: Tokenizer,
    *,
    max_length: int,
    max_prompt_length: int,
) -> LengthAnalysis:
    """Measure token lengths and count how many examples would be truncated.

    An example counts as truncated when **either** ``prompt + chosen`` or
    ``prompt + rejected`` exceeds ``max_length``: DPO sees both sequences, so losing
    either one damages the pair.
    """
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    prompt_chosen: list[int] = []
    prompt_rejected: list[int] = []
    truncated = 0
    prompt_overflow = 0

    for record in records:
        prompt_n = len(tokenizer.encode(record.prompt))
        chosen_n = len(tokenizer.encode(record.chosen))
        rejected_n = len(tokenizer.encode(record.rejected))

        prompt_lengths.append(prompt_n)
        chosen_lengths.append(chosen_n)
        rejected_lengths.append(rejected_n)
        prompt_chosen.append(prompt_n + chosen_n)
        prompt_rejected.append(prompt_n + rejected_n)

        if prompt_n + chosen_n > max_length or prompt_n + rejected_n > max_length:
            truncated += 1
        if prompt_n > max_prompt_length:
            prompt_overflow += 1

    return LengthAnalysis(
        distributions={
            "prompt": _distribution("prompt", prompt_lengths),
            "chosen": _distribution("chosen", chosen_lengths),
            "rejected": _distribution("rejected", rejected_lengths),
            "prompt_chosen": _distribution("prompt_chosen", prompt_chosen),
            "prompt_rejected": _distribution("prompt_rejected", prompt_rejected),
        },
        examples=len(records),
        truncated_examples=truncated,
        prompt_overflow_examples=prompt_overflow,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )


def enforce_truncation_threshold(
    analysis: LengthAnalysis,
    *,
    max_truncation_rate: float,
    override: bool = False,
) -> None:
    """Fail preflight when too much of the dataset would be truncated (sections 35, 36).

    ``override`` downgrades the failure to a warning for the rare case where truncation is
    understood and accepted; the rate is reported either way, never hidden.
    """
    rate = analysis.truncation_rate
    if rate <= max_truncation_rate:
        if analysis.truncated_examples:
            logger.info(
                "%d of %d example(s) exceed max_length=%d (%.1f%%), within the %.1f%% "
                "threshold",
                analysis.truncated_examples,
                analysis.examples,
                analysis.max_length,
                100 * rate,
                100 * max_truncation_rate,
            )
        return

    message = (
        f"{analysis.truncated_examples} of {analysis.examples} example(s) exceed "
        f"max_length={analysis.max_length} ({100 * rate:.1f}%), above the "
        f"{100 * max_truncation_rate:.1f}% threshold. Raise training.max_length, shorten "
        f"the prompts, or clean the dataset (p95 prompt+chosen is "
        f"{analysis.distributions['prompt_chosen'].p95} tokens)"
    )
    if override:
        logger.warning("%s; continuing because the threshold was explicitly overridden", message)
        return
    raise TruncationThresholdError(message)


__all__ = [
    "PERCENTILES",
    "LengthAnalysis",
    "LengthDistribution",
    "Tokenizer",
    "analyze_lengths",
    "enforce_truncation_threshold",
]
