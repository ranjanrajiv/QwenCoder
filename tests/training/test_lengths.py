"""Tests for token-length and truncation analysis (spec 09 sections 34, 35, 36)."""

from __future__ import annotations

import pytest

from python_dpo.training.dataset import PreferenceRecord
from python_dpo.training.errors import TruncationThresholdError
from python_dpo.training.lengths import analyze_lengths, enforce_truncation_threshold


class WordTokenizer:
    """One token per whitespace-separated word — enough to test the arithmetic."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())


def record(prompt_words: int, chosen_words: int, rejected_words: int) -> PreferenceRecord:
    return PreferenceRecord(
        prompt=" ".join(["p"] * prompt_words),
        chosen=" ".join(["c"] * chosen_words),
        rejected=" ".join(["r"] * rejected_words),
    )


def analyze(records, max_length=100, max_prompt_length=50):
    return analyze_lengths(
        records, WordTokenizer(), max_length=max_length, max_prompt_length=max_prompt_length
    )


# ---------------------------------------------------------------------- distributions


def test_measures_each_quantity():
    analysis = analyze([record(10, 5, 7)])
    assert analysis.distributions["prompt"].maximum == 10
    assert analysis.distributions["chosen"].maximum == 5
    assert analysis.distributions["rejected"].maximum == 7
    assert analysis.distributions["prompt_chosen"].maximum == 15
    assert analysis.distributions["prompt_rejected"].maximum == 17


def test_percentiles_are_real_observed_values():
    records = [record(n, 1, 1) for n in range(1, 101)]
    dist = analyze(records, max_length=1000).distributions["prompt"]
    assert dist.p50 == 50
    assert dist.p90 == 90
    assert dist.p99 == 99
    assert dist.maximum == 100


def test_mean_is_reported():
    analysis = analyze([record(10, 1, 1), record(20, 1, 1)])
    assert analysis.distributions["prompt"].mean == pytest.approx(15.0)


def test_empty_records_produce_zeros():
    analysis = analyze([])
    assert analysis.examples == 0
    assert analysis.truncation_rate == 0.0
    assert analysis.distributions["prompt"].maximum == 0


# ------------------------------------------------------------------------ truncation


def test_nothing_truncated_when_everything_fits():
    analysis = analyze([record(10, 5, 5)], max_length=100)
    assert analysis.truncated_examples == 0
    assert analysis.truncation_rate == 0.0


def test_an_example_is_truncated_if_either_side_overflows():
    # prompt+chosen fits (10+5=15) but prompt+rejected does not (10+50=60).
    analysis = analyze([record(10, 5, 50)], max_length=20)
    assert analysis.truncated_examples == 1


def test_truncation_rate_is_a_fraction_of_examples():
    records = [record(5, 5, 5)] * 3 + [record(50, 50, 50)]
    analysis = analyze(records, max_length=20)
    assert analysis.truncated_examples == 1
    assert analysis.truncation_rate == pytest.approx(0.25)


def test_prompt_overflow_is_counted_separately():
    analysis = analyze([record(80, 1, 1)], max_length=1000, max_prompt_length=50)
    assert analysis.prompt_overflow_examples == 1
    assert analysis.truncated_examples == 0  # it still fits within max_length


# --------------------------------------------------------------- threshold (35, 36)


def test_within_threshold_passes():
    analysis = analyze([record(5, 5, 5)] * 100, max_length=100)
    enforce_truncation_threshold(analysis, max_truncation_rate=0.05)


def test_above_threshold_raises():
    records = [record(5, 5, 5)] * 90 + [record(50, 50, 50)] * 10
    analysis = analyze(records, max_length=20)
    with pytest.raises(TruncationThresholdError, match="above the"):
        enforce_truncation_threshold(analysis, max_truncation_rate=0.05)


def test_the_error_recommends_a_remedy():
    analysis = analyze([record(50, 50, 50)], max_length=20)
    with pytest.raises(TruncationThresholdError, match="Raise training.max_length"):
        enforce_truncation_threshold(analysis, max_truncation_rate=0.0)


def test_override_downgrades_the_failure_to_a_warning(caplog):
    analysis = analyze([record(50, 50, 50)], max_length=20)
    enforce_truncation_threshold(analysis, max_truncation_rate=0.0, override=True)
    assert any("explicitly overridden" in r.message for r in caplog.records)


def test_exactly_at_the_threshold_passes():
    records = [record(5, 5, 5)] * 95 + [record(50, 50, 50)] * 5
    analysis = analyze(records, max_length=20)
    assert analysis.truncation_rate == pytest.approx(0.05)
    enforce_truncation_threshold(analysis, max_truncation_rate=0.05)


def test_analysis_serializes():
    payload = analyze([record(10, 5, 5)]).to_dict()
    assert payload["examples"] == 1
    assert "prompt_chosen" in payload["distributions"]
