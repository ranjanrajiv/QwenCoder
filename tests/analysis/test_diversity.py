"""Tests for output diversity and mode collapse (spec 11 sections 26-31, 88, 114)."""

from __future__ import annotations

from python_dpo.analysis.diversity import build_diversity_report

from .conftest import FakeGeneration


def generations(variant: str, codes: list[str]) -> list[FakeGeneration]:
    return [
        FakeGeneration("p001", variant, i, extracted_code=code)
        for i, code in enumerate(codes)
    ]


# ------------------------------------------------------------ section 114's two named cases


def test_ten_identical_candidates_score_one_tenth():
    report = build_diversity_report({"base": generations("base", ["same"] * 10), "dpo": []})
    assert report.base_diversity == 0.1


def test_ten_unique_candidates_score_one():
    report = build_diversity_report(
        {"base": generations("base", [f"code{i}" for i in range(10)]), "dpo": []}
    )
    assert report.base_diversity == 1.0


# ------------------------------------------------------------------- the section 88 gate


def test_mode_collapse_warning_fires_past_the_threshold():
    """Base 10/10 unique, DPO 5/10 -- a 50% relative fall, well past the 20% gate."""
    report = build_diversity_report(
        {
            "base": generations("base", [f"code{i}" for i in range(10)]),
            "dpo": generations("dpo", [f"code{i // 2}" for i in range(10)]),
        },
        mode_collapse_reduction=0.2,
    )
    assert report.mode_collapse_warning is True
    assert report.relative_change < 0


def test_mode_collapse_warning_does_not_fire_below_the_threshold():
    """Base 10/10, DPO 9/10 -- a 10% relative fall, below the gate."""
    report = build_diversity_report(
        {
            "base": generations("base", [f"code{i}" for i in range(10)]),
            "dpo": generations("dpo", [*[f"code{i}" for i in range(9)], "code0"]),
        },
        mode_collapse_reduction=0.2,
    )
    assert report.mode_collapse_warning is False


def test_threshold_is_relative_not_absolute():
    """Both variants already low-diversity: an absolute rule would miss a large
    proportional collapse. 0.2 -> 0.1 is a 50% relative fall."""
    report = build_diversity_report(
        {
            "base": generations("base", [f"code{i // 5}" for i in range(10)]),
            "dpo": generations("dpo", ["same"] * 10),
        },
        mode_collapse_reduction=0.2,
    )
    assert report.base_diversity == 0.2
    assert report.dpo_diversity == 0.1
    assert report.mode_collapse_warning is True


def test_samples_without_extracted_code_are_excluded():
    records = generations("base", ["a", "b"]) + [
        FakeGeneration("p001", "base", 9, extracted_code=None)
    ]
    report = build_diversity_report({"base": records, "dpo": []})
    assert report.base_total == 2


def test_no_warning_when_base_produced_nothing():
    """Guards the division; no base output means no baseline to collapse from."""
    report = build_diversity_report({"base": [], "dpo": generations("dpo", ["a"])})
    assert report.mode_collapse_warning is False
    assert report.relative_change is None
