"""Tests for PreferencePair/PreferenceRejection/PreferenceManifest/PreferenceStatistics
(spec 08 section 16 and surrounding schema).
"""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.preferences.models import (
    PreferenceManifest,
    PreferenceModelError,
    PreferencePair,
    PreferenceRejection,
    PreferenceStatistics,
    QualityReport,
    derive_candidates_considered,
)

PREF_RUN_ID = "pref_20260818_030805_73ce"
RANKING_RUN_ID = "rank_20260817_161726_a84d"
EVAL_RUN_ID = "eval_20260817_115154_dcd4"
CANDIDATE_RUN_ID = "run_20260817_055411"


def make_pair(**overrides: Any) -> PreferencePair:
    fields: dict[str, Any] = {
        "preference_id": "pref_p001_c001__p001_c002",
        "problem_id": "p001",
        "candidate_run_id": CANDIDATE_RUN_ID,
        "ranking_run_id": RANKING_RUN_ID,
        "evaluation_run_id": EVAL_RUN_ID,
        "chosen_candidate_id": "p001_c001",
        "rejected_candidate_id": "p001_c002",
        "prompt": "Solve the problem.",
        "chosen": "def f():\n    return 1",
        "rejected": "def f():\n    return 2",
        "chosen_score": 1.0,
        "rejected_score": 0.5,
        "score_margin": 0.5,
        "chosen_pass_rate": 1.0,
        "rejected_pass_rate": 0.5,
        "chosen_tests_passed": 10,
        "rejected_tests_passed": 5,
        "chosen_tests_total": 10,
        "rejected_tests_total": 10,
        "chosen_correctness": "correct",
        "rejected_correctness": "incorrect",
        "preference_strength": "strong",
        "selection_policy": "strict",
        "selection_policy_version": "strict_v1",
        "canonical_prompt_sha256": "a" * 64,
        "prompt_version": "v1",
        "chosen_generation_prompt_sha256": "b" * 64,
        "rejected_generation_prompt_sha256": "c" * 64,
        "chosen_strategy": "normal",
        "rejected_strategy": "alternative",
        "chosen_code_sha256": "d" * 64,
        "rejected_code_sha256": "e" * 64,
    }
    fields.update(overrides)
    return PreferencePair(**fields)


def make_rejection(**overrides: Any) -> PreferenceRejection:
    fields: dict[str, Any] = {
        "ranking_run_id": RANKING_RUN_ID,
        "problem_id": "p001",
        "candidate_a": "p001_c001",
        "candidate_b": "p001_c002",
        "reason": "tie",
        "detail": "equal score",
        "relation": "TIE",
        "score_a": 1.0,
        "score_b": 1.0,
        "score_margin": 0.0,
    }
    fields.update(overrides)
    return PreferenceRejection(**fields)


# --------------------------------------------------------------------------------- validity


def test_a_well_formed_pair_constructs():
    pair = make_pair()
    assert pair.preference_id == "pref_p001_c001__p001_c002"
    assert pair.created_at


def test_same_candidate_id_is_rejected():
    with pytest.raises(PreferenceModelError):
        make_pair(chosen_candidate_id="p001_c001", rejected_candidate_id="p001_c001")


def test_identical_code_text_is_rejected():
    with pytest.raises(PreferenceModelError):
        make_pair(rejected="def f():\n    return 1")  # same as chosen


def test_identical_code_sha256_is_rejected():
    with pytest.raises(PreferenceModelError):
        make_pair(rejected_code_sha256="d" * 64)  # same as chosen_code_sha256


def test_chosen_score_not_greater_than_rejected_is_rejected():
    with pytest.raises(PreferenceModelError):
        make_pair(chosen_score=0.5, rejected_score=0.5, score_margin=0.0)
    with pytest.raises(PreferenceModelError):
        make_pair(chosen_score=0.4, rejected_score=0.5, score_margin=-0.1)


def test_wrong_score_margin_is_rejected():
    with pytest.raises(PreferenceModelError):
        make_pair(score_margin=0.9)


def test_candidate_id_must_belong_to_problem():
    with pytest.raises(PreferenceModelError):
        make_pair(chosen_candidate_id="p002_c001")


def test_indeterminate_correctness_is_rejected():
    with pytest.raises(PreferenceModelError):
        make_pair(chosen_correctness="indeterminate")
    with pytest.raises(PreferenceModelError):
        make_pair(rejected_correctness="indeterminate")


def test_preference_strength_must_match_correctness():
    # correct vs incorrect must be "strong", not "medium".
    with pytest.raises(PreferenceModelError):
        make_pair(preference_strength="medium")
    # incorrect vs incorrect must be "medium", not "strong".
    with pytest.raises(PreferenceModelError):
        make_pair(chosen_correctness="incorrect", preference_strength="strong")


def test_medium_strength_pair_constructs():
    pair = make_pair(chosen_correctness="incorrect", preference_strength="medium")
    assert pair.preference_strength == "medium"


def test_duplicate_training_record_requires_canonical_preference_id():
    with pytest.raises(PreferenceModelError):
        make_pair(duplicate_training_record=True)


def test_duplicate_training_record_canonical_id_must_differ():
    with pytest.raises(PreferenceModelError):
        make_pair(
            duplicate_training_record=True,
            canonical_preference_id="pref_p001_c001__p001_c002",  # == own id
        )


def test_non_duplicate_pair_must_not_carry_a_canonical_id():
    with pytest.raises(PreferenceModelError):
        make_pair(canonical_preference_id="pref_p001_c003__p001_c004")


def test_a_valid_duplicate_pair_constructs():
    pair = make_pair(
        preference_id="pref_p001_c003__p001_c004",
        duplicate_training_record=True,
        canonical_preference_id="pref_p001_c001__p001_c002",
    )
    assert pair.duplicate_training_record
    assert pair.canonical_preference_id == "pref_p001_c001__p001_c002"


def test_pass_rate_must_match_tests_passed_over_total():
    with pytest.raises(PreferenceModelError):
        make_pair(chosen_pass_rate=0.9)


def test_tests_passed_cannot_exceed_tests_total():
    with pytest.raises(PreferenceModelError):
        make_pair(chosen_tests_passed=11)


def test_unknown_policy_is_rejected():
    with pytest.raises(PreferenceModelError):
        make_pair(selection_policy="not_a_policy")


# -------------------------------------------------------------------------- round trip


def test_to_dict_from_dict_round_trip():
    pair = make_pair()
    restored = PreferencePair.from_dict(pair.to_dict())
    assert restored == pair


def test_from_dict_rejects_unknown_field():
    data = make_pair().to_dict()
    data["bogus"] = "x"
    with pytest.raises(PreferenceModelError):
        PreferencePair.from_dict(data)


def test_from_dict_rejects_missing_field():
    data = make_pair().to_dict()
    del data["chosen_score"]
    with pytest.raises(PreferenceModelError):
        PreferencePair.from_dict(data)


def test_training_record_has_exactly_three_keys():
    record = make_pair().training_record()
    assert set(record) == {"prompt", "chosen", "rejected"}
    assert record["chosen"] == "def f():\n    return 1"


# ---------------------------------------------------------------------- PreferenceRejection


def test_rejection_round_trip():
    rejection = make_rejection()
    restored = PreferenceRejection.from_dict(rejection.to_dict())
    assert restored == rejection


def test_rejection_same_candidate_is_rejected():
    with pytest.raises(PreferenceModelError):
        make_rejection(candidate_a="p001_c001", candidate_b="p001_c001")


def test_rejection_unknown_reason_is_rejected():
    with pytest.raises(PreferenceModelError):
        make_rejection(reason="not_a_reason")


# ----------------------------------------------------------------------- PreferenceManifest


def make_manifest(**overrides: Any) -> PreferenceManifest:
    fields: dict[str, Any] = {
        "preference_run_id": PREF_RUN_ID,
        "ranking_run_id": RANKING_RUN_ID,
        "evaluation_run_id": EVAL_RUN_ID,
        "candidate_run_id": CANDIDATE_RUN_ID,
        "status": "created",
        "created_at": "2026-08-18T03:08:05Z",
        "preference_version": "v1",
        "selection_policy": "strict",
        "selection_policy_version": "strict_v1",
        "minimum_score_margin": 0.2,
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "split_seed": 42,
        "builder_version": "v1",
    }
    fields.update(overrides)
    return PreferenceManifest(**fields)


def test_manifest_round_trip():
    manifest = make_manifest()
    restored = PreferenceManifest.from_dict(manifest.to_dict())
    assert restored == manifest


def test_manifest_rejects_bad_split_ratios():
    with pytest.raises(PreferenceModelError):
        make_manifest(split_ratios={"train": 0.5, "validation": 0.1, "test": 0.1})


def test_manifest_status_transitions():
    manifest = make_manifest()
    running = manifest.with_status("running", started_at="2026-08-18T03:08:06Z")
    assert running.status == "running"
    with pytest.raises(PreferenceModelError):
        running.with_status("created")
    completed = running.with_status("completed", completed_at="2026-08-18T03:08:10Z")
    with pytest.raises(PreferenceModelError):
        completed.with_status("running")


# --------------------------------------------------------------------- PreferenceStatistics


def test_statistics_from_records_matches_hand_count():
    manifest = make_manifest()
    pairs = [
        make_pair(preference_id="pref_p001_c001__p001_c002"),
        make_pair(
            preference_id="pref_p001_c003__p001_c004",
            chosen_candidate_id="p001_c003",
            rejected_candidate_id="p001_c004",
            duplicate_training_record=True,
            canonical_preference_id="pref_p001_c001__p001_c002",
        ),
    ]
    rejections = [
        make_rejection(candidate_a="p001_c005", candidate_b="p001_c006", reason="tie"),
        make_rejection(
            candidate_a="p001_c007",
            candidate_b="p001_c008",
            reason="identical_code",
            detail="same code",
        ),
    ]
    stats = PreferenceStatistics.from_records(
        manifest, pairs, rejections, candidates_considered=8
    )
    assert stats.pairs_generated == 2
    assert stats.pairs_rejected == 2
    assert stats.candidate_pairs_considered == 4
    assert stats.ties == 1
    assert stats.duplicates == 1
    assert stats.strong_pairs == 2
    assert stats.training_records == 1


def test_statistics_rejects_inconsistent_totals():
    manifest = make_manifest()
    with pytest.raises(PreferenceModelError):
        PreferenceStatistics(
            preference_run_id=PREF_RUN_ID,
            problems_processed=1,
            candidates_considered=2,
            candidate_pairs_considered=1,
            pairs_generated=1,
            pairs_rejected=1,  # 1 + 1 != 1
            ties=0,
            duplicates=0,
            indeterminate=0,
            prompt_mismatches=0,
            integrity_failures=0,
            strong_pairs=1,
            medium_pairs=0,
            training_records=1,
            rejections_by_reason={},
            per_problem={},
            computed_at="2026-08-18T03:08:10Z",
        )


def test_derive_candidates_considered():
    pairs = [make_pair()]
    rejections = [make_rejection(candidate_a="p001_c003", candidate_b="p001_c004")]
    assert derive_candidates_considered(pairs, rejections) == 4


# ---------------------------------------------------------------------------- QualityReport


def test_quality_report_round_trip():
    report = QualityReport(
        preference_run_id=PREF_RUN_ID,
        total_pairs=2,
        strong_pairs=2,
        medium_pairs=0,
        score_margin_distribution={"0.5": 2},
        chosen_pass_rate_distribution={"1.0": 2},
        rejected_pass_rate_distribution={"0.5": 2},
        strategy_distribution={"chosen": {"normal": 2}, "rejected": {"alternative": 2}},
        problems_with_pairs=["p001"],
        problems_without_pairs={"p002": "all_candidates_correct"},
        computed_at="2026-08-18T03:08:10Z",
    )
    restored = QualityReport.from_dict(report.to_dict())
    assert restored == report


def test_quality_report_rejects_overlap_between_with_and_without_pairs():
    with pytest.raises(PreferenceModelError):
        QualityReport(
            preference_run_id=PREF_RUN_ID,
            total_pairs=0,
            strong_pairs=0,
            medium_pairs=0,
            score_margin_distribution={},
            chosen_pass_rate_distribution={},
            rejected_pass_rate_distribution={},
            strategy_distribution={},
            problems_with_pairs=["p001"],
            problems_without_pairs={"p001": "all_candidates_correct"},
            computed_at="2026-08-18T03:08:10Z",
        )
