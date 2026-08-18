"""Tests for PreferenceRepository (spec 08 section 82)."""

from __future__ import annotations

from python_dpo.preferences.repository import PreferenceRepository, PreferenceStoreError
from python_dpo.preferences.splitter import ProblemSplitter

from .test_models import make_pair, make_rejection


def test_save_and_load_pairs_round_trip(tmp_path):
    repo = PreferenceRepository(tmp_path)
    pair = make_pair()
    repo.save(pair)
    assert repo.load_pairs() == [pair]


def test_save_and_load_rejections_round_trip(tmp_path):
    repo = PreferenceRepository(tmp_path)
    rejection = make_rejection()
    repo.save_rejection(rejection)
    assert repo.load_rejections() == [rejection]


def test_get_and_exists(tmp_path):
    repo = PreferenceRepository(tmp_path)
    pair = make_pair()
    repo.save(pair)
    assert repo.get(pair.preference_id) == pair
    assert repo.exists(pair.preference_id)
    assert repo.get("nonexistent") is None
    assert not repo.exists("nonexistent")


def test_list_and_count(tmp_path):
    repo = PreferenceRepository(tmp_path)
    a = make_pair(preference_id="pref_p001_c001__p001_c002")
    b = make_pair(
        preference_id="pref_p001_c003__p001_c004",
        chosen_candidate_id="p001_c003",
        rejected_candidate_id="p001_c004",
    )
    repo.save_pairs([a, b])
    assert repo.count() == 2
    assert {p.preference_id for p in repo.list()} == {a.preference_id, b.preference_id}


def test_list_by_problem(tmp_path):
    repo = PreferenceRepository(tmp_path)
    p1 = make_pair(preference_id="pref_p001_c001__p001_c002", problem_id="p001")
    p2 = make_pair(
        preference_id="pref_p002_c001__p002_c002",
        problem_id="p002",
        chosen_candidate_id="p002_c001",
        rejected_candidate_id="p002_c002",
    )
    repo.save_pairs([p1, p2])
    assert repo.list_by_problem("p001") == [p1]
    assert repo.list_by_problem("p002") == [p2]
    assert repo.list_by_problem("p003") == []


def test_paired_problem_ids_counts_pairs_and_rejections(tmp_path):
    repo = PreferenceRepository(tmp_path)
    repo.save(make_pair(problem_id="p001"))
    repo.save_rejection(make_rejection(problem_id="p002"))
    assert repo.paired_problem_ids() == {"p001", "p002"}


def test_malformed_line_reports_a_line_number(tmp_path):
    repo = PreferenceRepository(tmp_path)
    repo.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    repo.metadata_path.write_text('{"preference_id": "bad"}\n', encoding="utf-8")
    try:
        repo.load_pairs()
        raise AssertionError("expected PreferenceStoreError")
    except PreferenceStoreError as exc:
        assert ":1:" in str(exc)


def test_write_dataset_produces_three_key_training_records(tmp_path):
    repo = PreferenceRepository(tmp_path)
    pair = make_pair(problem_id="p001")
    repo.save(pair)
    split_manifest = ProblemSplitter(seed=42).split(["p001"])
    repo.write_dataset(split_manifest)

    import json

    lines = repo.preferences_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {"prompt", "chosen", "rejected"}

    train_lines = repo.train_path.read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 1


def test_write_dataset_excludes_duplicate_training_records(tmp_path):
    repo = PreferenceRepository(tmp_path)
    survivor = make_pair(preference_id="pref_p001_c001__p001_c002", problem_id="p001")
    duplicate = make_pair(
        preference_id="pref_p001_c003__p001_c004",
        problem_id="p001",
        chosen_candidate_id="p001_c003",
        rejected_candidate_id="p001_c004",
        duplicate_training_record=True,
        canonical_preference_id=survivor.preference_id,
    )
    repo.save_pairs([survivor, duplicate])
    split_manifest = ProblemSplitter(seed=42).split(["p001"])
    repo.write_dataset(split_manifest)

    lines = repo.preferences_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # the duplicate is excluded from the training file
