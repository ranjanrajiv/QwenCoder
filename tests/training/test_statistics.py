"""Tests for the training formatters and metrics recorder (sections 19, 27, 34, 78, 94)."""

from __future__ import annotations

import json

from python_dpo.training.callbacks import JsonlMetricsRecorder
from python_dpo.training.dataset import PreferenceBalance, PreferenceRecord, TrainingDataset
from python_dpo.training.lengths import analyze_lengths
from python_dpo.training.loader import ParameterCounts
from python_dpo.training.statistics import (
    format_dataset_statistics,
    format_final_report,
    format_length_analysis,
    format_parameter_counts,
)

from .test_models import make_report


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())


def make_dataset() -> TrainingDataset:
    record = PreferenceRecord(prompt="a b c", chosen="d e", rejected="f g h")
    return TrainingDataset(
        preference_run_id="pref_x",
        train=[record],
        validation=[record],
        test_count=2,
        split_hashes={"train": "a", "validation": "b", "test": "c"},
        split_counts={"train": 1, "validation": 1, "test": 2},
        split_problem_ids={"train": ["p007"], "validation": ["p010"], "test": ["p004"]},
        statistics={
            "train": {
                "count": 1,
                "mean_prompt_chars": 5.0,
                "mean_chosen_chars": 3.0,
                "mean_rejected_chars": 5.0,
                "max_prompt_chars": 5,
                "max_chosen_chars": 3,
                "max_rejected_chars": 5,
            },
            "validation": {"count": 0},
        },
        provenance={
            "preference_run_id": "pref_x",
            "selection_policy": "all_better",
            "selection_policy_version": "all_better_v1",
            "ranking_run_id": "rank_x",
            "evaluation_run_id": "eval_x",
            "candidate_run_id": "run_x",
            "preference_version": "v1",
            "dataset_schema_version": "dpo_preference_v1",
        },
        balance=PreferenceBalance(
            pairs=2,
            mean_chosen_score=0.9,
            mean_rejected_score=0.6,
            mean_score_margin=0.3,
            strong_pair_percentage=50.0,
            medium_pair_percentage=50.0,
        ),
    )


def test_dataset_statistics_names_the_held_out_test_split():
    text = format_dataset_statistics(make_dataset())
    assert "pref_x" in text
    assert "held out, never trained on" in text
    assert "all_better" in text
    assert "mean score margin" in text


def test_length_analysis_renders_percentiles():
    analysis = analyze_lengths(
        [PreferenceRecord(prompt="a b", chosen="c", rejected="d")],
        WordTokenizer(),
        max_length=100,
        max_prompt_length=50,
    )
    text = format_length_analysis(analysis)
    assert "P50" in text
    assert "prompt_chosen" in text
    assert "Truncation:" in text


def test_parameter_counts_render_with_separators():
    text = format_parameter_counts(ParameterCounts(total=1_706_045_440, trainable=7_372_800))
    assert "1,706,045,440" in text
    assert "7,372,800" in text
    assert "0.4322%" in text


def test_final_report_includes_reward_metrics_and_reload_status():
    report = make_report(
        final_train_loss=0.69,
        reward_metrics={"rewards/margins": 0.015},
        adapter_reload_ok=True,
        adapter_path="/tmp/adapter",
    )
    text = format_final_report(report)
    assert "rewards/margins" in text
    assert "Adapter reload: OK" in text


def test_final_report_flags_an_unverified_adapter():
    assert "NOT VERIFIED" in format_final_report(make_report(adapter_reload_ok=False))


# ------------------------------------------------------------------ metrics recorder


def test_recorder_appends_one_row_per_log(tmp_path):
    recorder = JsonlMetricsRecorder(tmp_path / "metrics.jsonl")
    recorder.record({"loss": 0.7}, step=1, epoch=1.0)
    recorder.record({"loss": 0.6}, step=2, epoch=2.0)
    rows = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["loss"] == 0.6
    assert rows[1]["step"] == 2


def test_recorder_passes_through_unknown_reward_metrics(tmp_path):
    """Spec section 78: metric names vary by TRL version, so they are not enumerated."""
    recorder = JsonlMetricsRecorder(tmp_path / "metrics.jsonl")
    recorder.record({"rewards/some_future_metric": 1.5}, step=1)
    assert recorder.reward_metrics() == {"rewards/some_future_metric": 1.5}


def test_recorder_drops_non_numeric_values(tmp_path):
    """Spec section 55: candidate code must never reach the metrics file."""
    recorder = JsonlMetricsRecorder(tmp_path / "metrics.jsonl")
    row = recorder.record({"loss": 0.5, "sample_code": "def f(): ..."}, step=1)
    assert "sample_code" not in row
    assert "def f()" not in (tmp_path / "metrics.jsonl").read_text()


def test_final_metrics_keeps_the_last_value_of_each_key(tmp_path):
    recorder = JsonlMetricsRecorder(tmp_path / "metrics.jsonl")
    recorder.record({"loss": 0.9}, step=1)
    recorder.record({"loss": 0.4, "eval_loss": 0.5}, step=2)
    final = recorder.final_metrics()
    assert final["loss"] == 0.4
    assert final["eval_loss"] == 0.5
