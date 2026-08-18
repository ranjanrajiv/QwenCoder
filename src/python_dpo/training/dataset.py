"""Preference dataset loading and validation (spec 09 sections 21-27, 101-106).

Everything here runs **before the model is loaded** (spec section 24): there is no point
spending minutes on 4-bit weights for a dataset that was never trainable.

Two rules are structural rather than merely checked:

* **The test split never reaches the trainer.** :class:`TrainingDataset` exposes ``train``
  and ``validation`` as records, but ``test`` only as a hash, a count, and a problem-id
  set. There is no attribute a caller could accidentally pass to ``DPOTrainer`` (sections
  21, 22) — the test split is reserved for Step 10.
* **``chosen``/``rejected`` are only ever handled as text.** They are validated to *be*
  ``str`` and are never parsed, compiled, or executed (sections 100, 101). This module
  imports no ``ast``, no ``exec``, no ``subprocess``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..candidates.hashing import sha256_text
from .errors import DatasetValidationError

logger = logging.getLogger("python_dpo.training.dataset")

SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}
REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


@dataclass(frozen=True)
class PreferenceRecord:
    """One training example. Three strings, nothing more (spec sections 23, 101)."""

    prompt: str
    chosen: str
    rejected: str

    def to_dict(self) -> dict[str, str]:
        return {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}


@dataclass(frozen=True)
class SplitStatistics:
    """Character-level statistics for one split (spec section 27)."""

    count: int
    mean_prompt_chars: float
    mean_chosen_chars: float
    mean_rejected_chars: float
    max_prompt_chars: int
    max_chosen_chars: int
    max_rejected_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_prompt_chars": round(self.mean_prompt_chars, 2),
            "mean_chosen_chars": round(self.mean_chosen_chars, 2),
            "mean_rejected_chars": round(self.mean_rejected_chars, 2),
            "max_prompt_chars": self.max_prompt_chars,
            "max_chosen_chars": self.max_chosen_chars,
            "max_rejected_chars": self.max_rejected_chars,
        }

    @classmethod
    def from_records(cls, records: list[PreferenceRecord]) -> SplitStatistics:
        if not records:
            return cls(0, 0.0, 0.0, 0.0, 0, 0, 0)
        prompts = [len(r.prompt) for r in records]
        chosen = [len(r.chosen) for r in records]
        rejected = [len(r.rejected) for r in records]
        return cls(
            count=len(records),
            mean_prompt_chars=sum(prompts) / len(prompts),
            mean_chosen_chars=sum(chosen) / len(chosen),
            mean_rejected_chars=sum(rejected) / len(rejected),
            max_prompt_chars=max(prompts),
            max_chosen_chars=max(chosen),
            max_rejected_chars=max(rejected),
        )


@dataclass(frozen=True)
class PreferenceBalance:
    """Score distribution of the source pairs (spec sections 105, 106).

    Read from Stage 8's ``metadata.jsonl``, which carries the evaluation evidence behind
    each pair. Reported so the dataset is understood *before* training, never used to
    alter training.
    """

    pairs: int = 0
    mean_chosen_score: float = 0.0
    mean_rejected_score: float = 0.0
    mean_score_margin: float = 0.0
    strong_pair_percentage: float = 0.0
    medium_pair_percentage: float = 0.0
    chosen_pass_rate_histogram: dict[str, int] = field(default_factory=dict)
    rejected_pass_rate_histogram: dict[str, int] = field(default_factory=dict)
    score_margin_histogram: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "mean_chosen_score": round(self.mean_chosen_score, 6),
            "mean_rejected_score": round(self.mean_rejected_score, 6),
            "mean_score_margin": round(self.mean_score_margin, 6),
            "strong_pair_percentage": round(self.strong_pair_percentage, 2),
            "medium_pair_percentage": round(self.medium_pair_percentage, 2),
            "chosen_pass_rate_histogram": self.chosen_pass_rate_histogram,
            "rejected_pass_rate_histogram": self.rejected_pass_rate_histogram,
            "score_margin_histogram": self.score_margin_histogram,
        }


@dataclass(frozen=True)
class TrainingDataset:
    """A validated preference dataset, ready for the trainer.

    The test split is deliberately represented only by ``test_count``,
    ``split_hashes["test"]`` and ``split_problem_ids["test"]`` — there is no ``test``
    record list, so it cannot be handed to ``DPOTrainer`` even by mistake (sections 21, 22).
    """

    preference_run_id: str
    train: list[PreferenceRecord]
    validation: list[PreferenceRecord]
    test_count: int
    split_hashes: dict[str, str]
    split_counts: dict[str, int]
    split_problem_ids: dict[str, list[str]]
    statistics: dict[str, Any]
    provenance: dict[str, str]
    balance: PreferenceBalance

    @property
    def has_validation(self) -> bool:
        return bool(self.validation)


def _read_split(path: Path, split: str) -> list[PreferenceRecord]:
    """Parse and validate one split file (spec section 24, checks 2-7)."""
    if not path.is_file():
        raise DatasetValidationError(f"{split} split not found at {path}")

    records: list[PreferenceRecord] = []
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise DatasetValidationError(
                f"{path}:{number}: blank line; JSONL must hold one object per line"
            )
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise DatasetValidationError(f"{path}:{number}: expected a JSON object")

        missing = [f for f in REQUIRED_FIELDS if f not in raw]
        if missing:
            raise DatasetValidationError(
                f"{path}:{number}: missing required field(s): {', '.join(missing)}"
            )
        for name in REQUIRED_FIELDS:
            value = raw[name]
            # Spec section 101: these must be ordinary text, never anything the training
            # loop could be tempted to interpret.
            if not isinstance(value, str):
                raise DatasetValidationError(
                    f"{path}:{number}: {name!r} must be a string, got "
                    f"{type(value).__name__}"
                )
            if not value.strip():
                raise DatasetValidationError(f"{path}:{number}: {name!r} must not be empty")
        if raw["chosen"] == raw["rejected"]:
            raise DatasetValidationError(
                f"{path}:{number}: chosen and rejected are identical; there is no "
                "preference to learn"
            )
        records.append(
            PreferenceRecord(
                prompt=raw["prompt"], chosen=raw["chosen"], rejected=raw["rejected"]
            )
        )
    return records


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetValidationError(f"{label} not found at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DatasetValidationError(f"{path}: expected a JSON object")
    return data


def _load_balance(run_dir: Path) -> PreferenceBalance:
    """Spec sections 105/106 from Stage 8's metadata.jsonl. Absent metadata is not fatal."""
    path = run_dir / "metadata.jsonl"
    if not path.is_file():
        return PreferenceBalance()

    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    margins: list[float] = []
    strong = medium = 0
    chosen_hist: dict[str, int] = {}
    rejected_hist: dict[str, int] = {}
    margin_hist: dict[str, int] = {}

    def bucket(value: float) -> str:
        return f"{round(value, 1):.1f}"

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return PreferenceBalance()
        if not isinstance(row, dict):
            continue
        chosen_scores.append(float(row.get("chosen_score", 0.0)))
        rejected_scores.append(float(row.get("rejected_score", 0.0)))
        margins.append(float(row.get("score_margin", 0.0)))
        if row.get("preference_strength") == "strong":
            strong += 1
        elif row.get("preference_strength") == "medium":
            medium += 1
        key = bucket(float(row.get("chosen_pass_rate", 0.0)))
        chosen_hist[key] = chosen_hist.get(key, 0) + 1
        key = bucket(float(row.get("rejected_pass_rate", 0.0)))
        rejected_hist[key] = rejected_hist.get(key, 0) + 1
        key = bucket(float(row.get("score_margin", 0.0)))
        margin_hist[key] = margin_hist.get(key, 0) + 1

    total = len(chosen_scores)
    if total == 0:
        return PreferenceBalance()
    return PreferenceBalance(
        pairs=total,
        mean_chosen_score=sum(chosen_scores) / total,
        mean_rejected_score=sum(rejected_scores) / total,
        mean_score_margin=sum(margins) / total,
        strong_pair_percentage=100.0 * strong / total,
        medium_pair_percentage=100.0 * medium / total,
        chosen_pass_rate_histogram=chosen_hist,
        rejected_pass_rate_histogram=rejected_hist,
        score_margin_histogram=margin_hist,
    )


def load_training_dataset(
    preference_run_dir: Path,
    *,
    allow_small_dataset: bool = False,
    min_training_pairs: int = 500,
) -> TrainingDataset:
    """Load, validate, hash and describe a Stage 8 preference dataset.

    ``allow_small_dataset`` relaxes exactly one rule — spec section 24's requirement that
    the **validation** split be non-empty. It is needed because Stage 8's problem-level
    splitter floors validation at ``floor(n_problems x 0.1)``, which is zero for every
    pool below ten pair-bearing problems; without the escape hatch, most datasets this
    project can currently produce would be permanently untrainable. An empty *training*
    split is always fatal — there would be nothing to learn from.
    """
    preference_run_dir = Path(preference_run_dir)
    if not preference_run_dir.is_dir():
        raise DatasetValidationError(f"preference run not found at {preference_run_dir}")

    manifest = _read_json(preference_run_dir / "manifest.json", "preference manifest")
    split_manifest = _read_json(preference_run_dir / "split_manifest.json", "split manifest")

    splits: dict[str, list[PreferenceRecord]] = {}
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    statistics: dict[str, Any] = {}
    for split, filename in SPLIT_FILENAMES.items():
        path = preference_run_dir / filename
        records = _read_split(path, split)
        splits[split] = records
        # Hash the file bytes, not the parsed records: reproducing a run means proving
        # the same file was consumed (spec section 26).
        hashes[split] = sha256_text(path.read_text(encoding="utf-8"))
        counts[split] = len(records)
        statistics[split] = SplitStatistics.from_records(records).to_dict()

    if not splits["train"]:
        raise DatasetValidationError(
            "the training split is empty; there is nothing to train on"
        )

    if not splits["validation"]:
        message = (
            "the validation split is empty. Stage 8's problem-level splitter yields no "
            "validation problems below ten pair-bearing problems"
        )
        if not allow_small_dataset:
            raise DatasetValidationError(
                f"{message}; pass --allow-small-dataset to train without evaluation, or "
                "generate a dataset whose split ratios leave a validation problem"
            )
        logger.warning("%s; continuing with evaluation disabled", message)

    problem_ids = {
        "train": list(split_manifest.get("train_problem_ids", [])),
        "validation": list(split_manifest.get("validation_problem_ids", [])),
        "test": list(split_manifest.get("test_problem_ids", [])),
    }
    _assert_splits_disjoint(problem_ids)

    if counts["train"] < min_training_pairs:
        # Spec section 103: warn, do not block. At the current dataset scale this always
        # fires, and that is the correct signal rather than a nuisance.
        logger.warning(
            "Training dataset contains only %d preference pair(s), below the configured "
            "minimum of %d. This is suitable for pipeline validation but not for "
            "meaningful model adaptation.",
            counts["train"],
            min_training_pairs,
        )

    provenance = {
        "preference_run_id": manifest.get("preference_run_id", ""),
        "preference_version": manifest.get("preference_version", ""),
        "selection_policy": manifest.get("selection_policy", ""),
        "selection_policy_version": manifest.get("selection_policy_version", ""),
        "dataset_schema_version": manifest.get("dataset_schema_version", ""),
        "ranking_run_id": manifest.get("ranking_run_id", ""),
        "evaluation_run_id": manifest.get("evaluation_run_id", ""),
        "candidate_run_id": manifest.get("candidate_run_id", ""),
    }
    missing = sorted(k for k, v in provenance.items() if not v)
    if missing:
        raise DatasetValidationError(
            "preference manifest is missing provenance field(s): " + ", ".join(missing)
        )

    return TrainingDataset(
        preference_run_id=provenance["preference_run_id"],
        train=splits["train"],
        validation=splits["validation"],
        test_count=counts["test"],
        split_hashes=hashes,
        split_counts=counts,
        split_problem_ids=problem_ids,
        statistics=statistics,
        provenance=provenance,
        balance=_load_balance(preference_run_dir),
    )


def _assert_splits_disjoint(problem_ids: dict[str, list[str]]) -> None:
    """Spec section 102: training aborts if a problem appears in more than one split."""
    seen: dict[str, str] = {}
    for split, ids in problem_ids.items():
        for problem_id in ids:
            if problem_id in seen and seen[problem_id] != split:
                raise DatasetValidationError(
                    f"problem {problem_id!r} appears in both the {seen[problem_id]!r} and "
                    f"{split!r} splits; train/validation/test must be problem-disjoint"
                )
            seen[problem_id] = split


__all__ = [
    "REQUIRED_FIELDS",
    "SPLIT_FILENAMES",
    "PreferenceBalance",
    "PreferenceRecord",
    "SplitStatistics",
    "TrainingDataset",
    "load_training_dataset",
]
