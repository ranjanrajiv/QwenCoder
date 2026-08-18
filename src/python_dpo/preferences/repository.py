"""Durable, run-scoped persistence for preference pairs, rejections, and the derived
dataset JSONL files.

Mirrors :class:`python_dpo.ranking.repository.RankingRepository`: one instance owns
exactly one preference run directory, records are written the moment they exist via
fsynced :mod:`python_dpo.atomic_io` appends, and ``metadata.jsonl``/``rejections.jsonl``
are never rewritten — a killed run leaves a usable, resumable file behind (spec section
83). ``metadata.jsonl`` is both the durable, append-as-computed ledger of every generated
:class:`~python_dpo.preferences.models.PreferencePair` *and* the spec section 53 audit
artifact; nothing else in this package holds a pair record that ``metadata.jsonl`` does
not already have.

The three derived training files (``preferences.jsonl``, ``train.jsonl``,
``validation.jsonl``, ``test.jsonl``) are different in kind: they are always fully rebuilt
together from ``metadata.jsonl`` once building completes, so they are written as whole-file
atomic replacements rather than incremental appends — a reader must never observe a
training file that reflects only part of a run (spec section 82: "do not expose filesystem
operations outside the repository").
"""

from __future__ import annotations

import json
from pathlib import Path

from ..atomic_io import JsonlError, append_jsonl, iter_jsonl
from .errors import PreferenceStoreError
from .models import PreferenceModelError, PreferencePair, PreferenceRejection
from .splitter import SplitManifest

METADATA_FILENAME = "metadata.jsonl"
REJECTIONS_FILENAME = "rejections.jsonl"
PREFERENCES_FILENAME = "preferences.jsonl"
TRAIN_FILENAME = "train.jsonl"
VALIDATION_FILENAME = "validation.jsonl"
TEST_FILENAME = "test.jsonl"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"
QUALITY_REPORT_FILENAME = "quality_report.json"


def _load(path: Path, build) -> list:
    """Parse a JSONL file, validating every record. Never silently skips a bad line
    (CLAUDE.md Data Integrity), matching ``ranking/repository.py``'s ``_load``.
    """
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        raise PreferenceStoreError(str(exc)) from exc

    records = []
    for number, raw in raw_records:
        try:
            records.append(build(raw))
        except PreferenceModelError as exc:
            raise PreferenceStoreError(f"{path}:{number}: {exc}") from exc
    return records


class PreferenceRepository:
    """Reads and appends the ``metadata``/``rejections`` artifacts of one preference run,
    and writes the derived dataset files built from them (spec section 82).
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.metadata_path = self.directory / METADATA_FILENAME
        self.rejections_path = self.directory / REJECTIONS_FILENAME
        self.preferences_path = self.directory / PREFERENCES_FILENAME
        self.train_path = self.directory / TRAIN_FILENAME
        self.validation_path = self.directory / VALIDATION_FILENAME
        self.test_path = self.directory / TEST_FILENAME
        self.split_manifest_path = self.directory / SPLIT_MANIFEST_FILENAME
        self.quality_report_path = self.directory / QUALITY_REPORT_FILENAME

    # --------------------------------------------------------------------------- write

    def save(self, pair: PreferencePair) -> None:
        append_jsonl(self.metadata_path, pair.to_dict())

    def save_pairs(self, pairs: list[PreferencePair]) -> None:
        for pair in pairs:
            self.save(pair)

    def save_rejection(self, rejection: PreferenceRejection) -> None:
        append_jsonl(self.rejections_path, rejection.to_dict())

    def save_rejections(self, rejections: list[PreferenceRejection]) -> None:
        for rejection in rejections:
            self.save_rejection(rejection)

    # ---------------------------------------------------------------------------- read

    def load_pairs(self) -> list[PreferencePair]:
        return _load(self.metadata_path, PreferencePair.from_dict)

    def load_rejections(self) -> list[PreferenceRejection]:
        return _load(self.rejections_path, PreferenceRejection.from_dict)

    # ------------------------------------------------------------------ spec section 82

    def get(self, preference_id: str) -> PreferencePair | None:
        for pair in self.load_pairs():
            if pair.preference_id == preference_id:
                return pair
        return None

    def exists(self, preference_id: str) -> bool:
        return self.get(preference_id) is not None

    def list(self) -> list[PreferencePair]:
        return self.load_pairs()

    def list_by_problem(self, problem_id: str) -> list[PreferencePair]:
        return [p for p in self.load_pairs() if p.problem_id == problem_id]

    def count(self) -> int:
        return len(self.load_pairs())

    # ------------------------------------------------------------------------- indexes

    def paired_problem_ids(self) -> set[str]:
        """Problem ids already settled by this preference run — the resume index (spec
        section 83). A problem is settled once it has at least one persisted pair *or*
        rejection, since a problem can legitimately produce zero pairs (spec section 76).
        """
        return {p.problem_id for p in self.load_pairs()} | {
            r.problem_id for r in self.load_rejections()
        }

    # --------------------------------------------------------------- derived dataset files

    def write_dataset(self, split_manifest: SplitManifest) -> None:
        """(Re)write ``preferences.jsonl`` and the three split files from the pairs
        already durably persisted in ``metadata.jsonl`` (spec sections 52, 64) — a
        whole-file, atomic-by-replacement write (never a partial JSONL append), since
        these files are always fully rebuilt together.
        """
        pairs = self.load_pairs()
        training_pairs = [pair for pair in pairs if not pair.duplicate_training_record]
        _write_jsonl_atomic(
            self.preferences_path, (pair.training_record() for pair in training_pairs)
        )

        by_split: dict[str, list[PreferencePair]] = {"train": [], "validation": [], "test": []}
        for pair in training_pairs:
            split = split_manifest.split_of(pair.problem_id)
            if split is not None:
                by_split[split].append(pair)

        _write_jsonl_atomic(
            self.train_path, (pair.training_record() for pair in by_split["train"])
        )
        _write_jsonl_atomic(
            self.validation_path, (pair.training_record() for pair in by_split["validation"])
        )
        _write_jsonl_atomic(self.test_path, (pair.training_record() for pair in by_split["test"]))


def _write_jsonl_atomic(path: Path, records) -> None:
    """Write ``records`` to ``path`` as JSONL in one atomic replace, so a reader never
    observes a partially-written derived dataset file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    tmp_path.replace(path)


__all__ = [
    "METADATA_FILENAME",
    "PREFERENCES_FILENAME",
    "QUALITY_REPORT_FILENAME",
    "REJECTIONS_FILENAME",
    "SPLIT_MANIFEST_FILENAME",
    "TEST_FILENAME",
    "TRAIN_FILENAME",
    "VALIDATION_FILENAME",
    "PreferenceRepository",
]
