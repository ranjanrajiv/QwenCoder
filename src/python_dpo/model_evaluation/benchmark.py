"""The held-out evaluation benchmark (spec sections 6-10, 91, decision 3 of the plan).

``problems.jsonl`` stays the single source of truth for problem content; the benchmark
manifest references problem ids and hashes their canonical content rather than
snapshotting it, so drift is detected (spec section 10) rather than prevented by a second
copy that could silently disagree with the first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..candidates.hashing import sha256_text
from ..problems.models import Problem
from .errors import BenchmarkError, BenchmarkLeakageError

MANIFEST_FILENAME = "manifest.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_problems_json(problems: list[Problem]) -> str:
    """Deterministic JSON for a set of problems, independent of input order."""
    ordered = sorted(problems, key=lambda p: p.id)
    return json.dumps([p.to_dict() for p in ordered], sort_keys=True, ensure_ascii=False)


def compute_dataset_hash(problems: list[Problem]) -> str:
    """SHA-256 over the selected problems' canonical content (spec section 10)."""
    return sha256_text(_canonical_problems_json(problems))


@dataclass(frozen=True)
class BenchmarkManifest:
    """``benchmarks/<name>/manifest.json`` (spec section 9)."""

    benchmark_version: str
    problem_ids: tuple[str, ...]
    problem_count: int
    creation_date: str
    dataset_hash: str
    source_dataset_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_version, str) or not self.benchmark_version.strip():
            raise BenchmarkError("benchmark_version must be a non-empty string")
        if isinstance(self.problem_ids, (str, bytes)) or not isinstance(
            self.problem_ids, (list, tuple)
        ):
            raise BenchmarkError("problem_ids must be a sequence of problem ids")
        object.__setattr__(self, "problem_ids", tuple(self.problem_ids))
        if not self.problem_ids:
            raise BenchmarkError("problem_ids must not be empty")
        if len(set(self.problem_ids)) != len(self.problem_ids):
            raise BenchmarkError("problem_ids must not contain duplicates")
        if self.problem_count != len(self.problem_ids):
            raise BenchmarkError(
                f"problem_count ({self.problem_count}) does not match the number of "
                f"problem_ids ({len(self.problem_ids)})"
            )
        if not isinstance(self.creation_date, str) or not self.creation_date.strip():
            raise BenchmarkError("creation_date must be a non-empty string")
        if not isinstance(self.dataset_hash, str) or not self.dataset_hash.strip():
            raise BenchmarkError("dataset_hash must be a non-empty string")
        if not isinstance(self.source_dataset_version, str) or not self.source_dataset_version.strip():
            raise BenchmarkError("source_dataset_version must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "problem_ids": list(self.problem_ids),
            "problem_count": self.problem_count,
            "creation_date": self.creation_date,
            "dataset_hash": self.dataset_hash,
            "source_dataset_version": self.source_dataset_version,
        }

    @classmethod
    def from_dict(cls, data: Any) -> BenchmarkManifest:
        if not isinstance(data, dict):
            raise BenchmarkError("benchmark manifest: expected a JSON object")
        required = {
            "benchmark_version", "problem_ids", "problem_count", "creation_date",
            "dataset_hash", "source_dataset_version",
        }
        missing = sorted(required - set(data))
        if missing:
            raise BenchmarkError(f"benchmark manifest: missing field(s): {', '.join(missing)}")
        unknown = sorted(set(data) - required)
        if unknown:
            raise BenchmarkError(f"benchmark manifest: unknown field(s): {', '.join(unknown)}")
        return cls(
            benchmark_version=data["benchmark_version"],
            problem_ids=tuple(data["problem_ids"]),
            problem_count=data["problem_count"],
            creation_date=data["creation_date"],
            dataset_hash=data["dataset_hash"],
            source_dataset_version=data["source_dataset_version"],
        )


@dataclass(frozen=True)
class Benchmark:
    """A benchmark manifest resolved against the live problem dataset (spec section 6).

    ``problems`` never exposes anything a runner should not see: it is a tuple of
    :class:`~python_dpo.problems.models.Problem`, and it is the *caller's* responsibility
    (spec section 6) to build prompts from ``problem.prompt``/``problem.signature`` only --
    ``reference_solution`` is never read by :mod:`python_dpo.model_evaluation.generation`.
    """

    manifest: BenchmarkManifest
    problems: tuple[Problem, ...]

    @property
    def benchmark_version(self) -> str:
        return self.manifest.benchmark_version

    @property
    def problem_ids(self) -> tuple[str, ...]:
        return self.manifest.problem_ids

    def __len__(self) -> int:
        return len(self.problems)


def build_benchmark(
    name: str, problems: list[Problem], problem_ids: list[str]
) -> BenchmarkManifest:
    """Build a benchmark manifest over exactly ``problem_ids`` (spec sections 8, 9)."""
    if not problem_ids:
        raise BenchmarkError("a benchmark must select at least one problem id")
    if len(set(problem_ids)) != len(problem_ids):
        raise BenchmarkError("problem_ids must not contain duplicates")

    by_id = {p.id: p for p in problems}
    missing = sorted(set(problem_ids) - set(by_id))
    if missing:
        raise BenchmarkError(f"unknown problem id(s): {', '.join(missing)}")

    selected = [by_id[pid] for pid in problem_ids]
    versions = {p.dataset_version for p in selected}
    if len(versions) != 1:
        raise BenchmarkError(
            f"selected problems span multiple dataset versions: {sorted(versions)}"
        )

    return BenchmarkManifest(
        benchmark_version=name,
        problem_ids=tuple(sorted(problem_ids)),
        problem_count=len(problem_ids),
        creation_date=_utc_now_iso(),
        dataset_hash=compute_dataset_hash(selected),
        source_dataset_version=versions.pop(),
    )


def benchmark_dir(benchmarks_root: Path, name: str) -> Path:
    return Path(benchmarks_root) / name


def save_benchmark(benchmarks_root: Path, manifest: BenchmarkManifest) -> Path:
    """Persist ``benchmarks/<name>/manifest.json`` (spec section 8)."""
    path = benchmark_dir(benchmarks_root, manifest.benchmark_version) / MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest.to_dict(), sort_keys=True, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def load_benchmark(benchmarks_root: Path, name: str, problems: list[Problem]) -> Benchmark:
    """Load a benchmark manifest and resolve it against the live dataset, verifying the
    dataset hash has not drifted (spec sections 10, 91).
    """
    path = benchmark_dir(benchmarks_root, name) / MANIFEST_FILENAME
    if not path.is_file():
        raise BenchmarkError(f"no benchmark {name!r} at {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path}: invalid JSON: {exc}") from exc
    manifest = BenchmarkManifest.from_dict(raw)

    by_id = {p.id: p for p in problems}
    missing = sorted(set(manifest.problem_ids) - set(by_id))
    if missing:
        raise BenchmarkError(
            f"benchmark {name!r} references problem id(s) no longer in the dataset: "
            f"{', '.join(missing)}"
        )
    selected = [by_id[pid] for pid in manifest.problem_ids]

    current_hash = compute_dataset_hash(selected)
    if current_hash != manifest.dataset_hash:
        raise BenchmarkError(
            f"benchmark {name!r} has drifted: the referenced problems' content no longer "
            f"matches the recorded hash (expected {manifest.dataset_hash}, got "
            f"{current_hash}). The benchmark must not change between model comparisons "
            "(spec section 91); create a new benchmark_version instead"
        )

    return Benchmark(manifest=manifest, problems=tuple(selected))


def check_leakage(benchmark: Benchmark, split_manifest: dict[str, Any]) -> None:
    """Assert the benchmark is disjoint from a preference run's train/validation splits
    (spec section 7). Raises :class:`BenchmarkLeakageError` naming every offending id --
    a contaminated benchmark must stop the run, not warn.
    """
    train_ids = set(split_manifest.get("train_problem_ids", []))
    validation_ids = set(split_manifest.get("validation_problem_ids", []))
    benchmark_ids = set(benchmark.problem_ids)

    overlap_train = sorted(benchmark_ids & train_ids)
    overlap_validation = sorted(benchmark_ids & validation_ids)
    if overlap_train or overlap_validation:
        parts = []
        if overlap_train:
            parts.append(f"train split: {', '.join(overlap_train)}")
        if overlap_validation:
            parts.append(f"validation split: {', '.join(overlap_validation)}")
        raise BenchmarkLeakageError(
            f"benchmark {benchmark.benchmark_version!r} overlaps the preference "
            f"{'; '.join(parts)}. The evaluation benchmark must be completely independent "
            "of every problem used to create DPO preference pairs"
        )


__all__ = [
    "MANIFEST_FILENAME",
    "Benchmark",
    "BenchmarkManifest",
    "benchmark_dir",
    "build_benchmark",
    "check_leakage",
    "compute_dataset_hash",
    "load_benchmark",
    "save_benchmark",
]
