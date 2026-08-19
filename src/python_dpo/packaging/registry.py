"""``models/registry.json``, the project-root ledger of packaged models (spec 12 sections
45-48, 96).

Lives at the project root rather than under ``data/`` because it is the one artifact meant
to answer "what models exist and are any of them recommended" at a glance, independent of
any single experiment run. Atomic writes reuse :func:`~python_dpo.atomic_io.atomic_write_json`
(CLAUDE.md's Reproducibility rule -- nothing here is ever a partial file).

Nothing is promoted automatically (spec section 48): :meth:`ModelRegistry.register` only
ever writes ``EXPERIMENTAL``, and :meth:`ModelRegistry.promote` refuses ``RECOMMENDED``
unless the caller supplies proof of a passing Stage 10 success-criteria record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..atomic_io import atomic_write_json, read_json
from .errors import RegistryError

REGISTRY_VERSION = "registry_v1"

STATUSES = frozenset({"EXPERIMENTAL", "VALIDATED", "RECOMMENDED", "RETIRED", "REJECTED"})

# From EXPERIMENTAL, a model can be validated, retired, or rejected outright; RECOMMENDED
# is reachable only via VALIDATED, so a model is never recommended sight-unseen.
_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "EXPERIMENTAL": frozenset({"VALIDATED", "RETIRED", "REJECTED"}),
    "VALIDATED": frozenset({"RECOMMENDED", "RETIRED", "REJECTED"}),
    "RECOMMENDED": frozenset({"RETIRED"}),
    "RETIRED": frozenset(),
    "REJECTED": frozenset(),
}

_ENTRY_FIELDS = frozenset(
    {
        "status",
        "package_path",
        "base_model_name",
        "base_model_revision",
        "training_run_id",
        "experiment_run_id",
        "created_at",
        "verification",
        "evaluation_run_id",
        "notes",
    }
)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class RegistryEntry:
    """One row of ``models/registry.json`` (spec section 45)."""

    model_id: str
    status: str
    package_path: str
    base_model_name: str
    training_run_id: str
    created_at: str
    verification: dict[str, Any] = field(default_factory=dict)
    base_model_revision: str | None = None
    experiment_run_id: str | None = None
    evaluation_run_id: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.model_id, "model_id")
        if self.status not in STATUSES:
            raise RegistryError(f"status must be one of {sorted(STATUSES)}, got {self.status!r}")
        _require_text(self.package_path, "package_path")
        _require_text(self.base_model_name, "base_model_name")
        _require_text(self.training_run_id, "training_run_id")
        _require_text(self.created_at, "created_at")
        if not isinstance(self.verification, dict):
            raise RegistryError("verification must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "package_path": self.package_path,
            "base_model_name": self.base_model_name,
            "base_model_revision": self.base_model_revision,
            "training_run_id": self.training_run_id,
            "experiment_run_id": self.experiment_run_id,
            "created_at": self.created_at,
            "verification": dict(self.verification),
            "evaluation_run_id": self.evaluation_run_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, model_id: str, data: Any) -> RegistryEntry:
        if not isinstance(data, dict):
            raise RegistryError(f"registry entry {model_id!r}: expected a JSON object")
        unknown = sorted(set(data) - _ENTRY_FIELDS)
        if unknown:
            raise RegistryError(
                f"registry entry {model_id!r}: unknown field(s): {', '.join(unknown)}"
            )
        missing = sorted(
            {"status", "package_path", "base_model_name", "training_run_id", "created_at"}
            - set(data)
        )
        if missing:
            raise RegistryError(
                f"registry entry {model_id!r}: missing required field(s): {', '.join(missing)}"
            )
        return cls(
            model_id=model_id,
            status=data["status"],
            package_path=data["package_path"],
            base_model_name=data["base_model_name"],
            base_model_revision=data.get("base_model_revision"),
            training_run_id=data["training_run_id"],
            experiment_run_id=data.get("experiment_run_id"),
            created_at=data["created_at"],
            verification=data.get("verification") or {},
            evaluation_run_id=data.get("evaluation_run_id"),
            notes=data.get("notes"),
        )


class ModelRegistry:
    """Owns ``registry.json`` at ``path`` (typically ``<project_root>/models/registry.json``)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read_all(self) -> dict[str, RegistryEntry]:
        if not self.path.is_file():
            return {}
        data = read_json(self.path)
        unknown = sorted(set(data) - {"registry_version", "models"})
        if unknown:
            raise RegistryError(f"{self.path}: unknown top-level key(s): {', '.join(unknown)}")
        models = data.get("models") or {}
        if not isinstance(models, dict):
            raise RegistryError(f"{self.path}: 'models' must be a mapping")
        return {
            model_id: RegistryEntry.from_dict(model_id, value) for model_id, value in models.items()
        }

    def _write_all(self, entries: dict[str, RegistryEntry]) -> None:
        payload = {
            "registry_version": REGISTRY_VERSION,
            "models": {model_id: entry.to_dict() for model_id, entry in sorted(entries.items())},
        }
        atomic_write_json(self.path, payload)

    def register(self, entry: RegistryEntry) -> RegistryEntry:
        """Add a newly packaged model. Always ``EXPERIMENTAL`` (spec section 48) --
        packaging never registers anything at a higher trust level."""
        if entry.status != "EXPERIMENTAL":
            raise RegistryError(
                f"register() only accepts EXPERIMENTAL entries; got {entry.status!r} for "
                f"{entry.model_id!r}. Use promote() to change status after registration."
            )
        entries = self._read_all()
        if entry.model_id in entries:
            raise RegistryError(f"model {entry.model_id!r} is already registered")
        entries[entry.model_id] = entry
        self._write_all(entries)
        return entry

    def get(self, model_id: str) -> RegistryEntry:
        entries = self._read_all()
        try:
            return entries[model_id]
        except KeyError:
            raise RegistryError(f"no registered model {model_id!r} in {self.path}") from None

    def list(self) -> list[RegistryEntry]:
        return sorted(self._read_all().values(), key=lambda e: e.created_at, reverse=True)

    def promote(
        self,
        model_id: str,
        status: str,
        *,
        evaluation_run_id: str | None = None,
        success_criteria_passed: bool = False,
        notes: str | None = None,
    ) -> RegistryEntry:
        """Explicitly change a model's status (spec sections 47, 48).

        ``RECOMMENDED`` requires both an ``evaluation_run_id`` and
        ``success_criteria_passed=True`` -- the caller (the ``model promote`` CLI command)
        is responsible for having actually checked a recorded Stage 10
        ``evaluate_success_criteria`` result before setting that flag; this method refuses
        to take the caller's word for it without at least the evaluation run id on record.
        """
        if status not in STATUSES:
            raise RegistryError(f"status must be one of {sorted(STATUSES)}, got {status!r}")
        entries = self._read_all()
        try:
            entry = entries[model_id]
        except KeyError:
            raise RegistryError(f"no registered model {model_id!r} in {self.path}") from None

        if status not in _STATUS_TRANSITIONS[entry.status]:
            raise RegistryError(
                f"illegal registry status transition for {model_id!r}: "
                f"{entry.status!r} -> {status!r}"
            )
        if status == "RECOMMENDED":
            if not evaluation_run_id:
                raise RegistryError(
                    f"cannot promote {model_id!r} to RECOMMENDED without an evaluation_run_id"
                )
            if not success_criteria_passed:
                raise RegistryError(
                    f"cannot promote {model_id!r} to RECOMMENDED: evaluation run "
                    f"{evaluation_run_id!r} did not pass its recorded success criteria"
                )

        updated = RegistryEntry(
            model_id=entry.model_id,
            status=status,
            package_path=entry.package_path,
            base_model_name=entry.base_model_name,
            base_model_revision=entry.base_model_revision,
            training_run_id=entry.training_run_id,
            experiment_run_id=entry.experiment_run_id,
            created_at=entry.created_at,
            verification=entry.verification,
            evaluation_run_id=evaluation_run_id or entry.evaluation_run_id,
            notes=notes if notes is not None else entry.notes,
        )
        entries[model_id] = updated
        self._write_all(entries)
        return updated


__all__ = ["REGISTRY_VERSION", "STATUSES", "ModelRegistry", "RegistryEntry"]
