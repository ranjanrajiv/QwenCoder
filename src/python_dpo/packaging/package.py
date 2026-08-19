"""Building and reading a loadable model package (spec 12 sections 35-37, 96).

A package is ``adapter/`` (the trained LoRA weights) + ``tokenizer/`` + ``manifest.json``,
all under one directory. The **base model is referenced by name and revision, never
copied** (spec section 36) -- it is tens of gigabytes and already lives in the Hugging Face
cache or is fetched on demand; duplicating it per package would make every experiment run
carry its own multi-gigabyte copy for no benefit.

Named ``python_dpo.packaging`` rather than living under ``python_dpo.models`` because that
package name is already taken by the model-*client* protocol (``ModelClient``,
``QwenModelClient``) -- a deliberate deviation from spec section 96's literal path, flagged
in the Stage 12 plan.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..atomic_io import atomic_write_json, read_json
from .errors import PackagingError

MODEL_PACKAGE_VERSION = "model_package_v1"
ADAPTER_DIRNAME = "adapter"
TOKENIZER_DIRNAME = "tokenizer"
MANIFEST_FILENAME = "manifest.json"

_MANIFEST_FIELDS = frozenset(
    {
        "package_version",
        "base_model_name",
        "base_model_revision",
        "training_run_id",
        "experiment_run_id",
        "adapter_path",
        "tokenizer_path",
        "quantization",
        "created_at",
    }
)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PackagingError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ModelPackage:
    """A packaged, loadable base-model-plus-adapter (spec section 37's manifest fields).

    ``root`` is the package's own directory on disk -- not part of the serialized
    manifest, since it is meaningless outside the filesystem that holds it.
    """

    root: Path
    base_model_name: str
    training_run_id: str
    base_model_revision: str | None = None
    experiment_run_id: str | None = None
    adapter_path: str = ADAPTER_DIRNAME
    tokenizer_path: str = TOKENIZER_DIRNAME
    quantization: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    package_version: str = MODEL_PACKAGE_VERSION

    def __post_init__(self) -> None:
        _require_text(self.base_model_name, "base_model_name")
        _require_text(self.training_run_id, "training_run_id")
        _require_text(self.adapter_path, "adapter_path")
        _require_text(self.tokenizer_path, "tokenizer_path")
        if not isinstance(self.quantization, dict):
            raise PackagingError("quantization must be a mapping")

    @property
    def adapter_dir(self) -> Path:
        return self.root / self.adapter_path

    @property
    def tokenizer_dir(self) -> Path:
        return self.root / self.tokenizer_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_version": self.package_version,
            "base_model_name": self.base_model_name,
            "base_model_revision": self.base_model_revision,
            "training_run_id": self.training_run_id,
            "experiment_run_id": self.experiment_run_id,
            "adapter_path": self.adapter_path,
            "tokenizer_path": self.tokenizer_path,
            "quantization": dict(self.quantization),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, root: Path, data: Any) -> ModelPackage:
        if not isinstance(data, dict):
            raise PackagingError(f"{root}: model package manifest must be a JSON object")
        unknown = sorted(set(data) - _MANIFEST_FIELDS)
        if unknown:
            raise PackagingError(f"{root}: unknown manifest field(s): {', '.join(unknown)}")
        missing = sorted({"base_model_name", "training_run_id"} - set(data))
        if missing:
            raise PackagingError(f"{root}: missing required field(s): {', '.join(missing)}")
        return cls(
            root=Path(root),
            base_model_name=data["base_model_name"],
            base_model_revision=data.get("base_model_revision"),
            training_run_id=data["training_run_id"],
            experiment_run_id=data.get("experiment_run_id"),
            adapter_path=data.get("adapter_path", ADAPTER_DIRNAME),
            tokenizer_path=data.get("tokenizer_path", TOKENIZER_DIRNAME),
            quantization=data.get("quantization") or {},
            created_at=data.get("created_at", ""),
            package_version=data.get("package_version", MODEL_PACKAGE_VERSION),
        )

    @classmethod
    def load(cls, root: Path) -> ModelPackage:
        root = Path(root)
        manifest_path = root / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise PackagingError(f"no model package at {root} (missing {MANIFEST_FILENAME})")
        package = cls.from_dict(root, read_json(manifest_path))
        if not package.adapter_dir.is_dir():
            raise PackagingError(f"{root}: adapter directory {package.adapter_dir} is missing")
        return package


def build_package(
    *,
    dest_dir: Path,
    training_run_id: str,
    base_model_name: str,
    base_model_revision: str | None,
    adapter_source: Path,
    tokenizer_source: Path,
    quantization: dict[str, Any],
    created_at: str,
    experiment_run_id: str | None = None,
) -> ModelPackage:
    """Assemble ``dest_dir`` into a loadable package (spec sections 35, 37).

    Copies the trained adapter and tokenizer out of their canonical training-run
    directories; never touches or copies the base model itself (section 36).
    """
    adapter_source = Path(adapter_source)
    if not adapter_source.is_dir() or not any(adapter_source.iterdir()):
        raise PackagingError(
            f"training run {training_run_id!r} has no adapter at {adapter_source}"
        )

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    adapter_dest = dest_dir / ADAPTER_DIRNAME
    if adapter_dest.exists():
        shutil.rmtree(adapter_dest)
    # Skip TRL's frozen reference adapter ("ref/", ~2x the trained adapter's own size,
    # never loaded by anything in this codebase -- AdapterModelRunner loads the tokenizer
    # from the base model by name, never from the adapter directory) and the tokenizer
    # snapshot TRL duplicates alongside the adapter -- this package already carries its
    # own tokenizer/ directory. Mirrors the same exclusion training runs already apply
    # (see .gitignore's data/training/runs/*/adapter/ref/ and .../tokenizer.json rules);
    # without it a package that should be ~15 MB copies out at ~55 MB of dead weight.
    shutil.copytree(
        adapter_source, adapter_dest, ignore=shutil.ignore_patterns("ref", "tokenizer.json")
    )

    tokenizer_source = Path(tokenizer_source)
    tokenizer_dest = dest_dir / TOKENIZER_DIRNAME
    if tokenizer_dest.exists():
        shutil.rmtree(tokenizer_dest)
    if tokenizer_source.is_dir() and any(tokenizer_source.iterdir()):
        shutil.copytree(tokenizer_source, tokenizer_dest)
    else:
        tokenizer_dest.mkdir(parents=True, exist_ok=True)

    package = ModelPackage(
        root=dest_dir,
        base_model_name=base_model_name,
        base_model_revision=base_model_revision,
        training_run_id=training_run_id,
        experiment_run_id=experiment_run_id,
        quantization=quantization,
        created_at=created_at,
    )
    atomic_write_json(dest_dir / MANIFEST_FILENAME, package.to_dict())
    return package


__all__ = [
    "ADAPTER_DIRNAME",
    "MANIFEST_FILENAME",
    "MODEL_PACKAGE_VERSION",
    "TOKENIZER_DIRNAME",
    "ModelPackage",
    "build_package",
]
