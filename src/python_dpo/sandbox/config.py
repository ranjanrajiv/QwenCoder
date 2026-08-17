"""The ``sandbox:`` configuration section (spec 05 section 53).

A frozen dataclass validating in ``__post_init__``, matching the house style of
``models/base.py`` and ``problems/models.py``. Validation raises
:class:`~python_dpo.sandbox.errors.SandboxConfigError`; ``python_dpo.config`` catches and
re-raises it as ``ConfigError``, keeping the dependency one-way — the configuration layer
imports the sandbox layer, never the reverse.

Every field here is also recorded on each :class:`~python_dpo.sandbox.result.ExecutionResult`
(spec section 52), so a later stage can always determine the environment a result came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import SandboxConfigError

# Spec section 12 makes network isolation a MUST, so `none` is the only value this stage
# accepts. The key exists so the configuration is explicit and forward-compatible, but a
# value we would have to ignore is rejected rather than silently honoured — the same
# reasoning ModelConfig.quantization already applies.
NETWORK_MODES = frozenset({"none"})

DEFAULT_IMAGE = "python:3.12-slim"

# nobody:nogroup, present in Debian-derived images including python:*-slim. Spec section 19
# explicitly permits a numeric non-root UID/GID instead of a named user, which is what lets
# the sandbox run the stock image with no Dockerfile to maintain (spec section 10).
DEFAULT_USER = "65534:65534"

_SANDBOX_KEYS = frozenset(
    {
        "image",
        "image_digest",
        "network_mode",
        "cpus",
        "memory",
        "pids_limit",
        "timeout_seconds",
        "startup_grace_seconds",
        "max_output_bytes",
        "read_only_root",
        "run_as_non_root",
        "drop_capabilities",
        "user",
        "tmpfs_size",
        "workspace_root",
        "auto_pull",
    }
)

# Docker size suffixes: b, k, m, g (case-insensitive), or a bare byte count.
_SIZE_RE = re.compile(r"^\d+(\.\d+)?[bkmgBKMG]?$")
# A repository:tag or repository@sha256:... reference. Rejects a bare repository, so an
# unpinned `python` (implicitly :latest) cannot slip through — spec section 9.
_IMAGE_RE = re.compile(r"^[\w.\-/]+(:[\w.\-]+|@sha256:[0-9a-f]{64})$")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SandboxConfigError(f"sandbox.{label} must be a non-empty string")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label)


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SandboxConfigError(f"sandbox.{label} must be an integer of 1 or greater")
    return value


def _require_positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SandboxConfigError(f"sandbox.{label} must be a number greater than 0")
    return float(value)


def _require_flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SandboxConfigError(f"sandbox.{label} must be true or false")
    return value


def _require_size(value: Any, label: str) -> str:
    text = _require_text(value, label)
    if not _SIZE_RE.match(text):
        raise SandboxConfigError(
            f"sandbox.{label} must be a Docker size such as '512m' or '1g', got {text!r}"
        )
    return text


@dataclass(frozen=True)
class SandboxConfig:
    """Resource limits and isolation settings for one sandbox."""

    image: str = DEFAULT_IMAGE
    image_digest: str | None = None
    network_mode: str = "none"
    cpus: float = 1.0
    memory: str = "512m"
    pids_limit: int = 64
    timeout_seconds: int = 5
    # Container creation, image layer setup and interpreter start cost roughly 2s even on a
    # warm image, and that overhead is not the candidate's doing. Without a separate
    # allowance it would silently eat the candidate's budget — a 5s timeout would give a
    # candidate under 3s, and a loaded machine could time out a program that is merely
    # slow-ish rather than wrong. The wait budget is timeout_seconds + this.
    startup_grace_seconds: int = 10
    max_output_bytes: int = 1_000_000
    read_only_root: bool = True
    run_as_non_root: bool = True
    drop_capabilities: bool = True
    user: str = DEFAULT_USER
    tmpfs_size: str = "64m"
    workspace_root: str | None = None
    auto_pull: bool = True

    def __post_init__(self) -> None:
        image = _require_text(self.image, "image")
        if not _IMAGE_RE.match(image):
            raise SandboxConfigError(
                f"sandbox.image must be pinned as repository:tag or repository@sha256:..., "
                f"got {image!r}"
            )
        if image.endswith(":latest"):
            # Spec section 9: :latest can change under us, which would silently change what
            # every recorded result means.
            raise SandboxConfigError(
                "sandbox.image must not use the ':latest' tag; pin a specific version"
            )

        digest = _require_optional_text(self.image_digest, "image_digest")
        if digest is not None and not re.match(r"^sha256:[0-9a-f]{64}$", digest):
            raise SandboxConfigError(
                "sandbox.image_digest must look like 'sha256:<64 hex chars>'"
            )

        if self.network_mode not in NETWORK_MODES:
            raise SandboxConfigError(
                f"sandbox.network_mode must be {', '.join(sorted(NETWORK_MODES))} "
                f"(candidate containers must have no network access), got "
                f"{self.network_mode!r}"
            )

        _require_positive_number(self.cpus, "cpus")
        _require_size(self.memory, "memory")
        _require_positive_int(self.pids_limit, "pids_limit")
        _require_positive_int(self.timeout_seconds, "timeout_seconds")
        _require_positive_int(self.startup_grace_seconds, "startup_grace_seconds")
        _require_positive_int(self.max_output_bytes, "max_output_bytes")
        _require_flag(self.read_only_root, "read_only_root")
        _require_flag(self.run_as_non_root, "run_as_non_root")
        _require_flag(self.drop_capabilities, "drop_capabilities")
        _require_size(self.tmpfs_size, "tmpfs_size")
        _require_optional_text(self.workspace_root, "workspace_root")
        _require_flag(self.auto_pull, "auto_pull")

        user = _require_text(self.user, "user")
        if self.run_as_non_root:
            if not re.match(r"^\d+(:\d+)?$", user):
                raise SandboxConfigError(
                    "sandbox.user must be a numeric UID or UID:GID when run_as_non_root "
                    f"is true, got {user!r}"
                )
            if user.split(":")[0] == "0":
                raise SandboxConfigError(
                    "sandbox.user must not be UID 0; candidate code must not run as root"
                )

        object.__setattr__(self, "cpus", float(self.cpus))

    @property
    def image_reference(self) -> str:
        """The image to actually run: digest-pinned when a digest is configured.

        A digest is the strongest reproducibility pin available (spec section 9) because it
        names exact bytes rather than a mutable tag.
        """
        if self.image_digest is None:
            return self.image
        repository = self.image.split("@")[0].split(":")[0]
        return f"{repository}@{self.image_digest}"

    def to_dict(self) -> dict[str, Any]:
        """The spec section 52 environment record, stamped on every ExecutionResult."""
        return {
            "image": self.image,
            "image_digest": self.image_digest,
            "image_reference": self.image_reference,
            "network_mode": self.network_mode,
            "cpus": self.cpus,
            "memory": self.memory,
            "pids_limit": self.pids_limit,
            "timeout_seconds": self.timeout_seconds,
            "startup_grace_seconds": self.startup_grace_seconds,
            "max_output_bytes": self.max_output_bytes,
            "read_only_root": self.read_only_root,
            "run_as_non_root": self.run_as_non_root,
            "drop_capabilities": self.drop_capabilities,
            "user": self.user,
            "tmpfs_size": self.tmpfs_size,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> SandboxConfig:
        """Build from a parsed ``sandbox:`` section, rejecting unknown keys."""
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise SandboxConfigError("sandbox: expected a mapping")
        unknown = sorted(set(data) - _SANDBOX_KEYS)
        if unknown:
            raise SandboxConfigError(f"sandbox: unknown key(s): {', '.join(unknown)}")
        return cls(**data)


__all__ = [
    "DEFAULT_IMAGE",
    "DEFAULT_USER",
    "NETWORK_MODES",
    "SandboxConfig",
]
