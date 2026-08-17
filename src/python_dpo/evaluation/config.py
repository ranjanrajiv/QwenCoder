"""The ``evaluation:`` configuration section (spec 06 sections 33-35, 53).

Only the evaluation image, its timeout, its startup grace period, and whether the health
path may pull the image differ from the Stage 5 sandbox. Every other isolation setting —
network mode, user, capabilities, resource limits — is inherited unchanged from the
audited ``sandbox:`` section by
:func:`~python_dpo.evaluation.pytest_runner.build_evaluation_sandbox_config`, so no
evaluation-specific configuration can weaken it (spec section 78).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import EvaluationConfigError

# The evaluator image is built locally (docker/evaluator/Dockerfile) rather than pulled
# from a registry, since the container has no network at evaluation time and pytest must
# already be present at image-build time (spec section 36).
DEFAULT_EVALUATOR_IMAGE = "python-dpo-evaluator:1.0"
# pytest startup and test collection need more headroom than Stage 5's plain-script 5s
# default.
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_STARTUP_GRACE_SECONDS = 10

_EVALUATION_KEYS = frozenset({"image", "timeout_seconds", "startup_grace_seconds", "auto_pull"})
# Same pattern as sandbox.config: rejects a bare, unpinned repository reference.
_IMAGE_RE = re.compile(r"^[\w.\-/]+(:[\w.\-]+|@sha256:[0-9a-f]{64})$")


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationConfigError(f"evaluation.{label} must be an integer of 1 or greater")
    return value


@dataclass(frozen=True)
class EvaluationConfig:
    """Which image to run pytest in, and its own timeout budget."""

    image: str = DEFAULT_EVALUATOR_IMAGE
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    startup_grace_seconds: int = DEFAULT_STARTUP_GRACE_SECONDS
    # A locally-built image is not on any registry to pull from, unlike Stage 5's stock
    # python:3.12-slim, so auto-pulling defaults off here.
    auto_pull: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.image, str) or not self.image.strip():
            raise EvaluationConfigError("evaluation.image must be a non-empty string")
        if not _IMAGE_RE.match(self.image):
            raise EvaluationConfigError(
                "evaluation.image must be pinned as repository:tag or repository@sha256:..., "
                f"got {self.image!r}"
            )
        if self.image.endswith(":latest"):
            raise EvaluationConfigError(
                "evaluation.image must not use the ':latest' tag; pin a specific version"
            )
        _require_positive_int(self.timeout_seconds, "timeout_seconds")
        _require_positive_int(self.startup_grace_seconds, "startup_grace_seconds")
        if not isinstance(self.auto_pull, bool):
            raise EvaluationConfigError("evaluation.auto_pull must be true or false")

    @classmethod
    def from_mapping(cls, data: Any) -> EvaluationConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise EvaluationConfigError("evaluation: expected a mapping")
        unknown = sorted(set(data) - _EVALUATION_KEYS)
        if unknown:
            raise EvaluationConfigError(f"evaluation: unknown key(s): {', '.join(unknown)}")
        return cls(**data)


__all__ = [
    "DEFAULT_EVALUATOR_IMAGE",
    "DEFAULT_STARTUP_GRACE_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "EvaluationConfig",
]
