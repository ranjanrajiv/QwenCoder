"""Probes the evaluation image for the Python and pytest versions actually in use.

The manifest must record what genuinely ran, not what the Dockerfile asked for (spec
section 74) — so this runs a trivial program inside the real evaluation sandbox once per
run rather than assuming the pinned version strings.
"""

from __future__ import annotations

import json

from ..sandbox import SandboxConfig, SandboxExecutor
from .errors import EvaluationError

_PROBE_CODE = (
    "import json, sys, pytest\n"
    "print(json.dumps({'python_version': sys.version.split()[0], "
    "'pytest_version': pytest.__version__}))\n"
)


def probe_versions(
    config: SandboxConfig, executor: SandboxExecutor | None = None
) -> tuple[str, str]:
    """Run ``_PROBE_CODE`` in the evaluation sandbox and return ``(python_version,
    pytest_version)``. Raises :class:`EvaluationError` if the probe does not succeed —
    an evaluation run must not start against an image that cannot even report its own
    versions.
    """
    executor = executor if executor is not None else SandboxExecutor(config=config)
    result = executor.execute(_PROBE_CODE)
    if result.status != "success":
        raise EvaluationError(
            "could not probe the evaluation image's Python/pytest versions "
            f"(status={result.status}, stderr={result.stderr.strip()[:200]!r}); "
            "is the image built? docker build -t <image> docker/evaluator/"
        )
    try:
        payload = json.loads(result.stdout.strip())
        return payload["python_version"], payload["pytest_version"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise EvaluationError(
            f"evaluation image probe returned unexpected output: {result.stdout!r}"
        ) from exc


__all__ = ["probe_versions"]
