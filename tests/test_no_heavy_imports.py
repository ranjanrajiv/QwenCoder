"""Guard: importing the package must not load a model backend (spec 03 section 7).

``import python_dpo`` should cost milliseconds, not several GB of weights. The rule is
easy to break by accident — a single top-level ``import torch`` in the Qwen client would
do it — so it is asserted rather than assumed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HEAVY_MODULES = ("torch", "transformers", "accelerate")

_PROBE = """
import sys

import python_dpo
import python_dpo.candidates
import python_dpo.cli
import python_dpo.config
import python_dpo.generation
import python_dpo.models
import python_dpo.models.qwen

heavy = [name for name in {heavy!r} if name in sys.modules]
print(",".join(heavy))
"""


def test_importing_the_package_does_not_load_a_model_backend():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(heavy=HEAVY_MODULES)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    loaded = result.stdout.strip()
    assert loaded == "", f"these were imported eagerly: {loaded}"
