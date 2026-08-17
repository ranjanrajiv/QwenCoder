"""Environment metadata captured in the run manifest (spec 04 section 33).

Records only what future reproducibility needs: Python version, platform, and the
inference-stack versions when they are installed. Never records a username, a home
directory path, an API key, or a token — ``platform.node()`` (the machine's hostname) is
deliberately never called.

Every optional dependency is probed inside a ``try`` and recorded as ``None`` when
absent, so importing this module never forces a ``torch``/``transformers`` import —
``tests/test_no_heavy_imports.py`` must stay green.
"""

from __future__ import annotations

import platform
import sys
from typing import Any


def capture_environment() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "transformers_version": _version("transformers"),
        "torch_version": _version("torch"),
        "cuda_version": _cuda_version(),
    }


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def _cuda_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.version.cuda


__all__ = ["capture_environment"]
