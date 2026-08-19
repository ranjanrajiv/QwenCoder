"""The packaging package's exception hierarchy (spec 12 sections 35-49).

One base class, one subclass per failure mode a caller must distinguish -- matching
:mod:`python_dpo.pipeline.errors`'s house style.
"""

from __future__ import annotations


class PackagingError(Exception):
    """Base class for every error raised by :mod:`python_dpo.packaging`."""


class VerificationError(PackagingError):
    """Raised when a packaged model fails the load -> generate -> sandbox-execute check
    (spec section 38). Packaging has no ``--skip-verification`` escape hatch: this is
    always fatal to the package/pipeline stage that raised it.
    """


class MergeUnsupportedError(PackagingError):
    """Raised when a LoRA adapter cannot be safely merged into the base model's weights
    (spec sections 43, 44)."""


class RegistryError(PackagingError):
    """Raised when ``models/registry.json`` is malformed, or an entry/status change is
    invalid (spec sections 45-48)."""


__all__ = [
    "MergeUnsupportedError",
    "PackagingError",
    "RegistryError",
    "VerificationError",
]
