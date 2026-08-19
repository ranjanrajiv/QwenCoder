"""Model packaging, verification and registry (spec 12 sections 35-49, Stage 12 plan
Phase 3).

Named ``python_dpo.packaging`` rather than the spec's literal ``python_dpo.models``
because that name already belongs to the model-*client* package
(:class:`~python_dpo.models.base.ModelConfig`, :class:`~python_dpo.models.qwen.QwenModelClient`).
``models/registry.json`` still lives at the project root, per spec section 45.

Every module here defers its own heavy imports (torch/transformers/peft) into the
function that needs them, so importing this package costs nothing --
``tests/test_no_heavy_imports.py`` asserts it directly.
"""

from __future__ import annotations

from .compare import ModelComparisonRow, compare_models
from .errors import MergeUnsupportedError, PackagingError, RegistryError, VerificationError
from .inference import generate, generate_batch
from .merge import merge_adapter
from .package import ModelPackage, build_package
from .registry import ModelRegistry, RegistryEntry
from .verify import VerificationResult, verify_package

__all__ = [
    "MergeUnsupportedError",
    "ModelComparisonRow",
    "ModelPackage",
    "ModelRegistry",
    "PackagingError",
    "RegistryEntry",
    "RegistryError",
    "VerificationError",
    "VerificationResult",
    "build_package",
    "compare_models",
    "generate",
    "generate_batch",
    "merge_adapter",
    "verify_package",
]
