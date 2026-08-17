"""Model clients — the inference seam.

Importing this package is cheap: neither ``torch`` nor ``transformers`` is imported
until a :class:`~python_dpo.models.qwen.QwenModelClient` is actually asked to generate.
"""

from .base import (
    DEVICES,
    DTYPES,
    PROVIDER_MOCK,
    PROVIDER_TRANSFORMERS,
    PROVIDERS,
    GenerationConfig,
    InferenceError,
    ModelClient,
    ModelConfig,
    ModelError,
    ModelLoadError,
    RawGeneration,
)
from .mock import DEFAULT_MOCK_NAME, MockModelClient
from .qwen import QwenModelClient

__all__ = [
    "DEFAULT_MOCK_NAME",
    "DEVICES",
    "DTYPES",
    "PROVIDERS",
    "PROVIDER_MOCK",
    "PROVIDER_TRANSFORMERS",
    "GenerationConfig",
    "InferenceError",
    "MockModelClient",
    "ModelClient",
    "ModelConfig",
    "ModelError",
    "ModelLoadError",
    "QwenModelClient",
    "RawGeneration",
]
