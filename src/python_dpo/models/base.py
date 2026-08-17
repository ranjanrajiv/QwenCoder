"""Model-client abstraction — the seam between the pipeline and any inference stack.

Everything downstream depends on :class:`ModelClient`, never on Hugging Face
Transformers directly, so the Qwen backend can be swapped for a mock (tests), a
quantized loader, or a different runtime without touching the generator.

Nothing in this module imports ``torch`` or ``transformers``. Importing
``python_dpo.models`` must stay cheap — the model is loaded only when generation is
actually requested (spec 03 section 7), and ``tests/test_no_heavy_imports.py`` enforces
that.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Protocol, runtime_checkable

PROVIDER_TRANSFORMERS = "transformers"
PROVIDER_MOCK = "mock"
PROVIDERS = frozenset({PROVIDER_TRANSFORMERS, PROVIDER_MOCK})

DEVICES = frozenset({"auto", "cpu", "cuda"})
DTYPES = frozenset({"auto", "float32", "float16", "bfloat16"})

_MODEL_FIELDS = frozenset(
    {"provider", "name", "revision", "device", "dtype", "trust_remote_code", "quantization"}
)
_GENERATION_CONFIG_FIELDS = (
    "temperature",
    "top_p",
    "max_new_tokens",
    "do_sample",
    "repetition_penalty",
    "seed",
)


class ModelError(Exception):
    """Base class for model-client failures."""


class ModelLoadError(ModelError):
    """Raised when a model, tokenizer, or device cannot be initialized.

    Per spec 03 section 26.2 this is a *run-level* failure: no candidate in the run can
    succeed, so the generator records one failure and aborts rather than retrying.
    """


class InferenceError(ModelError):
    """Raised when a loaded model fails to produce output for one prompt."""


def _require_number(value: Any, label: str) -> float:
    """Accept an int or float but reject bool, which is an int subclass in Python."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{label} must be a number, got {type(value).__name__}")
    return float(value)


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelError(f"{label} must be an integer, got {type(value).__name__}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ModelError(f"{label} must be true or false, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters, validated on construction (spec 03 section 11).

    Stored verbatim on every candidate record so a generation can be reconstructed
    without consulting the config file that produced it.
    """

    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 512
    do_sample: bool = True
    repetition_penalty: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        temperature = _require_number(self.temperature, "generation.temperature")
        top_p = _require_number(self.top_p, "generation.top_p")
        max_new_tokens = _require_int(self.max_new_tokens, "generation.max_new_tokens")
        do_sample = _require_bool(self.do_sample, "generation.do_sample")
        repetition_penalty = _require_number(
            self.repetition_penalty, "generation.repetition_penalty"
        )
        seed = _require_int(self.seed, "generation.seed")

        if temperature < 0:
            raise ModelError("generation.temperature must not be negative")
        if do_sample and temperature <= 0:
            raise ModelError(
                "generation.temperature must be greater than 0 when do_sample is true; "
                "set do_sample to false for greedy decoding"
            )
        if not 0 < top_p <= 1:
            raise ModelError("generation.top_p must be in the range (0, 1]")
        if max_new_tokens < 1:
            raise ModelError("generation.max_new_tokens must be at least 1")
        if repetition_penalty <= 0:
            raise ModelError("generation.repetition_penalty must be greater than 0")
        if seed < 0:
            raise ModelError("generation.seed must not be negative")

        # Normalize numerics so the dict embedded in candidate records is stable.
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "repetition_penalty", repetition_penalty)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _GENERATION_CONFIG_FIELDS}

    @classmethod
    def from_dict(cls, data: Any) -> GenerationConfig:
        if not isinstance(data, dict):
            raise ModelError("generation config: expected a JSON object")
        unknown = sorted(set(data) - set(_GENERATION_CONFIG_FIELDS))
        if unknown:
            raise ModelError(f"generation config: unknown field(s): {', '.join(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class ModelConfig:
    """Which model to load and how (spec 03 sections 6, 8, 9, 10).

    Lives here rather than in ``config.py`` so the dependency runs one way — the
    configuration layer imports the model layer, never the reverse.
    """

    provider: str
    name: str
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    quantization: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ModelError(
                f"model.provider must be one of {', '.join(sorted(PROVIDERS))}, "
                f"got {self.provider!r}"
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelError("model.name must be a non-empty string")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ModelError("model.revision must be a non-empty string or null")

        if not isinstance(self.device, str) or not _is_valid_device(self.device):
            raise ModelError(
                f"model.device must be one of {', '.join(sorted(DEVICES))} or 'cuda:N', "
                f"got {self.device!r}"
            )
        if self.dtype not in DTYPES:
            raise ModelError(
                f"model.dtype must be one of {', '.join(sorted(DTYPES))}, got {self.dtype!r}"
            )
        _require_bool(self.trust_remote_code, "model.trust_remote_code")

        # Section 9 reserves the field for later without asking us to implement it now.
        # Accepting a value we ignore would silently mislead, so reject it outright.
        if self.quantization is not None:
            raise ModelError(
                "model.quantization is not implemented in Stage 3; set it to null"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "name": self.name,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "quantization": self.quantization,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> ModelConfig:
        """Build from a parsed ``model:`` section, rejecting unknown keys."""
        if not isinstance(data, dict):
            raise ModelError("model: expected a mapping")
        unknown = sorted(set(data) - _MODEL_FIELDS)
        if unknown:
            raise ModelError(f"model: unknown field(s): {', '.join(unknown)}")
        missing = sorted({"provider", "name"} - set(data))
        if missing:
            raise ModelError(f"model: missing required field(s): {', '.join(missing)}")

        defaults = {field.name: field.default for field in fields(cls)}
        return cls(
            provider=data["provider"],
            name=data["name"],
            revision=data.get("revision"),
            device=data.get("device", defaults["device"]),
            dtype=data.get("dtype", defaults["dtype"]),
            trust_remote_code=data.get("trust_remote_code", defaults["trust_remote_code"]),
            quantization=data.get("quantization"),
        )


def _is_valid_device(device: str) -> bool:
    if device in DEVICES:
        return True
    if device.startswith("cuda:"):
        index = device.split(":", 1)[1]
        return index.isdigit()
    return False


@dataclass(frozen=True)
class RawGeneration:
    """What a model client returns — text, not a candidate.

    Turning this into a :class:`~python_dpo.candidates.models.Candidate` is the
    generator's job; a client never decides whether its own output is usable.
    """

    text: str
    prompt_token_count: int | None = None
    completion_token_count: int | None = None
    finish_reason: str | None = None


@runtime_checkable
class ModelClient(Protocol):
    """The interface the pipeline programs against."""

    @property
    def name(self) -> str:
        """Model identifier, recorded on every candidate for provenance."""

    @property
    def revision(self) -> str | None:
        """Pinned revision, or ``None`` when the default was used."""

    @property
    def provider(self) -> str:
        """Backend that produced the output — one of :data:`PROVIDERS`."""

    def generate(self, prompt: str, generation_config: GenerationConfig) -> RawGeneration:
        """Produce one completion for ``prompt``.

        Raises :class:`ModelLoadError` if the backend cannot be initialized and
        :class:`InferenceError` if a loaded backend fails on this prompt.
        """
