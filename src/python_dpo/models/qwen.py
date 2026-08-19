"""Qwen Coder inference via Hugging Face Transformers.

Two rules shape this module:

1. **Nothing heavy is imported at module scope.** ``torch`` and ``transformers`` are
   imported inside :func:`_import_backend`, which runs on the first ``generate()`` call.
   ``import python_dpo`` must never pull several GB of weights into memory
   (spec 03 section 7).
2. **All hardware-specific behavior lives here.** Device and dtype resolution is
   confined to this file so the rest of the pipeline carries no GPU assumptions
   (spec 03 section 10).

The model identifier is never hard-coded — it arrives via :class:`ModelConfig`, which is
built from ``config.yaml`` (spec 03 section 6).
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    PROVIDER_TRANSFORMERS,
    GenerationConfig,
    InferenceError,
    ModelConfig,
    ModelLoadError,
    RawGeneration,
)

logger = logging.getLogger("python_dpo.models.qwen")

_INSTALL_HINT = "install the model backend with: pip install -e '.[model]'"


def _import_backend() -> tuple[Any, Any]:
    """Import torch and transformers, turning a missing extra into a clear message."""
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatched sys.modules
        raise ModelLoadError(f"{exc}; {_INSTALL_HINT}") from exc
    return torch, transformers


def resolve_device(requested: str, *, cuda_available: bool) -> str:
    """Map a configured device onto a concrete one.

    ``auto`` prefers CUDA and falls back to CPU. An explicit CUDA request on a machine
    without CUDA is an error rather than a silent CPU fallback — a run that quietly takes
    a hundred times longer than expected is worse than one that stops and says why.
    """
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cpu":
        return "cpu"
    if not cuda_available:
        raise ModelLoadError(
            f"model.device is {requested!r} but CUDA is not available; "
            "set model.device to 'cpu' or 'auto'"
        )
    return requested


def resolve_dtype(requested: str, device: str) -> str:
    """Map a configured dtype onto a concrete torch dtype name.

    ``auto`` picks bfloat16 on CUDA (halves the footprint at no practical quality cost
    for a coder model) and float32 on CPU, where half precision is usually unsupported
    or slower.
    """
    if requested != "auto":
        return requested
    return "bfloat16" if device.startswith("cuda") else "float32"


def build_generation_kwargs(generation_config: GenerationConfig) -> dict[str, Any]:
    """Translate a :class:`GenerationConfig` into ``model.generate`` keyword arguments.

    ``temperature`` and ``top_p`` are only meaningful when sampling; passing them for a
    greedy decode makes Transformers emit warnings about ignored parameters.
    """
    kwargs: dict[str, Any] = {
        "max_new_tokens": generation_config.max_new_tokens,
        "do_sample": generation_config.do_sample,
        "repetition_penalty": generation_config.repetition_penalty,
    }
    if generation_config.do_sample:
        kwargs["temperature"] = generation_config.temperature
        kwargs["top_p"] = generation_config.top_p
    return kwargs


class QwenModelClient:
    """A :class:`~python_dpo.models.base.ModelClient` backed by Transformers.

    Construction is free: it stores configuration and loads nothing. The first
    ``generate()`` call triggers the download/load.
    """

    def __init__(self, config: ModelConfig) -> None:
        if config.provider != PROVIDER_TRANSFORMERS:
            raise ModelLoadError(
                f"QwenModelClient requires model.provider {PROVIDER_TRANSFORMERS!r}, "
                f"got {config.provider!r}"
            )
        self._config = config
        self._torch: Any = None
        self._transformers: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def revision(self) -> str | None:
        return self._config.revision

    @property
    def provider(self) -> str:
        return self._config.provider

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        """Free GPU memory before another model loads in the same process.

        Never called from within this class -- ``generate()`` is designed to be called
        repeatedly across many candidates without reloading. Callers that run several
        model-loading stages back to back in one process (the Stage 12 orchestrator)
        must call this explicitly once they are done with this client, matching the
        pattern already used by :class:`~python_dpo.model_evaluation.runners.BaseModelRunner`
        and :class:`~python_dpo.model_evaluation.runners.AdapterModelRunner`.
        """
        if self._model is not None:
            del self._model
            self._model = None
        self._tokenizer = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        torch, transformers = _import_backend()
        device = resolve_device(self._config.device, cuda_available=torch.cuda.is_available())
        dtype_name = resolve_dtype(self._config.dtype, device)
        torch_dtype = getattr(torch, dtype_name, None)
        if torch_dtype is None:
            raise ModelLoadError(f"torch has no dtype named {dtype_name!r}")

        if device == "cpu":
            logger.warning(
                "Running %s on CPU; generation will be substantially slower than on GPU.",
                self._config.name,
            )

        logger.info(
            "Loading model %s | device=%s dtype=%s revision=%s trust_remote_code=%s",
            self._config.name,
            device,
            dtype_name,
            self._config.revision or "default",
            self._config.trust_remote_code,
        )

        shared = {
            "revision": self._config.revision,
            "trust_remote_code": self._config.trust_remote_code,
        }
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self._config.name, **shared)
            model = _load_model_with_dtype(
                transformers.AutoModelForCausalLM, self._config.name, torch_dtype, **shared
            )
            model.to(device)
            model.eval()
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"failed to load model {self._config.name!r}: {exc}") from exc

        self._torch = torch
        self._transformers = transformers
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        logger.info("Model loaded: %s", self._config.name)

    def _render(self, prompt: str) -> str:
        """Wrap the prompt in the model's chat template when it declares one.

        Qwen ``-Instruct`` checkpoints are trained with a chat template; feeding them a
        bare prompt measurably degrades output. Base checkpoints have no template, so the
        prompt is passed through unchanged.
        """
        template = getattr(self._tokenizer, "chat_template", None)
        if not template:
            logger.debug("Tokenizer declares no chat template; using the raw prompt.")
            return prompt
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str, generation_config: GenerationConfig) -> RawGeneration:
        self._ensure_loaded()
        torch = self._torch

        # Seeded per call so a given (prompt, seed) pair behaves the same regardless of
        # how many generations preceded it. See the reproducibility caveat in the README:
        # this is seeding, not a bit-for-bit guarantee.
        self._transformers.set_seed(generation_config.seed)

        try:
            inputs = self._tokenizer(self._render(prompt), return_tensors="pt").to(self._device)
            prompt_length = int(inputs["input_ids"].shape[-1])
            pad_token_id = self._tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self._tokenizer.eos_token_id

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    **build_generation_kwargs(generation_config),
                    pad_token_id=pad_token_id,
                )
        except Exception as exc:
            raise InferenceError(f"generation failed: {type(exc).__name__}: {exc}") from exc

        # Slice off the prompt so raw_output is the completion alone; otherwise every
        # candidate record would carry its own prompt twice.
        completion_ids = output_ids[0][prompt_length:]
        completion_length = int(completion_ids.shape[-1])
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)

        return RawGeneration(
            text=text,
            prompt_token_count=prompt_length,
            completion_token_count=completion_length,
            finish_reason=(
                "length" if completion_length >= generation_config.max_new_tokens else "stop"
            ),
        )


def _load_model_with_dtype(loader: Any, name: str, torch_dtype: Any, **kwargs: Any) -> Any:
    """Load a model across the Transformers releases that renamed ``torch_dtype``.

    Newer versions take ``dtype``; older ones take ``torch_dtype``. Probing the keyword
    is more robust than pinning a version range, since ``from_pretrained`` accepts
    ``**kwargs`` and cannot be inspected.
    """
    try:
        return loader.from_pretrained(name, dtype=torch_dtype, **kwargs)
    except TypeError:
        return loader.from_pretrained(name, torch_dtype=torch_dtype, **kwargs)
