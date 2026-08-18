"""Base and DPO-adapter inference runners (spec sections 16-19, 28, 118).

All heavy imports are deferred into :func:`_import_backend`, so importing this module
costs nothing and ``tests/test_no_heavy_imports.py`` keeps passing.

Both runners are built from the **same** :class:`~python_dpo.model_evaluation.config.QuantizationSettings`
and the **same** chat-template rendering, so spec section 19's "identical quantization for
both models" and section 28's "identical prompt formatting" are structural rather than
something a caller could get wrong. The model is loaded once (spec section 17) and reused
across every problem and sample; :meth:`ModelRunnerBase.unload` frees it before the other
variant loads, following the Stage 9 GPU-test lesson that a held model starves the next
step.

:class:`AdapterModelRunner` performs spec section 15's integrity checks --  adapter
directory, ``adapter_config.json``, weights present, recorded base model matching the
configured one -- and raises :class:`~python_dpo.model_evaluation.errors.AdapterIntegrityError`
rather than **ever** falling back to the base model (spec sections 15, 118).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import GenerationSettings, QuantizationSettings
from .errors import AdapterIntegrityError, ModelEvaluationError

logger = logging.getLogger("python_dpo.model_evaluation.runners")

_INSTALL_HINT = "install the training backend with: pip install -e '.[training]'"
_ADAPTER_WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")


class ModelDependencyError(ModelEvaluationError):
    """Raised when the torch/transformers/peft backend is not installed."""


def _import_backend() -> dict[str, Any]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise ModelDependencyError(f"{exc}; {_INSTALL_HINT}") from exc
    try:
        transformers.utils.logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 - cosmetic only
        pass
    return {"torch": torch, "transformers": transformers, "peft": peft}


def resolve_torch_dtype(name: str):
    backend = _import_backend()
    torch = backend["torch"]
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ModelEvaluationError(f"unsupported compute dtype {name!r}")
    return mapping[name]


def build_quantization_config(settings: QuantizationSettings):
    """The spec section 19 ``BitsAndBytesConfig``, or ``None`` when disabled."""
    if not settings.enabled:
        return None
    backend = _import_backend()
    transformers = backend["transformers"]
    compute_dtype = resolve_torch_dtype(settings.compute_dtype)
    if settings.bits == 4:
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=settings.quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=settings.double_quant,
        )
    return transformers.BitsAndBytesConfig(load_in_8bit=True)


def build_generation_kwargs(generation: GenerationSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": generation.max_new_tokens,
        "do_sample": generation.do_sample,
        "repetition_penalty": generation.repetition_penalty,
    }
    if generation.do_sample:
        kwargs["temperature"] = generation.temperature
        kwargs["top_p"] = generation.top_p
    return kwargs


@dataclass(frozen=True)
class Generation:
    """One raw completion from a runner (not yet extracted or evaluated)."""

    text: str
    generation_time_ms: int
    generated_tokens: int


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    """Apply the model's chat template when it declares one (spec section 28), matching
    Stage 3 generation and Stage 9 training exactly -- the same rendering both variants
    were built and trained against."""
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


class _BaseRunnerImpl:
    """Shared load/generate/unload machinery for both variants.

    Not itself the public seam -- :class:`BaseModelRunner` and :class:`AdapterModelRunner`
    are the two concrete runners, kept as separate classes (rather than one class with an
    ``adapter_dir=None`` flag) so spec section 118's isolation guarantee is visible in the
    type itself: nothing in ``BaseModelRunner`` can load an adapter because it has no
    adapter-loading code path at all.
    """

    variant: str

    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str | None,
        quantization: QuantizationSettings,
        generation: GenerationSettings,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.quantization = quantization
        self.generation = generation
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._transformers: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load_tokenizer_and_base(self) -> tuple[Any, Any]:
        backend = _import_backend()
        transformers = backend["transformers"]
        self._torch = backend["torch"]
        self._transformers = transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name, revision=self.model_revision, trust_remote_code=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = build_quantization_config(self.quantization)
        base = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            quantization_config=quantization_config,
            dtype=resolve_torch_dtype(self.quantization.compute_dtype),
            device_map={"": 0},
            trust_remote_code=False,
        )
        return tokenizer, base

    def unload(self) -> None:
        """Free the model before the other variant loads (spec section 17)."""
        if self._model is not None:
            del self._model
            self._model = None
        self._tokenizer = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def generate(self, prompt: str, *, seed: int) -> Generation:
        if self._model is None:
            raise ModelEvaluationError(f"{self.variant} runner has not been loaded")
        torch = self._torch
        self._transformers.set_seed(seed)

        rendered = _render_prompt(self._tokenizer, prompt)
        inputs = self._tokenizer(rendered, return_tensors="pt").to(self._model.device)
        prompt_length = int(inputs["input_ids"].shape[-1])
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, **build_generation_kwargs(self.generation), pad_token_id=pad_token_id
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        completion_ids = output_ids[0][prompt_length:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return Generation(
            text=text,
            generation_time_ms=elapsed_ms,
            generated_tokens=int(completion_ids.shape[-1]),
        )


class BaseModelRunner(_BaseRunnerImpl):
    """Unmodified base weights, quantized identically to the DPO runner (spec section 19)."""

    variant = "base"

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        tokenizer, model = self._load_tokenizer_and_base()
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        logger.info("Base runner loaded: %s", self.model_name)


def verify_adapter_integrity(adapter_dir: Path, model_name: str) -> dict[str, Any]:
    """Spec section 15's adapter integrity checks, standalone so ``evaluate-model
    validate`` (spec section 102) can run them without constructing a runner or touching
    torch. Returns the parsed ``adapter_config.json`` on success.
    """
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise AdapterIntegrityError(f"no adapter directory at {adapter_dir}")

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise AdapterIntegrityError(f"{adapter_dir} has no adapter_config.json")
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdapterIntegrityError(f"{config_path}: invalid JSON: {exc}") from exc

    if not any((adapter_dir / name).is_file() for name in _ADAPTER_WEIGHT_FILENAMES):
        raise AdapterIntegrityError(
            f"{adapter_dir} has no adapter weights file ({' or '.join(_ADAPTER_WEIGHT_FILENAMES)})"
        )

    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base != model_name:
        raise AdapterIntegrityError(
            f"adapter at {adapter_dir} was trained against base model {adapter_base!r}, "
            f"but this evaluation is configured for {model_name!r}"
        )
    return adapter_config


class AdapterModelRunner(_BaseRunnerImpl):
    """Base weights plus the trained LoRA adapter (spec sections 13, 15, 118)."""

    variant = "dpo"

    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str | None,
        adapter_dir: Path,
        quantization: QuantizationSettings,
        generation: GenerationSettings,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_revision=model_revision,
            quantization=quantization,
            generation=generation,
        )
        self.adapter_dir = Path(adapter_dir)

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Never falls back to the base model on failure (spec section 118).
        verify_adapter_integrity(self.adapter_dir, self.model_name)

        backend = _import_backend()
        peft = backend["peft"]
        tokenizer, base = self._load_tokenizer_and_base()
        try:
            model = peft.PeftModel.from_pretrained(base, str(self.adapter_dir))
        except Exception as exc:  # noqa: BLE001 - any failure means the adapter is unusable
            raise AdapterIntegrityError(
                f"adapter at {self.adapter_dir} could not be loaded: {exc}"
            ) from exc
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        logger.info("Adapter runner loaded: %s + %s", self.model_name, self.adapter_dir)


__all__ = [
    "AdapterModelRunner",
    "BaseModelRunner",
    "Generation",
    "ModelDependencyError",
    "build_generation_kwargs",
    "build_quantization_config",
    "resolve_torch_dtype",
    "verify_adapter_integrity",
]
