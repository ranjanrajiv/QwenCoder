"""Tokenizer, quantization, model and LoRA loading (spec 09 sections 6-20, 28-30).

Every heavy import is deferred into a function, so importing this module costs nothing and
``tests/test_no_heavy_imports.py`` keeps passing. That guard is not ceremony: a single
top-level ``import torch`` here would make ``python -m python_dpo --help`` load CUDA.

Two functions in this module are the stage's real safety net:

* :func:`validate_target_modules` (section 16) — training with zero LoRA targets would
  produce an adapter that changes nothing while reporting success.
* :func:`count_parameters` (section 20) — if every parameter is trainable, LoRA did not
  apply and this is a full fine-tune, which Step 9 must never perform by accident.

Both fail loudly rather than warn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig, LoraSettings, ModelSpec, QuantizationSettings
from .errors import (
    FullFineTuneError,
    HardwareError,
    TargetModuleError,
    TrainingDependencyError,
)

logger = logging.getLogger("python_dpo.training.loader")

_INSTALL_HINT = "install the training backend with: pip install -e '.[training]'"

# Spec section 6: the architecture this stage is built and tested against.
SUPPORTED_ARCHITECTURE_SUFFIX = "ForCausalLM"


@dataclass(frozen=True)
class TokenizerInfo:
    """What was actually loaded (spec section 28), recorded in the manifest."""

    name: str
    revision: str | None
    vocab_size: int
    pad_token: str | None
    eos_token: str | None
    padding_side: str
    pad_token_from_eos: bool
    has_chat_template: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "padding_side": self.padding_side,
            "pad_token_from_eos": self.pad_token_from_eos,
            "has_chat_template": self.has_chat_template,
        }


@dataclass(frozen=True)
class ParameterCounts:
    """Spec section 19's parameter accounting."""

    total: int
    trainable: int

    @property
    def percentage(self) -> float:
        return 100.0 * self.trainable / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_parameters": self.total,
            "trainable_parameters": self.trainable,
            "trainable_percentage": round(self.percentage, 6),
        }


def import_backend() -> dict[str, Any]:
    """Import the training stack, turning a missing extra into one actionable message.

    Mirrors ``python_dpo.models.qwen._import_backend``. Called at the top of every
    function that needs the backend, never at module scope.
    """
    try:
        import bitsandbytes  # noqa: F401
        import peft
        import torch
        import transformers
        import trl
    except ImportError as exc:
        raise TrainingDependencyError(f"{exc}; {_INSTALL_HINT}") from exc

    # Hugging Face's weight-loading progress bar writes hundreds of carriage-returned
    # frames, which is fine on a TTY and unreadable in a captured log. This stage's own
    # logging reports what matters, so the bar is turned off rather than tolerated.
    try:
        transformers.utils.logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 - cosmetic only; never fail a run over a progress bar
        pass

    return {"torch": torch, "transformers": transformers, "peft": peft, "trl": trl}


def resolve_torch_dtype(name: str):
    """Map a configured dtype name onto a torch dtype."""
    backend = import_backend()
    torch = backend["torch"]
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise HardwareError(f"unsupported compute dtype {name!r}")
    return mapping[name]


def load_tokenizer(model_spec: ModelSpec) -> tuple[Any, TokenizerInfo]:
    """Load the tokenizer belonging to the exact base model (spec sections 28, 29).

    A missing pad token falls back to EOS — the usual choice for a decoder-only model —
    and the fallback is **recorded** rather than applied silently, since it changes how
    sequences are batched. ``padding_side`` is set to ``left``, which TRL's processing
    class expects for DPO.
    """
    backend = import_backend()
    transformers = backend["transformers"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_spec.name,
        revision=model_spec.revision,
        trust_remote_code=False,
    )

    pad_from_eos = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise HardwareError(
                f"tokenizer for {model_spec.name!r} has neither a pad token nor an EOS "
                "token; DPO batching requires padding"
            )
        tokenizer.pad_token = tokenizer.eos_token
        pad_from_eos = True
        logger.info(
            "Tokenizer has no pad token; using EOS (%r) for padding.", tokenizer.eos_token
        )
    tokenizer.padding_side = "left"

    info = TokenizerInfo(
        name=model_spec.name,
        revision=model_spec.revision,
        vocab_size=int(getattr(tokenizer, "vocab_size", 0)),
        pad_token=tokenizer.pad_token,
        eos_token=tokenizer.eos_token,
        padding_side=tokenizer.padding_side,
        pad_token_from_eos=pad_from_eos,
        has_chat_template=bool(getattr(tokenizer, "chat_template", None)),
    )
    return tokenizer, info


def build_quantization_config(settings: QuantizationSettings, compute_dtype: str):
    """The spec section 9 ``BitsAndBytesConfig``, or ``None`` when disabled."""
    if not settings.enabled:
        return None
    backend = import_backend()
    transformers = backend["transformers"]
    torch_dtype = resolve_torch_dtype(compute_dtype)

    if settings.bits == 4:
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=settings.quant_type,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=settings.double_quant,
        )
    return transformers.BitsAndBytesConfig(load_in_8bit=True)


def load_model(
    model_spec: ModelSpec,
    quantization: QuantizationSettings,
    compute_dtype: str,
    *,
    gradient_checkpointing: bool = True,
    max_length: int = 1024,
):
    """Load the base model in 4-bit and prepare it for k-bit training (sections 6, 8).

    Every section 6 compatibility check runs *before* returning, so an incompatible model
    stops the run before any artifact is written.
    """
    backend = import_backend()
    transformers = backend["transformers"]
    peft = backend["peft"]

    quantization_config = build_quantization_config(quantization, compute_dtype)
    torch_dtype = resolve_torch_dtype(compute_dtype)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_spec.name,
        revision=model_spec.revision,
        quantization_config=quantization_config,
        dtype=torch_dtype,
        device_map={"": 0},
        trust_remote_code=False,
    )

    _check_model_compatibility(model, model_spec, max_length)

    if quantization.enabled:
        model = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )

    # Spec section 49: gradient checkpointing is incompatible with the KV cache, so the
    # cache is disabled explicitly for training rather than left to warn at every step.
    if gradient_checkpointing:
        model.config.use_cache = False

    return model


def _check_model_compatibility(model: Any, model_spec: ModelSpec, max_length: int) -> None:
    """Spec section 6's pre-training verification."""
    config = model.config

    architectures = list(getattr(config, "architectures", None) or [])
    if architectures and not any(
        arch.endswith(SUPPORTED_ARCHITECTURE_SUFFIX) for arch in architectures
    ):
        raise HardwareError(
            f"{model_spec.name!r} has architecture(s) {architectures}, which are not "
            f"causal language models; DPO training requires a *{SUPPORTED_ARCHITECTURE_SUFFIX}"
        )

    supported = getattr(config, "max_position_embeddings", None)
    if isinstance(supported, int) and supported < max_length:
        raise HardwareError(
            f"{model_spec.name!r} supports a maximum sequence length of {supported}, "
            f"but training.max_length is {max_length}; lower max_length"
        )


def validate_target_modules(model: Any, configured: tuple[str, ...]) -> tuple[str, ...]:
    """Check configured LoRA targets against the real module tree (spec section 16).

    Returns the subset that exists. Raises when **none** match — silently training with
    zero LoRA targets would produce an adapter that changes nothing.
    """
    present: set[str] = set()
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in configured:
            present.add(leaf)

    missing = tuple(sorted(set(configured) - present))
    if not present:
        raise TargetModuleError(
            f"none of the configured LoRA target modules {list(configured)} exist on this "
            "model; inspect the architecture and set lora.target_modules to real module "
            "names"
        )
    if missing:
        logger.warning(
            "Configured LoRA target module(s) %s do not exist on this model; "
            "continuing with %s",
            ", ".join(missing),
            ", ".join(sorted(present)),
        )
    return tuple(sorted(present))


def apply_lora(model: Any, settings: LoraSettings, target_modules: tuple[str, ...]):
    """Attach the LoRA adapter (spec sections 14, 17, 18)."""
    backend = import_backend()
    peft = backend["peft"]

    lora_config = peft.LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        target_modules=list(target_modules),
        task_type=settings.task_type,
    )
    return peft.get_peft_model(model, lora_config), lora_config


def build_lora_config(settings: LoraSettings, target_modules: tuple[str, ...]):
    """A PEFT ``LoraConfig`` without attaching it — for handing to ``DPOTrainer``."""
    backend = import_backend()
    peft = backend["peft"]
    return peft.LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        target_modules=list(target_modules),
        task_type=settings.task_type,
    )


def count_parameters(model: Any) -> ParameterCounts:
    """Count total and trainable parameters, and refuse a full fine-tune (19, 20).

    This is the single most important check in the stage. ``trainable == total`` means the
    base model was never frozen, so the run would be a full fine-tune of a quantized model
    — expensive, wrong, and explicitly forbidden by spec section 3.
    """
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count

    if total == 0:
        raise FullFineTuneError("the model reports zero parameters; refusing to train")
    if trainable == 0:
        raise FullFineTuneError(
            "no parameters are trainable; LoRA did not attach, so training would be a "
            "no-op"
        )
    if trainable == total:
        raise FullFineTuneError(
            f"every one of the {total:,} parameters is trainable, so the base model is "
            "not frozen. This would be a full fine-tune, which Step 9 must never perform; "
            "check that LoRA was applied"
        )

    counts = ParameterCounts(total=total, trainable=trainable)
    logger.info(
        "Total parameters: %s | Trainable parameters: %s | Trainable percentage: %.4f%%",
        f"{counts.total:,}",
        f"{counts.trainable:,}",
        counts.percentage,
    )
    return counts


def resolve_optimizer(name: str) -> tuple[str, bool]:
    """Return ``(optimizer_name, fell_back)`` (spec section 50).

    ``paged_adamw_8bit`` needs a working bitsandbytes; when it is unavailable the caller
    gets plain AdamW and the substitution is recorded rather than hidden, because it
    changes the memory profile of the run.
    """
    from .config import FALLBACK_OPTIMIZER

    if not name.startswith("paged_") and "8bit" not in name:
        return name, False
    try:
        import bitsandbytes  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken CUDA build raises more than ImportError
        logger.warning(
            "bitsandbytes is unavailable; falling back from %s to %s", name, FALLBACK_OPTIMIZER
        )
        return FALLBACK_OPTIMIZER, True
    return name, False


__all__ = [
    "SUPPORTED_ARCHITECTURE_SUFFIX",
    "ParameterCounts",
    "TokenizerInfo",
    "apply_lora",
    "build_lora_config",
    "build_quantization_config",
    "count_parameters",
    "import_backend",
    "load_model",
    "load_tokenizer",
    "resolve_optimizer",
    "resolve_torch_dtype",
    "validate_target_modules",
]
