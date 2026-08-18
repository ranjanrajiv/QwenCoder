"""Adapter reload verification and inference (spec 09 sections 74, 75, 97).

Section 74 makes the reload test **mandatory**, and section 82 explains why: an adapter
that cannot be reloaded is not a training artifact, whatever the loss curve said. A run
is not successful until its output has been loaded back from disk and generated with.

All heavy imports are deferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .errors import AdapterReloadError
from .loader import import_backend, resolve_torch_dtype

logger = logging.getLogger("python_dpo.training.verify")

# A short, harmless prompt. Deliberately not one of the dataset's problems: this checks
# that the stack generates at all, and reusing a training problem would invite reading it
# as a quality signal, which section 95 forbids.
VERIFICATION_PROMPT = "Write a Python function that returns the square of a number."

# Spec section 97's fixed smoke-test prompt set for the pre-training baseline.
BASELINE_PROMPTS = (
    "Write a Python function that reverses a string.",
    "Write a Python function that returns the sum of a list of integers.",
)


@dataclass(frozen=True)
class ReloadResult:
    """The outcome of a section 74 adapter reload."""

    ok: bool
    adapter_path: str
    prompt: str
    response: str
    generated_tokens: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "adapter_path": self.adapter_path,
            "prompt": self.prompt,
            "response": self.response,
            "generated_tokens": self.generated_tokens,
            "detail": self.detail,
        }


def _render_prompt(tokenizer: Any, prompt: str, *, apply_chat_template: bool) -> str:
    """Apply the model's chat template, matching Stage 3 generation and training.

    Stage 3 wrapped every generation prompt in the tokenizer's chat template. Verification
    must do the same, or it measures the model under a format it was never trained or
    generated under (spec sections 30, 31).
    """
    template = getattr(tokenizer, "chat_template", None)
    if not apply_chat_template or not template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    apply_chat_template: bool,
    max_new_tokens: int,
) -> tuple[str, int]:
    backend = import_backend()
    torch = backend["torch"]

    rendered = _render_prompt(tokenizer, prompt, apply_chat_template=apply_chat_template)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_tokens = inputs["input_ids"].shape[-1]
    generated = output[0][prompt_tokens:]
    return tokenizer.decode(generated, skip_special_tokens=True), int(generated.shape[-1])


def verify_adapter(
    adapter_dir: Path,
    config: ExperimentConfig,
    *,
    compute_dtype: str = "bfloat16",
    prompt: str = VERIFICATION_PROMPT,
    max_new_tokens: int = 64,
) -> ReloadResult:
    """Load base + adapter from disk and generate (spec section 74).

    Deliberately loads the base model **fresh** rather than reusing the in-memory training
    model: the point is to prove the *saved artifact* works, which an in-memory handle
    would not establish.
    """
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise AdapterReloadError(f"no adapter directory at {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise AdapterReloadError(
            f"{adapter_dir} has no adapter_config.json; the adapter was not saved in "
            "PEFT format"
        )

    backend = import_backend()
    transformers = backend["transformers"]
    peft = backend["peft"]

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model.name, revision=config.model.revision, trust_remote_code=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        if config.quantization.enabled:
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.quantization.quant_type,
                bnb_4bit_compute_dtype=resolve_torch_dtype(compute_dtype),
                bnb_4bit_use_double_quant=config.quantization.double_quant,
            )

        base = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name,
            revision=config.model.revision,
            quantization_config=quantization_config,
            dtype=resolve_torch_dtype(compute_dtype),
            device_map={"": 0},
            trust_remote_code=False,
        )
        model = peft.PeftModel.from_pretrained(base, str(adapter_dir))
        model.eval()

        response, tokens = _generate(
            model,
            tokenizer,
            prompt,
            apply_chat_template=config.training.apply_chat_template,
            max_new_tokens=max_new_tokens,
        )
    except AdapterReloadError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure here means the artifact is bad
        raise AdapterReloadError(
            f"the adapter at {adapter_dir} could not be reloaded and generated from: {exc}"
        ) from exc

    if tokens == 0:
        raise AdapterReloadError(
            f"the adapter at {adapter_dir} reloaded but generated zero tokens"
        )

    logger.info("Adapter reload successful (%d token(s) generated).", tokens)
    return ReloadResult(
        ok=True,
        adapter_path=str(adapter_dir),
        prompt=prompt,
        response=response,
        generated_tokens=tokens,
        detail="reloaded and generated successfully",
    )


def run_inference(
    adapter_dir: Path,
    config: ExperimentConfig,
    prompt: str,
    *,
    compute_dtype: str = "bfloat16",
    max_new_tokens: int = 256,
) -> str:
    """Base model + trained adapter, generating a response (spec section 75)."""
    result = verify_adapter(
        adapter_dir,
        config,
        compute_dtype=compute_dtype,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    return result.response


def baseline_responses(
    config: ExperimentConfig,
    *,
    compute_dtype: str = "bfloat16",
    prompts: tuple[str, ...] = BASELINE_PROMPTS,
    max_new_tokens: int = 64,
) -> dict[str, str]:
    """The untrained base model's answers to a fixed prompt set (spec section 97).

    Recorded for reference only. Section 97 is explicit that this is **not** a benchmark —
    the real evaluation is Step 10's job.
    """
    backend = import_backend()
    transformers = backend["transformers"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.name, revision=config.model.revision, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config.quantization.enabled:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_compute_dtype=resolve_torch_dtype(compute_dtype),
            bnb_4bit_use_double_quant=config.quantization.double_quant,
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        quantization_config=quantization_config,
        dtype=resolve_torch_dtype(compute_dtype),
        device_map={"": 0},
        trust_remote_code=False,
    )
    model.eval()

    responses: dict[str, str] = {}
    for prompt in prompts:
        response, _ = _generate(
            model,
            tokenizer,
            prompt,
            apply_chat_template=config.training.apply_chat_template,
            max_new_tokens=max_new_tokens,
        )
        responses[prompt] = response
    return responses


__all__ = [
    "BASELINE_PROMPTS",
    "VERIFICATION_PROMPT",
    "ReloadResult",
    "baseline_responses",
    "run_inference",
    "verify_adapter",
]
