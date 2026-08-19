"""Single-prompt and batch generation from a packaged model (spec 12 sections 39, 40).

Built entirely on :class:`~python_dpo.model_evaluation.runners.AdapterModelRunner`, which
already performs the section 15 adapter integrity checks and never falls back to the base
model on failure (spec section 118) -- this module adds only the package/JSONL plumbing
around it, per the plan's "assembly of existing parts" finding. Requires the ``model``
extra (torch/transformers/peft) at call time, never at import time (spec section 42).
"""

from __future__ import annotations

from pathlib import Path

from ..atomic_io import append_jsonl, iter_jsonl
from ..model_evaluation.config import GenerationSettings, QuantizationSettings
from ..model_evaluation.runners import AdapterModelRunner
from .errors import PackagingError
from .package import ModelPackage


def generate(
    package: ModelPackage,
    prompt: str,
    *,
    quantization: QuantizationSettings,
    generation: GenerationSettings,
    seed: int = 42,
) -> str:
    """One prompt in, one response out (spec section 39)."""
    runner = AdapterModelRunner(
        model_name=package.base_model_name,
        model_revision=package.base_model_revision,
        adapter_dir=package.adapter_dir,
        quantization=quantization,
        generation=generation,
    )
    runner.ensure_loaded()
    try:
        return runner.generate(prompt, seed=seed).text
    finally:
        runner.unload()


def generate_batch(
    package: ModelPackage,
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    quantization: QuantizationSettings,
    generation: GenerationSettings,
    seed: int = 42,
) -> int:
    """Every ``{"id": ..., "prompt": ...}`` line of ``input_jsonl`` in, one matching
    ``{"id", "prompt", "response", "generated_tokens"}`` line per input in ``output_jsonl``
    (spec section 40). The model loads once and serves every prompt (spec section 17).

    Returns the number of prompts generated for.
    """
    input_jsonl = Path(input_jsonl)
    output_jsonl = Path(output_jsonl)
    if not input_jsonl.is_file():
        raise PackagingError(f"no input JSONL at {input_jsonl}")
    if output_jsonl.exists():
        output_jsonl.unlink()

    runner = AdapterModelRunner(
        model_name=package.base_model_name,
        model_revision=package.base_model_revision,
        adapter_dir=package.adapter_dir,
        quantization=quantization,
        generation=generation,
    )
    runner.ensure_loaded()
    count = 0
    try:
        for line_number, record in iter_jsonl(input_jsonl):
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise PackagingError(f"{input_jsonl}:{line_number}: missing or empty 'prompt'")
            result = runner.generate(prompt, seed=seed)
            append_jsonl(
                output_jsonl,
                {
                    "id": record.get("id", line_number),
                    "prompt": prompt,
                    "response": result.text,
                    "generated_tokens": result.generated_tokens,
                },
            )
            count += 1
    finally:
        runner.unload()
    return count


__all__ = ["generate", "generate_batch"]
