"""Merging a LoRA adapter into full-precision base weights (spec 12 sections 43, 44).

PEFT cannot merge LoRA weights into a base model held in 4-bit (bitsandbytes does not
support the in-place dequantize-and-add merge needs), so this module never reuses a
package's *training-time* quantized weights for the merge -- it reloads the base model
fresh, unquantized, in ``compute_dtype``, purely for this operation. ``adapter/`` is never
touched or deleted (spec section 44): a merge failure, or simply not wanting the merged
copy anymore, must never cost the original adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import MergeUnsupportedError, PackagingError
from .package import ModelPackage

_INSTALL_HINT = "install the training backend with: pip install -e '.[training]'"


def _import_backend() -> dict[str, Any]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise MergeUnsupportedError(f"{exc}; {_INSTALL_HINT}") from exc
    return {"torch": torch, "transformers": transformers, "peft": peft}


def _resolve_dtype(torch_module: Any, name: str):
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if name not in mapping:
        raise PackagingError(f"unsupported compute dtype {name!r}")
    return mapping[name]


def merge_adapter(package: ModelPackage, dest_dir: Path, *, compute_dtype: str = "bfloat16") -> Path:
    """Reload the base model unquantized, apply the adapter, merge, and save the result
    to ``dest_dir``. Returns ``dest_dir``. Raises :class:`MergeUnsupportedError` if the
    backend is unavailable or the merge itself fails for any reason.
    """
    dest_dir = Path(dest_dir)
    backend = _import_backend()
    transformers = backend["transformers"]
    peft = backend["peft"]
    torch = backend["torch"]

    dtype = _resolve_dtype(torch, compute_dtype)

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            package.base_model_name, revision=package.base_model_revision, trust_remote_code=False
        )
        base = transformers.AutoModelForCausalLM.from_pretrained(
            package.base_model_name,
            revision=package.base_model_revision,
            quantization_config=None,
            dtype=dtype,
            device_map={"": 0},
            trust_remote_code=False,
        )
        model = peft.PeftModel.from_pretrained(base, str(package.adapter_dir))
        merged = model.merge_and_unload()
    except Exception as exc:  # noqa: BLE001 - any failure means this stack cannot merge safely
        raise MergeUnsupportedError(
            f"could not merge adapter {package.adapter_dir} into {package.base_model_name}: {exc}"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(dest_dir))
    tokenizer.save_pretrained(str(dest_dir))
    return dest_dir


__all__ = ["merge_adapter"]
