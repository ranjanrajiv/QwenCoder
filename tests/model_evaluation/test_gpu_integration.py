"""GPU integration tests for the Stage 10 inference runners (spec sections 15, 17-19,
118-119).

These need a CUDA GPU and the training extra, so this module is deselected by default.
Run explicitly:

    pytest -q -m gpu

Mirrors ``tests/training/test_gpu_integration.py``'s philosophy: fail loudly rather than
skip silently when the GPU or backend is unavailable.

Covers spec section 118's adapter isolation guarantee by inspecting the *loaded module
tree* for LoRA layers, not by comparing generated text (which spec section 119 says may
legitimately be identical for a lightly-trained adapter) -- and a one-sample smoke
generation per variant, proving the whole seeded/quantized/chat-templated path actually
produces tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python_dpo.model_evaluation.config import GenerationSettings, QuantizationSettings
from python_dpo.model_evaluation.runners import AdapterModelRunner, BaseModelRunner

pytestmark = pytest.mark.gpu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_RUN_DIR = PROJECT_ROOT / "data" / "training" / "runs" / "dpo_20260818_081231_a91d"
ADAPTER_DIR = TRAINING_RUN_DIR / "adapter"
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"

QUANTIZATION = QuantizationSettings(enabled=True, bits=4, quant_type="nf4", double_quant=True, compute_dtype="bfloat16")
GENERATION = GenerationSettings(temperature=0.2, max_new_tokens=16, num_samples=1, base_seed=1000)

SMOKE_PROMPT = "Write a Python function that returns the square of a number."


@pytest.fixture(scope="module", autouse=True)
def require_training_run():
    manifest_path = TRAINING_RUN_DIR / "manifest.json"
    if not manifest_path.is_file():
        pytest.fail(
            f"GPU integration tests require the committed Stage 9 training run at "
            f"{TRAINING_RUN_DIR}, which is missing manifest.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_name"] == MODEL_NAME, "MODEL_NAME must match the training run's base model"


def _has_lora_layers(model) -> bool:
    return any("lora" in name.lower() for name, _module in model.named_modules())


def test_base_runner_never_loads_lora_layers():
    """Spec section 118: BaseModelRunner has no adapter-loading code path at all."""
    runner = BaseModelRunner(
        model_name=MODEL_NAME, model_revision=None, quantization=QUANTIZATION, generation=GENERATION
    )
    try:
        runner.ensure_loaded()
        assert _has_lora_layers(runner._model) is False
    finally:
        runner.unload()


def test_adapter_runner_loads_lora_layers():
    """Spec section 118: AdapterModelRunner never silently falls back to the base model."""
    runner = AdapterModelRunner(
        model_name=MODEL_NAME,
        model_revision=None,
        adapter_dir=ADAPTER_DIR,
        quantization=QUANTIZATION,
        generation=GENERATION,
    )
    try:
        runner.ensure_loaded()
        assert _has_lora_layers(runner._model) is True
    finally:
        runner.unload()


def test_adapter_runner_rejects_a_corrupted_adapter_directory(tmp_path):
    """Spec section 15: never falls back to the base model on integrity failure."""
    from python_dpo.model_evaluation.errors import AdapterIntegrityError

    empty_dir = tmp_path / "empty_adapter"
    empty_dir.mkdir()
    runner = AdapterModelRunner(
        model_name=MODEL_NAME,
        model_revision=None,
        adapter_dir=empty_dir,
        quantization=QUANTIZATION,
        generation=GENERATION,
    )
    with pytest.raises(AdapterIntegrityError):
        runner.ensure_loaded()
    assert runner.loaded is False


def test_base_runner_smoke_generation_produces_tokens():
    runner = BaseModelRunner(
        model_name=MODEL_NAME, model_revision=None, quantization=QUANTIZATION, generation=GENERATION
    )
    try:
        runner.ensure_loaded()
        result = runner.generate(SMOKE_PROMPT, seed=1000)
        assert result.generated_tokens > 0
        assert isinstance(result.text, str)
    finally:
        runner.unload()


def test_adapter_runner_smoke_generation_produces_tokens():
    """Spec section 119: base and DPO output may legitimately be identical; only that
    both variants independently produce a real response is asserted here."""
    runner = AdapterModelRunner(
        model_name=MODEL_NAME,
        model_revision=None,
        adapter_dir=ADAPTER_DIR,
        quantization=QUANTIZATION,
        generation=GENERATION,
    )
    try:
        runner.ensure_loaded()
        result = runner.generate(SMOKE_PROMPT, seed=1000)
        assert result.generated_tokens > 0
        assert isinstance(result.text, str)
    finally:
        runner.unload()


def test_base_and_dpo_seeded_generation_are_independently_recorded():
    """Spec section 119: identical output is not an error -- both are recorded regardless."""
    base_runner = BaseModelRunner(
        model_name=MODEL_NAME, model_revision=None, quantization=QUANTIZATION, generation=GENERATION
    )
    try:
        base_runner.ensure_loaded()
        base_result = base_runner.generate(SMOKE_PROMPT, seed=1000)
    finally:
        base_runner.unload()

    dpo_runner = AdapterModelRunner(
        model_name=MODEL_NAME,
        model_revision=None,
        adapter_dir=ADAPTER_DIR,
        quantization=QUANTIZATION,
        generation=GENERATION,
    )
    try:
        dpo_runner.ensure_loaded()
        dpo_result = dpo_runner.generate(SMOKE_PROMPT, seed=1000)
    finally:
        dpo_runner.unload()

    assert base_result.generated_tokens > 0
    assert dpo_result.generated_tokens > 0
