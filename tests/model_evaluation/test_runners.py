"""Offline tests for adapter integrity checking (spec section 15).

``verify_adapter_integrity`` touches no torch/transformers/peft, so its failure modes are
fully testable without a GPU -- exercised against a corrupted adapter directory here.
GPU-gated adapter *loading* and isolation live in ``test_gpu_integration.py``.
"""

from __future__ import annotations

import json

import pytest

from python_dpo.model_evaluation.errors import AdapterIntegrityError
from python_dpo.model_evaluation.runners import verify_adapter_integrity

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"


def write_adapter_config(path, base_model_name: str = MODEL_NAME) -> None:
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model_name, "peft_type": "LORA"}),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"not real weights, just a marker")


def test_missing_adapter_directory_raises(tmp_path):
    with pytest.raises(AdapterIntegrityError, match="no adapter directory"):
        verify_adapter_integrity(tmp_path / "does_not_exist", MODEL_NAME)


def test_missing_adapter_config_raises(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    with pytest.raises(AdapterIntegrityError, match="adapter_config.json"):
        verify_adapter_integrity(adapter_dir, MODEL_NAME)


def test_missing_weights_file_raises(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": MODEL_NAME}), encoding="utf-8"
    )
    with pytest.raises(AdapterIntegrityError, match="weights"):
        verify_adapter_integrity(adapter_dir, MODEL_NAME)


def test_base_model_mismatch_raises(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    write_adapter_config(adapter_dir, base_model_name="Qwen/SomeOtherModel-7B")
    with pytest.raises(AdapterIntegrityError, match="SomeOtherModel"):
        verify_adapter_integrity(adapter_dir, MODEL_NAME)


def test_malformed_json_raises(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(AdapterIntegrityError, match="invalid JSON"):
        verify_adapter_integrity(adapter_dir, MODEL_NAME)


def test_valid_adapter_passes_and_returns_config(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    write_adapter_config(adapter_dir)
    config = verify_adapter_integrity(adapter_dir, MODEL_NAME)
    assert config["base_model_name_or_path"] == MODEL_NAME


def test_real_committed_adapter_passes_integrity_check():
    """The Stage 9 adapter this evaluation stage actually consumes."""
    from pathlib import Path

    adapter_dir = (
        Path(__file__).resolve().parents[2]
        / "data" / "training" / "runs" / "dpo_20260818_081231_a91d" / "adapter"
    )
    assert adapter_dir.is_dir(), f"expected the committed Stage 9 adapter at {adapter_dir}"
    config = verify_adapter_integrity(adapter_dir, MODEL_NAME)
    assert config["peft_type"] == "LORA"
