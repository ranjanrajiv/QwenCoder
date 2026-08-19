"""Shared fixtures for the packaging test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from python_dpo.training.run_repository import TrainingRunRepository


def make_training_run(
    tmp_path: Path,
    *,
    training_run_id: str = "dpo_20260819_000000_aaaa",
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    quantization: dict[str, Any] | None = None,
    with_adapter: bool = True,
    with_tokenizer: bool = True,
) -> tuple[TrainingRunRepository, str]:
    """A real :class:`TrainingRunRepository` run, with a fake (but shaped-correctly)
    adapter and tokenizer on disk, so packaging code can operate on it without torch."""
    repo = TrainingRunRepository(tmp_path / "training" / "runs")
    quantization = quantization if quantization is not None else {
        "enabled": True, "bits": 4, "quant_type": "nf4", "double_quant": True,
        "compute_dtype": "bfloat16",
    }
    repo.create_run(
        experiment_name="test-experiment",
        model_name=model_name,
        model_revision=None,
        tokenizer_revision=None,
        preference_run_id="pref_x",
        ranking_run_id="rank_x",
        evaluation_run_id="eval_x",
        candidate_run_id="run_x",
        dataset_hashes={},
        hardware={},
        environment={},
        configuration={"quantization": quantization},
        seed=42,
        data_seed=42,
        trainer_version="test",
        training_run_id=training_run_id,
    )

    if with_adapter:
        adapter_dir = repo.adapter_dir(training_run_id)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": model_name, "peft_type": "LORA"}), encoding="utf-8"
        )
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-weights")

    if with_tokenizer:
        tokenizer_dir = repo.tokenizer_dir(training_run_id)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    return repo, training_run_id


@pytest.fixture
def training_run(tmp_path: Path) -> tuple[TrainingRunRepository, str]:
    return make_training_run(tmp_path)
