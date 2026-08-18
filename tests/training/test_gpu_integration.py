"""GPU integration tests for DPO/QLoRA training (spec 09 sections 67, 107).

These need a CUDA GPU and the training extra, so this module is deselected by default
(``addopts = "-ra -m 'not integration and not gpu'"`` in pyproject.toml). Run explicitly:

    pytest -q -m gpu

Mirrors ``tests/sandbox/test_sandbox_integration.py``'s philosophy: **fail loudly rather
than skip silently** when the GPU or the backend is unavailable. These were asked for
explicitly, and a quietly-unrun QLoRA suite is worse than a red one — the default
``pytest -q`` remains offline and zero-skip.

Every item on spec section 67's smoke-test acceptance list is covered here: load model,
load tokenizer, load dataset, apply QLoRA, initialize DPOTrainer, forward pass, backward
pass, update LoRA parameters, save checkpoint, reload adapter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python_dpo.training.config import ExperimentConfig
from python_dpo.training.dataset import PreferenceRecord, load_training_dataset
from python_dpo.training.hardware import check_hardware, resolve_compute_dtype
from python_dpo.training.loader import (
    apply_lora,
    count_parameters,
    load_model,
    load_tokenizer,
    validate_target_modules,
)
from python_dpo.training.run_repository import TrainingRunRepository
from python_dpo.training.trainer import DpoTrainingJob

pytestmark = pytest.mark.gpu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "dpo_qlora.yaml"

# Qwen2.5-Coder-3B-Instruct: 36 layers x (q,k,v,o) at r=16 with GQA (k/v are 2048->256).
EXPECTED_LORA_PARAMETERS = 7_372_800


@pytest.fixture(scope="session")
def hardware():
    """Fail fast, and legibly, when the suite is run without a usable GPU."""
    report = check_hardware()
    if not report.passed:
        detail = "; ".join(check.detail for check in report.failures)
        pytest.fail(
            f"gpu tests require a working CUDA GPU and the training extra: {detail}\n"
            "Install it with `pip install -e '.[training]'`, or run the offline suite "
            "with `pytest -q` instead."
        )
    return report.info


@pytest.fixture(scope="session")
def config() -> ExperimentConfig:
    return ExperimentConfig.load(CONFIG_PATH)


@pytest.fixture(scope="session")
def compute_dtype(hardware, config) -> str:
    return resolve_compute_dtype(config.quantization.compute_dtype, hardware)


@pytest.fixture(scope="session")
def tokenizer(config):
    tok, _info = load_tokenizer(config.model)
    return tok


@pytest.fixture
def quantized_model(config, compute_dtype):
    """A real 4-bit model, freed on teardown.

    Deliberately **function**-scoped despite costing ~90s to load. A session-scoped model
    holds roughly 7 GiB for the whole run, which starves the full-job tests below on a
    12 GiB card — they then fail the 6 GiB preflight floor rather than the thing they
    meant to test. Correctness over speed: there is exactly one consumer.
    """
    import torch

    model = load_model(
        config.model,
        config.quantization,
        compute_dtype,
        gradient_checkpointing=config.training.gradient_checkpointing,
        max_length=config.training.max_length,
    )
    yield model
    del model
    torch.cuda.empty_cache()


def write_dataset(tmp_path: Path, records: list[dict[str, str]]) -> Path:
    run_dir = tmp_path / "pref_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        (run_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "preference_run_id": "pref_gpu_test",
                "preference_version": "v1",
                "selection_policy": "all_better",
                "selection_policy_version": "all_better_v1",
                "dataset_schema_version": "dpo_preference_v1",
                "ranking_run_id": "rank_x",
                "evaluation_run_id": "eval_x",
                "candidate_run_id": "run_x",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "train_problem_ids": ["p001"],
                "validation_problem_ids": ["p002"],
                "test_problem_ids": ["p003"],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


# ------------------------------------------------------------------- hardware, model


def test_hardware_check_passes_on_this_machine(hardware):
    assert hardware.cuda_available
    assert hardware.free_vram_bytes > 0


def test_tokenizer_loads_with_a_pad_token_and_left_padding(config):
    tok, info = load_tokenizer(config.model)
    assert tok.pad_token is not None
    # TRL's DPO processing class expects left padding.
    assert info.padding_side == "left"
    assert info.vocab_size > 0


# ----------------------------------------------- 4-bit load and LoRA (8, 16, 19, 20)


def test_quantized_load_and_lora_application(quantized_model, config):
    """4-bit load, target-module resolution and the parameter accounting, in one test.

    Combined deliberately: each assertion needs the same 3B model, and loading it per
    assertion would triple the suite's runtime for no additional coverage.
    """
    # Section 49: gradient checkpointing requires the KV cache off.
    assert quantized_model.config.use_cache is False

    # Section 16: every configured target must exist on a real Qwen2 module tree.
    resolved = validate_target_modules(quantized_model, config.lora.target_modules)
    assert set(resolved) == set(config.lora.target_modules)

    # Sections 19, 20: LoRA attaches, and the base model stays frozen.
    peft_model, _ = apply_lora(quantized_model, config.lora, resolved)
    counts = count_parameters(peft_model)
    assert counts.trainable == EXPECTED_LORA_PARAMETERS
    assert counts.trainable < counts.total
    assert counts.percentage < 1.0


# --------------------------------------------------- section 67's acceptance list


def test_smoke_test_completes_the_whole_stack(tmp_path, config, hardware):
    """One test covering every item on spec section 67's list.

    Deliberately a single test: the items are sequential phases of one job, and splitting
    them would mean reloading a 3B model per assertion.
    """
    records = [
        {
            "prompt": "Write a Python function that adds two numbers.",
            "chosen": "def add(a, b):\n    return a + b",
            "rejected": "def add(a, b):\n    return a - b",
        },
        {
            "prompt": "Write a Python function that reverses a string.",
            "chosen": "def rev(s):\n    return s[::-1]",
            "rejected": "def rev(s):\n    return s",
        },
    ]
    run_dir = write_dataset(tmp_path, records)
    dataset = load_training_dataset(run_dir, min_training_pairs=0)

    repository = TrainingRunRepository(tmp_path / "training_runs")
    manifest = repository.create_run(
        experiment_name="gpu-smoke",
        model_name=config.model.name,
        model_revision=config.model.revision,
        tokenizer_revision=config.model.revision,
        preference_run_id=dataset.preference_run_id,
        ranking_run_id="rank_x",
        evaluation_run_id="eval_x",
        candidate_run_id="run_x",
        dataset_hashes=dataset.split_hashes,
        hardware=hardware.to_dict(),
        environment={},
        configuration=config.to_dict(),
        seed=config.training.seed,
        data_seed=config.training.data_seed,
        trainer_version="v1",
        mode="smoke_test",
    )
    repository.start_run(manifest.training_run_id)

    job = DpoTrainingJob(
        config,
        dataset,
        repository,
        manifest.training_run_id,
        mode="smoke_test",
        allow_small_dataset=True,
    )
    outcome = job.run()

    assert outcome.status == "completed"
    assert outcome.steps >= 1                       # forward + backward happened
    assert outcome.preflight.parameter_counts.trainable == EXPECTED_LORA_PARAMETERS
    assert outcome.adapter_path is not None         # adapter saved
    assert Path(outcome.adapter_path, "adapter_model.safetensors").is_file()
    assert outcome.reload is not None and outcome.reload.ok   # adapter reloaded (74)
    assert outcome.reload.generated_tokens > 0

    # A checkpoint was written (section 57).
    assert repository.latest_checkpoint(manifest.training_run_id) is not None

    # Metrics were persisted, not merely printed (sections 78, 93).
    metrics_path = repository.metrics_path(manifest.training_run_id)
    assert metrics_path.is_file()
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    assert rows, "no metrics were recorded"
    assert any("loss" in row for row in rows)


def test_lora_parameters_actually_change(tmp_path, config, hardware):
    """A training step that does not move the adapter weights has achieved nothing."""
    import torch

    records = [
        {
            "prompt": "Write a Python function that adds two numbers.",
            "chosen": "def add(a, b):\n    return a + b",
            "rejected": "def add(a, b):\n    return a - b",
        }
    ] * 2
    run_dir = write_dataset(tmp_path, records)
    dataset = load_training_dataset(run_dir, min_training_pairs=0)

    repository = TrainingRunRepository(tmp_path / "training_runs")
    manifest = repository.create_run(
        experiment_name="gpu-delta",
        model_name=config.model.name,
        model_revision=config.model.revision,
        tokenizer_revision=config.model.revision,
        preference_run_id=dataset.preference_run_id,
        ranking_run_id="rank_x",
        evaluation_run_id="eval_x",
        candidate_run_id="run_x",
        dataset_hashes=dataset.split_hashes,
        hardware=hardware.to_dict(),
        environment={},
        configuration=config.to_dict(),
        seed=config.training.seed,
        data_seed=config.training.data_seed,
        trainer_version="v1",
        mode="smoke_test",
    )
    repository.start_run(manifest.training_run_id)

    # A higher learning rate so a couple of steps produce a visible delta.
    tuned = config.with_overrides(learning_rate=1e-3)
    job = DpoTrainingJob(
        tuned,
        dataset,
        repository,
        manifest.training_run_id,
        mode="smoke_test",
        allow_small_dataset=True,
    )
    outcome = job.run()
    assert outcome.status == "completed"

    from safetensors import safe_open

    adapter_file = Path(outcome.adapter_path) / "adapter_model.safetensors"
    with safe_open(str(adapter_file), framework="pt") as handle:
        # lora_B is zero-initialized by PEFT, so any nonzero value proves an update ran.
        b_keys = [k for k in handle.keys() if "lora_B" in k]
        assert b_keys, "no lora_B tensors in the saved adapter"
        moved = any(torch.any(handle.get_tensor(k) != 0).item() for k in b_keys)
    assert moved, "LoRA B matrices are still all zero; no parameter update occurred"
