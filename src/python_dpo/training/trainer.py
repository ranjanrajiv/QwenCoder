"""DPO/QLoRA training orchestration (spec 09 sections 64-67, 82-85).

``--dry-run``, ``--smoke-test`` and a real run are the **same code path stopped at
different points**, which is the whole value of the dry run: it exercises the code that
will actually train, not a parallel implementation that might diverge from it.

```
preflight   hardware -> versions -> dataset -> tokenizer -> lengths/truncation
            -> model -> quantization -> LoRA -> parameter counts
dry_run     stop here (section 65)
smoke_test  subset to a handful of examples, cap the steps (section 66)
train       DPOConfig -> DPOTrainer -> train -> save adapter -> reload (67, 74)
```

Two TRL 1.10 realities shape this module, both verified by introspection rather than
assumed:

* ``DPOConfig`` has **no** ``max_prompt_length``; prompt and completion are truncated
  together against ``max_length``. Ours drives the preflight length analysis instead.
* ``DPOConfig`` has **no** ``warmup_ratio``, only ``warmup_steps``. The configured ratio
  is converted once the schedule length is known.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .callbacks import JsonlMetricsRecorder, build_metrics_callback, reset_peak_memory_stats
from .config import ExperimentConfig
from .dataset import PreferenceRecord, TrainingDataset
from .errors import HardwareError, TrainingError
from .hardware import HardwareInfo, HardwareReport, check_hardware, resolve_compute_dtype
from .lengths import LengthAnalysis, analyze_lengths, enforce_truncation_threshold
from .loader import (
    ParameterCounts,
    TokenizerInfo,
    build_lora_config,
    count_parameters,
    import_backend,
    load_model,
    load_tokenizer,
    resolve_optimizer,
    validate_target_modules,
)
from .run_repository import TrainingRunRepository
from .verify import ReloadResult, verify_adapter

logger = logging.getLogger("python_dpo.training.trainer")

TRAINER_VERSION = "v1"

# Spec section 66: a smoke test is about proving the stack works end to end, not learning.
SMOKE_TEST_TRAIN_EXAMPLES = 4
SMOKE_TEST_EVAL_EXAMPLES = 2
SMOKE_TEST_MAX_STEPS = 2


@dataclass
class PreflightResult:
    """Everything preflight established, and the objects it built."""

    hardware_report: HardwareReport
    compute_dtype: str
    tokenizer: Any = None
    tokenizer_info: TokenizerInfo | None = None
    length_analysis: LengthAnalysis | None = None
    model: Any = None
    parameter_counts: ParameterCounts | None = None
    resolved_target_modules: tuple[str, ...] = ()
    optimizer_name: str = ""
    optimizer_fell_back: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware_report.info.to_dict(),
            "compute_dtype": self.compute_dtype,
            "tokenizer": self.tokenizer_info.to_dict() if self.tokenizer_info else None,
            "lengths": self.length_analysis.to_dict() if self.length_analysis else None,
            "parameters": (
                self.parameter_counts.to_dict() if self.parameter_counts else None
            ),
            "resolved_target_modules": list(self.resolved_target_modules),
            "optimizer": self.optimizer_name,
            "optimizer_fell_back": self.optimizer_fell_back,
        }


@dataclass
class TrainingOutcome:
    """What a completed (or stopped) job produced."""

    mode: str
    status: str
    preflight: PreflightResult
    steps: int = 0
    epochs: float = 0.0
    final_train_loss: float | None = None
    final_eval_loss: float | None = None
    reward_metrics: dict[str, float] = field(default_factory=dict)
    peak_gpu_memory_bytes: int | None = None
    adapter_path: str | None = None
    checkpoint_path: str | None = None
    reload: ReloadResult | None = None
    duration_seconds: float | None = None


class DpoTrainingJob:
    """Runs one DPO/QLoRA training job in a given mode."""

    def __init__(
        self,
        config: ExperimentConfig,
        dataset: TrainingDataset,
        repository: TrainingRunRepository,
        training_run_id: str,
        *,
        mode: str = "train",
        allow_small_dataset: bool = False,
        override_truncation: bool = False,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.repository = repository
        self.training_run_id = training_run_id
        self.mode = mode
        self.allow_small_dataset = allow_small_dataset
        self.override_truncation = override_truncation

    # ------------------------------------------------------------------- preflight

    def preflight(self) -> PreflightResult:
        """Everything that must hold before a single gradient is computed."""
        report = check_hardware()
        if not report.passed:
            detail = "; ".join(check.detail for check in report.failures)
            raise HardwareError(f"hardware check failed: {detail}")

        compute_dtype = resolve_compute_dtype(
            self.config.quantization.compute_dtype, report.info
        )
        if compute_dtype != self.config.quantization.compute_dtype:
            logger.warning(
                "Configured compute dtype %s is unsupported here; using %s instead",
                self.config.quantization.compute_dtype,
                compute_dtype,
            )

        result = PreflightResult(hardware_report=report, compute_dtype=compute_dtype)

        tokenizer, tokenizer_info = load_tokenizer(self.config.model)
        result.tokenizer = tokenizer
        result.tokenizer_info = tokenizer_info

        if self.config.training.apply_chat_template and not tokenizer_info.has_chat_template:
            # Not fatal: a base (non-instruct) checkpoint legitimately has no template.
            logger.warning(
                "training.apply_chat_template is true but %s declares no chat template; "
                "prompts will be used verbatim",
                self.config.model.name,
            )

        analysis = analyze_lengths(
            self.dataset.train + self.dataset.validation,
            tokenizer,
            max_length=self.config.training.max_length,
            max_prompt_length=self.config.training.max_prompt_length,
        )
        result.length_analysis = analysis
        enforce_truncation_threshold(
            analysis,
            max_truncation_rate=self.config.training.max_truncation_rate,
            override=self.override_truncation,
        )

        model = load_model(
            self.config.model,
            self.config.quantization,
            compute_dtype,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            max_length=self.config.training.max_length,
        )
        result.resolved_target_modules = validate_target_modules(
            model, self.config.lora.target_modules
        )

        # Attach LoRA purely to *count* parameters here. The trainer is given the base
        # model plus a peft_config and attaches its own, which is what lets TRL use the
        # adapter-disabled model as the implicit reference (sections 46, 47).
        from .loader import apply_lora

        peft_model, _ = apply_lora(model, self.config.lora, result.resolved_target_modules)
        result.parameter_counts = count_parameters(peft_model)
        result.model = peft_model

        optimizer_name, fell_back = resolve_optimizer(self.config.optimizer.name)
        result.optimizer_name = optimizer_name
        result.optimizer_fell_back = fell_back

        return result

    # ----------------------------------------------------------------------- run

    def run(self) -> TrainingOutcome:
        started = time.monotonic()
        preflight = self.preflight()

        if self.mode == "dry_run":
            logger.info("Dry run complete; no training was performed.")
            return TrainingOutcome(
                mode=self.mode,
                status="completed",
                preflight=preflight,
                duration_seconds=time.monotonic() - started,
            )

        train_records, eval_records = self._select_records()
        outcome = self._train(preflight, train_records, eval_records)
        outcome.duration_seconds = time.monotonic() - started
        return outcome

    def _select_records(self) -> tuple[list[PreferenceRecord], list[PreferenceRecord]]:
        """Spec section 66: a smoke test trains on a handful of examples."""
        train = self.dataset.train
        validation = self.dataset.validation
        if self.mode == "smoke_test":
            train = train[:SMOKE_TEST_TRAIN_EXAMPLES]
            validation = validation[:SMOKE_TEST_EVAL_EXAMPLES]
            logger.info(
                "Smoke test: %d training and %d validation example(s), max %d step(s)",
                len(train),
                len(validation),
                SMOKE_TEST_MAX_STEPS,
            )
        return train, validation

    def _to_hf_dataset(self, records: list[PreferenceRecord]):
        """Convert to a ``datasets.Dataset`` in TRL's expected shape.

        When ``apply_chat_template`` is on, ``prompt`` becomes a message list —
        TRL's *conversational* format — and TRL applies the model's own template. That
        matches how Stage 3 generated every candidate, so training sees the format the
        data was produced under (sections 30, 31).
        """
        from datasets import Dataset

        if self.config.training.apply_chat_template:
            rows = {
                "prompt": [[{"role": "user", "content": r.prompt}] for r in records],
                "chosen": [[{"role": "assistant", "content": r.chosen}] for r in records],
                "rejected": [
                    [{"role": "assistant", "content": r.rejected}] for r in records
                ],
            }
        else:
            rows = {
                "prompt": [r.prompt for r in records],
                "chosen": [r.chosen for r in records],
                "rejected": [r.rejected for r in records],
            }
        return Dataset.from_dict(rows)

    def _build_dpo_config(
        self,
        preflight: PreflightResult,
        train_count: int,
        eval_count: int,
    ):
        from trl import DPOConfig

        settings = self.config.training
        run_dir = self.repository.run_dir(self.training_run_id)

        max_steps = settings.max_steps
        epochs = settings.num_train_epochs
        if self.mode == "smoke_test":
            max_steps = min(SMOKE_TEST_MAX_STEPS, max_steps) if max_steps > 0 else SMOKE_TEST_MAX_STEPS
            epochs = 1

        # Spec section 53 asks for warmup_ratio, which TRL 1.10's DPOConfig does not
        # expose. Convert it to a step count against the real schedule length.
        steps_per_epoch = max(
            1,
            train_count
            // max(1, settings.per_device_train_batch_size * settings.gradient_accumulation_steps),
        )
        total_steps = max_steps if max_steps > 0 else steps_per_epoch * epochs
        warmup_steps = int(round(settings.warmup_ratio * total_steps))

        # Spec section 56: never evaluate against a split we do not have.
        eval_strategy = settings.eval_strategy if eval_count > 0 else "no"
        if eval_count == 0 and settings.eval_strategy != "no":
            logger.warning(
                "No validation examples available; disabling evaluation during training"
            )

        bf16 = preflight.compute_dtype == "bfloat16"
        return DPOConfig(
            output_dir=str(self.repository.checkpoints_dir(self.training_run_id)),
            beta=self.config.dpo.beta,
            loss_type=self.config.dpo.loss_type,
            max_length=settings.max_length,
            learning_rate=settings.learning_rate,
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            gradient_checkpointing=settings.gradient_checkpointing,
            warmup_steps=warmup_steps,
            max_grad_norm=settings.max_grad_norm,
            lr_scheduler_type=settings.lr_scheduler_type,
            optim=preflight.optimizer_name,
            # Spec section 51: never both at once.
            bf16=bf16,
            fp16=not bf16,
            eval_strategy=eval_strategy,
            eval_steps=settings.eval_steps if eval_strategy == "steps" else None,
            save_steps=settings.save_steps,
            save_strategy="steps",
            logging_steps=settings.logging_steps,
            seed=settings.seed,
            data_seed=settings.data_seed,
            report_to=[],
            remove_unused_columns=False,
            # Spec section 58: do not aggressively delete checkpoints. A lower DPO loss is
            # not known to mean better Python, so Step 10 chooses, not this stage.
            save_total_limit=None,
            load_best_model_at_end=False,
        )

    def _train(
        self,
        preflight: PreflightResult,
        train_records: list[PreferenceRecord],
        eval_records: list[PreferenceRecord],
    ) -> TrainingOutcome:
        from trl import DPOTrainer

        backend = import_backend()
        torch = backend["torch"]

        train_dataset = self._to_hf_dataset(train_records)
        eval_dataset = self._to_hf_dataset(eval_records) if eval_records else None

        dpo_config = self._build_dpo_config(
            preflight, len(train_records), len(eval_records)
        )
        recorder = JsonlMetricsRecorder(self.repository.metrics_path(self.training_run_id))
        reset_peak_memory_stats()

        trainer = DPOTrainer(
            model=preflight.model,
            # Spec sections 46, 47: no second full-precision model on a 12 GB GPU. With a
            # PEFT model, TRL uses the adapter-disabled base as the implicit reference.
            ref_model=None,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=preflight.tokenizer,
            callbacks=[build_metrics_callback(recorder)],
        )

        try:
            result = trainer.train()
        except torch.cuda.OutOfMemoryError as exc:
            # Spec section 83: report the whole configuration, and never silently retry
            # with something smaller.
            raise TrainingError(self._oom_message(preflight, exc)) from exc

        adapter_dir = self.repository.adapter_dir(self.training_run_id)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(adapter_dir))
        preflight.tokenizer.save_pretrained(
            str(self.repository.tokenizer_dir(self.training_run_id))
        )

        metrics = getattr(result, "metrics", {}) or {}
        final = recorder.final_metrics()
        checkpoint = self.repository.latest_checkpoint(self.training_run_id)

        # Spec sections 74, 82: the adapter is not final until it reloads. Free the
        # training model first so the reload has VRAM to work with.
        del trainer
        preflight.model = None
        torch.cuda.empty_cache()

        reload_result = verify_adapter(
            adapter_dir, self.config, compute_dtype=preflight.compute_dtype
        )

        return TrainingOutcome(
            mode=self.mode,
            status="completed",
            preflight=preflight,
            steps=int(getattr(result, "global_step", 0) or metrics.get("step", 0) or 0),
            epochs=float(metrics.get("epoch", 0.0) or 0.0),
            final_train_loss=_as_float(metrics.get("train_loss", final.get("loss"))),
            final_eval_loss=_as_float(final.get("eval_loss")),
            reward_metrics=recorder.reward_metrics(),
            peak_gpu_memory_bytes=recorder.peak_gpu_memory(),
            adapter_path=str(adapter_dir),
            checkpoint_path=str(checkpoint) if checkpoint else None,
            reload=reload_result,
        )

    def _oom_message(self, preflight: PreflightResult, exc: Exception) -> str:
        """Spec section 83's diagnostic, with everything needed to choose a smaller run."""
        settings = self.config.training
        info: HardwareInfo = preflight.hardware_report.info
        free_gib = (info.free_vram_bytes or 0) / 1024**3
        total_gib = (info.total_vram_bytes or 0) / 1024**3
        return (
            "CUDA ran out of memory during training. The configuration is not being "
            "changed automatically.\n"
            f"  model:                       {self.config.model.name}\n"
            f"  max_length:                  {settings.max_length}\n"
            f"  per_device_train_batch_size: {settings.per_device_train_batch_size}\n"
            f"  gradient_accumulation_steps: {settings.gradient_accumulation_steps}\n"
            f"  gradient_checkpointing:      {settings.gradient_checkpointing}\n"
            f"  quantization:                {self.config.quantization.bits}-bit "
            f"{self.config.quantization.quant_type}\n"
            f"  lora:                        r={self.config.lora.r} "
            f"targets={list(preflight.resolved_target_modules)}\n"
            f"  gpu:                         {info.gpu_name} "
            f"({free_gib:.1f} GiB free of {total_gib:.1f} GiB at start)\n"
            f"  underlying error:            {exc}\n"
            "Reduce training.max_length or per_device_train_batch_size and run again."
        )


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def format_exception(exc: BaseException) -> tuple[str, str, str]:
    """``(error_type, message, traceback)`` for the section 82 failure record."""
    return (
        type(exc).__name__,
        str(exc),
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


__all__ = [
    "SMOKE_TEST_EVAL_EXAMPLES",
    "SMOKE_TEST_MAX_STEPS",
    "SMOKE_TEST_TRAIN_EXAMPLES",
    "TRAINER_VERSION",
    "DpoTrainingJob",
    "PreflightResult",
    "TrainingOutcome",
    "format_exception",
]
