"""DPO/QLoRA training: fine-tuning a LoRA adapter over a frozen Qwen Coder base.

This package turns Stage 8's ``{prompt, chosen, rejected}`` preference data into a trained
LoRA adapter, using 4-bit NF4 quantization and TRL's ``DPOTrainer``. Three rules govern it:

* **The base model stays frozen.** Only LoRA parameters train, and
  :func:`~python_dpo.training.loader.count_parameters` refuses to proceed if every
  parameter is trainable — that would be a full fine-tune, which this stage must never
  perform by accident (spec 09 sections 3, 20).
* **No candidate code is executed.** ``chosen``/``rejected`` are validated to be strings
  and are only ever tokenized. Executing generated code belongs exclusively to the Stage 5
  sandbox and Stage 6 evaluator (sections 100, 101).
* **No performance claim.** A falling DPO loss is not evidence of better Python. This
  stage produces ``base model + adapter``; Step 10 evaluates it (sections 95, 110).

**Import discipline.** Nothing here imports torch, transformers, trl, peft, bitsandbytes
or datasets at module scope — every heavy import is deferred into the function that needs
it, so ``import python_dpo`` stays cheap. ``tests/test_no_heavy_imports.py`` enforces this.
"""

from .callbacks import (
    JsonlMetricsRecorder,
    build_metrics_callback,
    gpu_memory_snapshot,
    reset_peak_memory_stats,
)
from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OPTIMIZER,
    FALLBACK_OPTIMIZER,
    DatasetSpec,
    DistributedSettings,
    DpoSettings,
    ExperimentConfig,
    LoraSettings,
    ModelSpec,
    OptimizerSettings,
    QuantizationSettings,
    TrainingSettings,
)
from .dataset import (
    PreferenceBalance,
    PreferenceRecord,
    SplitStatistics,
    TrainingDataset,
    load_training_dataset,
)
from .errors import (
    AdapterReloadError,
    CheckpointCompatibilityError,
    DatasetValidationError,
    FullFineTuneError,
    HardwareError,
    TargetModuleError,
    TrainingConfigError,
    TrainingDependencyError,
    TrainingError,
    TrainingRunError,
    TrainingRunNotFoundError,
    TrainingStoreError,
    TruncationThresholdError,
)
from .hardware import (
    MIN_FREE_VRAM_BYTES,
    HardwareCheck,
    HardwareInfo,
    HardwareReport,
    TorchHardwareProbe,
    check_hardware,
    format_hardware_report,
    resolve_compute_dtype,
)
from .lengths import (
    LengthAnalysis,
    LengthDistribution,
    analyze_lengths,
    enforce_truncation_threshold,
)
from .loader import (
    ParameterCounts,
    TokenizerInfo,
    apply_lora,
    build_lora_config,
    build_quantization_config,
    count_parameters,
    import_backend,
    load_model,
    load_tokenizer,
    resolve_optimizer,
    validate_target_modules,
)
from .models import (
    TRAINING_RUN_STATUSES,
    TRAINING_RUN_STATUS_TRANSITIONS,
    DatasetManifest,
    FinalReport,
    TrainingManifest,
    TrainingModelError,
    utc_now_iso,
)
from .run_repository import (
    ADAPTER_DIRNAME,
    CHECKPOINTS_DIRNAME,
    MANIFEST_FILENAME,
    TrainingRunRepository,
    read_metrics,
)
from .statistics import (
    format_dataset_statistics,
    format_final_report,
    format_length_analysis,
    format_parameter_counts,
    format_run_table,
)
from .trainer import (
    TRAINER_VERSION,
    DpoTrainingJob,
    PreflightResult,
    TrainingOutcome,
    format_exception,
)
from .verify import (
    BASELINE_PROMPTS,
    VERIFICATION_PROMPT,
    ReloadResult,
    baseline_responses,
    run_inference,
    verify_adapter,
)

__all__ = [
    "ADAPTER_DIRNAME",
    "BASELINE_PROMPTS",
    "CHECKPOINTS_DIRNAME",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OPTIMIZER",
    "FALLBACK_OPTIMIZER",
    "MANIFEST_FILENAME",
    "MIN_FREE_VRAM_BYTES",
    "TRAINER_VERSION",
    "TRAINING_RUN_STATUSES",
    "TRAINING_RUN_STATUS_TRANSITIONS",
    "VERIFICATION_PROMPT",
    "AdapterReloadError",
    "CheckpointCompatibilityError",
    "DatasetManifest",
    "DatasetSpec",
    "DatasetValidationError",
    "DistributedSettings",
    "DpoSettings",
    "DpoTrainingJob",
    "ExperimentConfig",
    "FinalReport",
    "FullFineTuneError",
    "HardwareCheck",
    "HardwareError",
    "HardwareInfo",
    "HardwareReport",
    "JsonlMetricsRecorder",
    "LengthAnalysis",
    "LengthDistribution",
    "LoraSettings",
    "ModelSpec",
    "OptimizerSettings",
    "ParameterCounts",
    "PreferenceBalance",
    "PreferenceRecord",
    "PreflightResult",
    "QuantizationSettings",
    "ReloadResult",
    "SplitStatistics",
    "TargetModuleError",
    "TokenizerInfo",
    "TorchHardwareProbe",
    "TrainingConfigError",
    "TrainingDataset",
    "TrainingDependencyError",
    "TrainingError",
    "TrainingManifest",
    "TrainingModelError",
    "TrainingOutcome",
    "TrainingRunError",
    "TrainingRunNotFoundError",
    "TrainingRunRepository",
    "TrainingSettings",
    "TrainingStoreError",
    "TruncationThresholdError",
    "analyze_lengths",
    "apply_lora",
    "baseline_responses",
    "build_lora_config",
    "build_metrics_callback",
    "build_quantization_config",
    "check_hardware",
    "count_parameters",
    "enforce_truncation_threshold",
    "format_dataset_statistics",
    "format_exception",
    "format_final_report",
    "format_hardware_report",
    "format_length_analysis",
    "format_parameter_counts",
    "format_run_table",
    "gpu_memory_snapshot",
    "import_backend",
    "load_model",
    "load_tokenizer",
    "load_training_dataset",
    "read_metrics",
    "reset_peak_memory_stats",
    "resolve_compute_dtype",
    "resolve_optimizer",
    "run_inference",
    "utc_now_iso",
    "validate_target_modules",
    "verify_adapter",
]
