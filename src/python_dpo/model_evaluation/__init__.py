"""Stage 10 — Base vs DPO model evaluation.

Answers, with sandboxed execution evidence rather than training-loss numbers, whether the
Stage 9 DPO adapter actually improved Python programming correctness over the base Qwen
model on held-out problems. See ``.claude/specs/10_model_evaluation.md`` and
``.claude/plans/10_model_evaluation_plan.md``.
"""

from __future__ import annotations

from .benchmark import (
    Benchmark,
    BenchmarkManifest,
    build_benchmark,
    check_leakage,
    compute_dataset_hash,
    load_benchmark,
    save_benchmark,
)
from .cache import EvaluationCacheKey, GenerationCacheKey, JsonCacheStore
from .comparison import ComparisonResult, ProblemComparison, compare
from .config import (
    DEFAULT_BENCHMARK_NAME,
    DEFAULT_CONFIG_PATH,
    BenchmarkSpec,
    EvaluationExperimentConfig,
    GenerationSettings,
    QuantizationSettings,
    StatisticsSettings,
    SuccessCriteria,
)
from .errors import (
    AdapterIntegrityError,
    BenchmarkError,
    BenchmarkLeakageError,
    EvaluationRunError,
    EvaluationRunNotFoundError,
    EvaluationStoreError,
    IncompleteBenchmarkError,
    ModelEvaluationConfigError,
    ModelEvaluationError,
    ModelIdentityError,
    PairingError,
)
from .evaluation import EvaluationDriver
from .generation import GenerationDriver, compute_seed
from .metrics import (
    TEST_FAILURE_BUCKETS,
    execution_success_rate,
    generation_failure_rate,
    mean_pass_at_k,
    mean_test_pass_rate,
    pass_at_k,
    solve_rate,
    syntax_success_rate,
    test_failure_distribution,
    timeout_rate,
)
from .models import (
    EVALUATION_ERROR_TYPES,
    GENERATION_STATUSES,
    MODEL_VARIANTS,
    EvaluationRecord,
    GenerationRecord,
    MetricsSummary,
    ModelEvaluationManifest,
    ModelEvaluationModelError,
)
from .report import (
    SuccessEvaluation,
    build_comparison_report_json,
    build_failure_analysis,
    build_metrics_summary,
    classify_problem_examples,
    evaluate_success_criteria,
    render_executive_summary,
    render_markdown_report,
)
from .runners import (
    AdapterModelRunner,
    BaseModelRunner,
    Generation,
    ModelDependencyError,
    verify_adapter_integrity,
)
from .run_repository import MANIFEST_FILENAME, ModelEvaluationRunRepository, run_log_file
from .statistics import BootstrapResult, McNemarResult, bootstrap_ci, mcnemar, paired_bootstrap

MODEL_EVALUATION_VERSION = "v1"

__all__ = [
    "DEFAULT_BENCHMARK_NAME",
    "DEFAULT_CONFIG_PATH",
    "EVALUATION_ERROR_TYPES",
    "GENERATION_STATUSES",
    "MANIFEST_FILENAME",
    "MODEL_EVALUATION_VERSION",
    "MODEL_VARIANTS",
    "TEST_FAILURE_BUCKETS",
    "AdapterIntegrityError",
    "AdapterModelRunner",
    "BaseModelRunner",
    "Benchmark",
    "BenchmarkError",
    "BenchmarkLeakageError",
    "BenchmarkManifest",
    "BenchmarkSpec",
    "BootstrapResult",
    "ComparisonResult",
    "EvaluationCacheKey",
    "EvaluationDriver",
    "EvaluationExperimentConfig",
    "EvaluationRecord",
    "EvaluationRunError",
    "EvaluationRunNotFoundError",
    "EvaluationStoreError",
    "Generation",
    "GenerationCacheKey",
    "GenerationDriver",
    "GenerationRecord",
    "GenerationSettings",
    "IncompleteBenchmarkError",
    "JsonCacheStore",
    "McNemarResult",
    "MetricsSummary",
    "ModelDependencyError",
    "ModelEvaluationConfigError",
    "ModelEvaluationError",
    "ModelEvaluationManifest",
    "ModelEvaluationModelError",
    "ModelEvaluationRunRepository",
    "ModelIdentityError",
    "PairingError",
    "ProblemComparison",
    "QuantizationSettings",
    "StatisticsSettings",
    "SuccessCriteria",
    "SuccessEvaluation",
    "bootstrap_ci",
    "build_benchmark",
    "build_comparison_report_json",
    "build_failure_analysis",
    "build_metrics_summary",
    "check_leakage",
    "classify_problem_examples",
    "compare",
    "compute_dataset_hash",
    "compute_seed",
    "evaluate_success_criteria",
    "execution_success_rate",
    "generation_failure_rate",
    "load_benchmark",
    "mcnemar",
    "mean_pass_at_k",
    "mean_test_pass_rate",
    "paired_bootstrap",
    "pass_at_k",
    "render_executive_summary",
    "render_markdown_report",
    "run_log_file",
    "save_benchmark",
    "solve_rate",
    "syntax_success_rate",
    "test_failure_distribution",
    "timeout_rate",
    "verify_adapter_integrity",
]
