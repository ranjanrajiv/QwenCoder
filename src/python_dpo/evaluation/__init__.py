"""Candidate test execution: turning a problem's declared tests into objective evidence.

This package answers *"what happened when this candidate was tested against its
problem's test suite?"* — never *"is this the best candidate?"* (spec 06 section 2). It
never produces ``chosen``/``rejected`` preference pairs, a reward, or a ranking; those
belong to a later stage (spec section 89).

Candidate code and generated tests execute only inside the Stage 5 Docker sandbox
(``python_dpo.sandbox``) — this package never calls ``exec``, ``eval``, or a shell on
candidate-derived text, and never runs generated Python on the host (spec sections 78-80).
"""

from .config import (
    DEFAULT_EVALUATOR_IMAGE,
    DEFAULT_STARTUP_GRACE_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    EvaluationConfig,
)
from .errors import (
    CandidateNotFoundError,
    EvaluationConfigError,
    EvaluationError,
    InvalidProblemError,
    ProblemNotFoundError,
    ResultParseError,
    TestGenerationError,
)
from .executor import EVALUATOR_VERSION, CandidateEvaluator, EvaluationSummary
from .models import (
    CANDIDATE_EVALUATION_STATUSES,
    EVALUATION_FAILURE_TYPES,
    EVALUATION_RESULT_STATUSES,
    EVALUATION_RUN_STATUSES,
    MANIFEST_VERSION,
    STATISTICS_VERSION,
    TEST_CASE_STATUSES,
    EvaluationFailure,
    EvaluationManifest,
    EvaluationModelError,
    EvaluationResult,
    EvaluationStatistics,
    TestCaseResult,
)
from .probe import probe_versions
from .pytest_runner import PYTEST_COMMAND, PytestRunner, build_evaluation_sandbox_config
from .repository import (
    EVALUATIONS_FILENAME,
    FAILURES_FILENAME,
    TEST_RESULTS_FILENAME,
    EvaluationRepository,
    EvaluationStoreError,
)
from .run_repository import (
    MANIFEST_FILENAME,
    STATISTICS_FILENAME,
    EvaluationRunError,
    EvaluationRunNotFoundError,
    EvaluationRunRepository,
)
from .result_parser import (
    CONFTEST_FILENAME,
    ParsedPytestRun,
    PytestResultParser,
    RawTestEvent,
    new_nonce,
    reconcile,
    render_conftest,
)
from .test_generator import TEST_FILENAME, TEST_GENERATOR_VERSION, TestGenerator, TestJob

__all__ = [
    "CANDIDATE_EVALUATION_STATUSES",
    "CONFTEST_FILENAME",
    "DEFAULT_EVALUATOR_IMAGE",
    "DEFAULT_STARTUP_GRACE_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "EVALUATIONS_FILENAME",
    "EVALUATION_FAILURE_TYPES",
    "EVALUATION_RESULT_STATUSES",
    "EVALUATION_RUN_STATUSES",
    "EVALUATOR_VERSION",
    "FAILURES_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "PYTEST_COMMAND",
    "STATISTICS_FILENAME",
    "STATISTICS_VERSION",
    "TEST_CASE_STATUSES",
    "TEST_FILENAME",
    "TEST_GENERATOR_VERSION",
    "TEST_RESULTS_FILENAME",
    "CandidateEvaluator",
    "CandidateNotFoundError",
    "EvaluationConfig",
    "EvaluationConfigError",
    "EvaluationError",
    "EvaluationFailure",
    "EvaluationManifest",
    "EvaluationModelError",
    "EvaluationRepository",
    "EvaluationResult",
    "EvaluationRunError",
    "EvaluationRunNotFoundError",
    "EvaluationRunRepository",
    "EvaluationStatistics",
    "EvaluationStoreError",
    "EvaluationSummary",
    "InvalidProblemError",
    "ParsedPytestRun",
    "ProblemNotFoundError",
    "PytestResultParser",
    "PytestRunner",
    "RawTestEvent",
    "ResultParseError",
    "TestCaseResult",
    "TestGenerationError",
    "TestGenerator",
    "TestJob",
    "build_evaluation_sandbox_config",
    "new_nonce",
    "probe_versions",
    "reconcile",
    "render_conftest",
]
