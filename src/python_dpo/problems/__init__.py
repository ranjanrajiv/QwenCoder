"""Problem dataset: schema, curated catalog, persistence, and validation.

This is the ground-truth layer of the pipeline. Every later stage joins on the
problem ids produced here.
"""

from .catalog import build_catalog
from .executor import InProcessReferenceExecutor, ReferenceExecutor
from .models import (
    CATEGORIES,
    DATASET_VERSION,
    DIFFICULTIES,
    EXPECTED_PROBLEM_COUNT,
    MIN_TESTS_PER_PROBLEM,
    Problem,
    ProblemError,
    TestCase,
    TestResult,
)
from .storage import (
    PROBLEMS_FILENAME,
    DatasetError,
    dataset_path,
    load_problems,
    save_problems,
)
from .validation import (
    DatasetReport,
    ProblemReport,
    format_report,
    validate_dataset,
    validate_problem,
)

__all__ = [
    "CATEGORIES",
    "DATASET_VERSION",
    "DIFFICULTIES",
    "EXPECTED_PROBLEM_COUNT",
    "MIN_TESTS_PER_PROBLEM",
    "PROBLEMS_FILENAME",
    "DatasetError",
    "DatasetReport",
    "InProcessReferenceExecutor",
    "Problem",
    "ProblemError",
    "ProblemReport",
    "ReferenceExecutor",
    "TestCase",
    "TestResult",
    "build_catalog",
    "dataset_path",
    "format_report",
    "load_problems",
    "save_problems",
    "validate_dataset",
    "validate_problem",
]
