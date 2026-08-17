"""Generation runs: manifests, statistics, and the repository that owns them.

A run is one self-contained, independently auditable directory under
``data/candidates/runs/`` (spec 04 sections 3, 6). This package answers "what was this
run asked to do, what state is it in, and what did it actually produce?" — never whether
a candidate is correct, which stays out of scope until the Docker sandbox stage.
"""

from .environment import capture_environment
from .models import (
    MANIFEST_VERSION,
    RUN_SOURCES,
    RUN_STATUSES,
    RUN_STATUS_TRANSITIONS,
    STATISTICS_VERSION,
    RunError,
    RunFailure,
    RunManifest,
    RunStatistics,
)
from .repository import MANIFEST_FILENAME, STATISTICS_FILENAME, RunNotFoundError, RunRepository
from .migration import MigrationError, migrate_flat_file
from .validation import RunValidationIssue, RunValidationReport, format_run_report, validate_run

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "RUN_SOURCES",
    "RUN_STATUSES",
    "RUN_STATUS_TRANSITIONS",
    "STATISTICS_FILENAME",
    "STATISTICS_VERSION",
    "MigrationError",
    "RunError",
    "RunFailure",
    "RunManifest",
    "RunNotFoundError",
    "RunRepository",
    "RunStatistics",
    "RunValidationIssue",
    "RunValidationReport",
    "capture_environment",
    "format_run_report",
    "migrate_flat_file",
    "validate_run",
]
