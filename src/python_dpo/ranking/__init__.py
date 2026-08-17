"""Candidate evaluation, scoring, and ranking: turning Stage 6's objective execution
evidence into an objective ordering.

This package answers *"was this candidate good?"* and *"how does it compare to its
siblings?"* — never *"which pair should DPO train on?"* (spec 07 sections 2, 81). It
produces no ``chosen``/``rejected`` labels, no DPO JSONL, and calls no LLM judge; the
ranking signal is test-case performance alone (spec sections 2, 56).

Candidates are ranked strictly **within their own problem** (spec sections 7, 8) and,
where the evidence ties two candidates, they stay tied (spec sections 29, 35, 38) —
nothing here invents a preference. Given identical evaluation evidence and configuration,
the output is fully deterministic (spec section 46): no randomness, no LLM calls, no
timestamp-influenced ordering.
"""

from .classifier import CLASSIFIER_VERSION, CorrectnessClassifier
from .comparator import COMPARATOR_VERSION, CandidateComparator
from .errors import (
    EvaluationRunNotFoundError,
    RankingConfigError,
    RankingError,
)
from .models import (
    COMPARISON_RELATIONS,
    CORRECTNESS_VALUES,
    MANIFEST_VERSION,
    RANKING_RUN_STATUSES,
    RANKING_RUN_STATUS_TRANSITIONS,
    STATISTICS_VERSION,
    CandidateAssessment,
    ComparisonResult,
    RankingManifest,
    RankingModelError,
    RankingResult,
    RankingStatistics,
    compute_pass_rate,
)
from .ranker import RANKING_VERSION, CandidateRanker
from .repository import (
    ASSESSMENTS_FILENAME,
    COMPARISONS_FILENAME,
    RANKINGS_FILENAME,
    RankingRepository,
    RankingStoreError,
)
from .run_repository import (
    MANIFEST_FILENAME,
    STATISTICS_FILENAME,
    RankingRunError,
    RankingRunNotFoundError,
    RankingRunRepository,
)
from .scorer import SCORING_VERSION, CandidateScorer
from .statistics import format_ranking_statistics, format_ranking_table
from .validation import (
    RankingValidationIssue,
    RankingValidationReport,
    format_ranking_report,
    validate_ranking_run,
)

__all__ = [
    "ASSESSMENTS_FILENAME",
    "CLASSIFIER_VERSION",
    "COMPARATOR_VERSION",
    "COMPARISONS_FILENAME",
    "COMPARISON_RELATIONS",
    "CORRECTNESS_VALUES",
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "RANKINGS_FILENAME",
    "RANKING_RUN_STATUSES",
    "RANKING_RUN_STATUS_TRANSITIONS",
    "RANKING_VERSION",
    "SCORING_VERSION",
    "STATISTICS_FILENAME",
    "STATISTICS_VERSION",
    "CandidateAssessment",
    "CandidateComparator",
    "CandidateRanker",
    "CandidateScorer",
    "ComparisonResult",
    "CorrectnessClassifier",
    "EvaluationRunNotFoundError",
    "RankingConfigError",
    "RankingError",
    "RankingManifest",
    "RankingModelError",
    "RankingRepository",
    "RankingResult",
    "RankingRunError",
    "RankingRunNotFoundError",
    "RankingRunRepository",
    "RankingStatistics",
    "RankingStoreError",
    "RankingValidationIssue",
    "RankingValidationReport",
    "compute_pass_rate",
    "format_ranking_report",
    "format_ranking_statistics",
    "format_ranking_table",
    "validate_ranking_run",
]
