"""DPO preference pair generation: turning Stage 7's objective ranking into
``{prompt, chosen, rejected}`` training data.

This is the first package allowed to say ``chosen``/``rejected`` (Stage 7's spec reserved
that vocabulary for here, spec 07 section 67). It converts an objective ordering into a
labeled preference — never the other way around: no LLM judge, no model call of any kind
(spec section 87), and the candidate code itself is never rewritten, repaired, or
reformatted (spec sections 12-14, 97, 99).

A preference pair exists only when the evidence is decisive: ties never produce a pair
(spec section 17), indeterminate candidates never participate (spec section 38), and
identical code never produces a pair against itself (spec sections 33, 34). Three
selection policies — ``strict``, ``margin``, ``all_better`` — trade off confidence against
pair count (spec sections 24-28); the same ranking run can produce several preference
datasets from these without rerunning Qwen, Docker, or pytest (spec section 48).
"""

from .builder import BUILDER_VERSION, BuildResult, PreferencePairBuilder, ProblemBuildResult
from .config import DEFAULT_POLICY, PreferenceConfig, SplitSettings
from .dedup import code_identical, dedupe_training_records, pair_key, training_key
from .errors import (
    PreferenceConfigError,
    PreferenceError,
    PreferencePolicyError,
    PreferenceRunNotFoundError,
    PreferenceStoreError,
    RankingRunNotFoundError,
)
from .models import (
    DATASET_SCHEMA_VERSION,
    MANIFEST_VERSION,
    PAIR_CORRECTNESS_VALUES,
    POLICIES,
    PREFERENCE_RUN_STATUSES,
    PREFERENCE_RUN_STATUS_TRANSITIONS,
    PREFERENCE_STRENGTHS,
    PREFERENCE_VERSION,
    QUALITY_REPORT_VERSION,
    REJECTION_REASONS,
    SPLITS,
    STATISTICS_VERSION,
    PreferenceManifest,
    PreferenceModelError,
    PreferencePair,
    PreferenceRejection,
    PreferenceStatistics,
    QualityReport,
    build_quality_report,
    compute_pass_rate,
    derive_candidates_considered,
)
from .policies import (
    DEFAULT_MINIMUM_SCORE_MARGIN,
    AllBetterPolicy,
    MarginPolicy,
    PreferencePolicy,
    StrictPolicy,
    make_policy,
)
from .prompt import PromptLineageError, verify_prompt_lineage
from .repository import (
    METADATA_FILENAME,
    PREFERENCES_FILENAME,
    QUALITY_REPORT_FILENAME,
    REJECTIONS_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    PreferenceRepository,
)
from .run_repository import (
    MANIFEST_FILENAME as RUN_MANIFEST_FILENAME,
    STATISTICS_FILENAME as RUN_STATISTICS_FILENAME,
    PreferenceRunError,
    PreferenceRunRepository,
)
from .splitter import (
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SPLIT_SEED,
    SPLITTER_VERSION,
    ProblemSplitter,
    SplitConfigError,
    SplitManifest,
)
from .statistics import (
    format_pair_detail,
    format_pair_table,
    format_preference_statistics,
    format_quality_report,
)
from .validation import (
    PreferenceValidationIssue,
    PreferenceValidationReport,
    format_preference_report,
    validate_preference_run,
)

__all__ = [
    "BUILDER_VERSION",
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_MINIMUM_SCORE_MARGIN",
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_SPLIT_SEED",
    "MANIFEST_VERSION",
    "METADATA_FILENAME",
    "PAIR_CORRECTNESS_VALUES",
    "POLICIES",
    "PREFERENCES_FILENAME",
    "PREFERENCE_RUN_STATUSES",
    "PREFERENCE_RUN_STATUS_TRANSITIONS",
    "PREFERENCE_STRENGTHS",
    "PREFERENCE_VERSION",
    "QUALITY_REPORT_FILENAME",
    "QUALITY_REPORT_VERSION",
    "REJECTIONS_FILENAME",
    "REJECTION_REASONS",
    "RUN_MANIFEST_FILENAME",
    "RUN_STATISTICS_FILENAME",
    "SPLITTER_VERSION",
    "SPLITS",
    "SPLIT_MANIFEST_FILENAME",
    "STATISTICS_VERSION",
    "TEST_FILENAME",
    "TRAIN_FILENAME",
    "VALIDATION_FILENAME",
    "AllBetterPolicy",
    "BuildResult",
    "DEFAULT_POLICY",
    "MarginPolicy",
    "PreferenceConfig",
    "PreferenceConfigError",
    "PreferenceError",
    "PreferenceManifest",
    "PreferenceModelError",
    "PreferencePair",
    "PreferencePairBuilder",
    "PreferencePolicy",
    "PreferencePolicyError",
    "PreferenceRejection",
    "PreferenceRepository",
    "PreferenceRunError",
    "PreferenceRunNotFoundError",
    "PreferenceRunRepository",
    "PreferenceStatistics",
    "PreferenceStoreError",
    "PreferenceValidationIssue",
    "PreferenceValidationReport",
    "ProblemBuildResult",
    "ProblemSplitter",
    "PromptLineageError",
    "QualityReport",
    "RankingRunNotFoundError",
    "SplitConfigError",
    "SplitManifest",
    "SplitSettings",
    "StrictPolicy",
    "build_quality_report",
    "code_identical",
    "compute_pass_rate",
    "dedupe_training_records",
    "derive_candidates_considered",
    "format_pair_detail",
    "format_pair_table",
    "format_preference_report",
    "format_preference_statistics",
    "format_quality_report",
    "make_policy",
    "pair_key",
    "training_key",
    "validate_preference_run",
    "verify_prompt_lineage",
]
