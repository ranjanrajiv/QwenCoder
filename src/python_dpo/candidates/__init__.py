"""Generated candidates: schema and durable, run-scoped JSONL persistence.

This package answers "what code did the model generate, in which run?" and nothing
else. Whether a candidate is *correct* is determined in a later stage by executing it
against the problem's tests — never here.
"""

from .hashing import sha256_text
from .models import (
    CANDIDATE_ERROR_TYPES,
    CANDIDATE_SCHEMA_VERSION,
    CANDIDATE_SCHEMA_VERSIONS,
    ERROR_TYPES,
    EXTRACTION_FORMATS,
    INFRASTRUCTURE_ERROR_TYPES,
    LEGACY_CANDIDATE_SCHEMA_VERSION,
    Candidate,
    CandidateError,
    GenerationFailure,
    build_candidate_id,
    utc_now_iso,
)
from .repository import (
    CANDIDATES_FILENAME,
    FAILURES_FILENAME,
    LEGACY_FAILURES_FILENAME,
    PROMPTS_DIRNAME,
    PROMPTS_FILENAME,
    CandidateRepository,
    CandidateStoreError,
    PromptRecord,
)

__all__ = [
    "CANDIDATES_FILENAME",
    "CANDIDATE_ERROR_TYPES",
    "CANDIDATE_SCHEMA_VERSION",
    "CANDIDATE_SCHEMA_VERSIONS",
    "ERROR_TYPES",
    "EXTRACTION_FORMATS",
    "FAILURES_FILENAME",
    "INFRASTRUCTURE_ERROR_TYPES",
    "LEGACY_CANDIDATE_SCHEMA_VERSION",
    "LEGACY_FAILURES_FILENAME",
    "PROMPTS_DIRNAME",
    "PROMPTS_FILENAME",
    "Candidate",
    "CandidateError",
    "CandidateRepository",
    "CandidateStoreError",
    "GenerationFailure",
    "PromptRecord",
    "build_candidate_id",
    "sha256_text",
    "utc_now_iso",
]
