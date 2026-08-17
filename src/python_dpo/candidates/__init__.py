"""Generated candidates: schema and append-only JSONL persistence.

This package answers "what code did the model generate?" and nothing else. Whether a
candidate is *correct* is determined in a later stage by executing it against the
problem's tests — never here.
"""

from .models import (
    ERROR_TYPES,
    EXTRACTION_FORMATS,
    Candidate,
    CandidateError,
    GenerationFailure,
    build_candidate_id,
    utc_now_iso,
)
from .repository import (
    CANDIDATES_FILENAME,
    FAILURES_FILENAME,
    CandidateRepository,
    CandidateStoreError,
)

__all__ = [
    "CANDIDATES_FILENAME",
    "ERROR_TYPES",
    "EXTRACTION_FORMATS",
    "FAILURES_FILENAME",
    "Candidate",
    "CandidateError",
    "CandidateRepository",
    "CandidateStoreError",
    "GenerationFailure",
    "build_candidate_id",
    "utc_now_iso",
]
