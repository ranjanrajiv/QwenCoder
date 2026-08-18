"""Candidate generation: strategies, prompts, extraction, and static validation.

The generator's responsibility is to *generate and persist* candidate programs. It never
decides whether one is correct — that belongs to the evaluator in a later stage.
"""

from .code_extractor import NO_CODE_DETECTED, ExtractionResult, extract_code
from .generator import CandidateGenerator, GenerationSummary
from .prompt_builder import (
    CANONICAL_PROMPT_VERSION,
    PROMPT_VERSION,
    build_canonical_prompt,
    build_prompt,
)
from .strategies import (
    STRATEGIES,
    STRATEGY_INSTRUCTIONS,
    StrategyError,
    instruction_for,
    resolve_strategies,
)
from .validation import SyntaxCheck, check_function_name, check_syntax

__all__ = [
    "CANONICAL_PROMPT_VERSION",
    "NO_CODE_DETECTED",
    "PROMPT_VERSION",
    "STRATEGIES",
    "STRATEGY_INSTRUCTIONS",
    "CandidateGenerator",
    "ExtractionResult",
    "GenerationSummary",
    "StrategyError",
    "SyntaxCheck",
    "build_canonical_prompt",
    "build_prompt",
    "check_function_name",
    "check_syntax",
    "extract_code",
    "instruction_for",
    "resolve_strategies",
]
