"""Error analysis, preference refinement and iteration proposals (spec 11).

Stage 10 answers *"did DPO make the model better?"* with a number. This package asks the
question that number cannot: **what should change in the next iteration?**

Four boundaries define it, all of them enforced in code rather than by convention:

* **Classification is deterministic, never an LLM** (sections 12, 295). pytest status, the
  raw exception class, the exit code and the timeout flag are the evidence.
* **Correlation is never stated as causation** (section 38). Reports say *potential data
  gap*; :data:`~python_dpo.analysis.report.FORBIDDEN_CAUSAL_PHRASES` is asserted absent in
  tests.
* **The benchmark is never contaminated** (sections 65, 66, 104, 117). A refined dataset
  that would carry a held-out problem raises
  :class:`~python_dpo.analysis.errors.RefinementLeakageError` before anything is written.
* **Nothing is retrained** (sections 5, 113). The stage emits ``next_experiment.yaml`` and
  stops.

This is the one stage in the back half of the pipeline that is pure computation over
persisted artifacts -- no model, no GPU, no Docker -- so all of it runs in the default
offline test suite.
"""

from __future__ import annotations

from .config import AnalysisConfig
from .driver import run_analysis
from .errors import (
    AnalysisConfigError,
    AnalysisError,
    AnalysisInputError,
    AnalysisRunError,
    AnalysisRunNotFoundError,
    AnalysisStoreError,
    LineageError,
    RefinementLeakageError,
)
from .ingest import AnalysisInputs, load_analysis_inputs, resolve_lineage
from .models import (
    AnalysisManifest,
    CategoryGap,
    DiversityReport,
    ErrorProfile,
    ErrorRateComparison,
    ExperimentLineage,
    ProblemOutcome,
    Recommendation,
    TestFailureStat,
)
from .refinement import assert_no_benchmark_leakage
from .run_repository import AnalysisRunRepository
from .taxonomy import ERROR_CATEGORIES, ErrorClassification, classify

__all__ = [
    "ERROR_CATEGORIES",
    "AnalysisConfig",
    "AnalysisConfigError",
    "AnalysisError",
    "AnalysisInputError",
    "AnalysisInputs",
    "AnalysisManifest",
    "AnalysisRunError",
    "AnalysisRunNotFoundError",
    "AnalysisRunRepository",
    "AnalysisStoreError",
    "CategoryGap",
    "DiversityReport",
    "ErrorClassification",
    "ErrorProfile",
    "ErrorRateComparison",
    "ExperimentLineage",
    "LineageError",
    "ProblemOutcome",
    "Recommendation",
    "RefinementLeakageError",
    "TestFailureStat",
    "assert_no_benchmark_leakage",
    "classify",
    "load_analysis_inputs",
    "resolve_lineage",
    "run_analysis",
]
