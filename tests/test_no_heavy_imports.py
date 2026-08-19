"""Guard: importing the package must not load a model backend (spec 03 section 7).

``import python_dpo`` should cost milliseconds, not several GB of weights. The rule is
easy to break by accident — a single top-level ``import torch`` in the Qwen client would
do it — so it is asserted rather than assumed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HEAVY_MODULES = (
    "torch",
    "transformers",
    "accelerate",
    "trl",
    "peft",
    "bitsandbytes",
    "datasets",
)

_PROBE = """
import sys

import python_dpo
import python_dpo.analysis
import python_dpo.analysis.classification
import python_dpo.analysis.config
import python_dpo.analysis.coverage
import python_dpo.analysis.diversity
import python_dpo.analysis.driver
import python_dpo.analysis.errors
import python_dpo.analysis.failures
import python_dpo.analysis.ingest
import python_dpo.analysis.models
import python_dpo.analysis.outcomes
import python_dpo.analysis.recommend
import python_dpo.analysis.refinement
import python_dpo.analysis.report
import python_dpo.analysis.run_repository
import python_dpo.analysis.taxonomy
import python_dpo.analysis.training_curve
import python_dpo.candidates
import python_dpo.cli
import python_dpo.config
import python_dpo.generation
import python_dpo.model_evaluation
import python_dpo.model_evaluation.benchmark
import python_dpo.model_evaluation.cache
import python_dpo.model_evaluation.comparison
import python_dpo.model_evaluation.config
import python_dpo.model_evaluation.evaluation
import python_dpo.model_evaluation.generation
import python_dpo.model_evaluation.metrics
import python_dpo.model_evaluation.models
import python_dpo.model_evaluation.report
import python_dpo.model_evaluation.run_repository
import python_dpo.model_evaluation.runners
import python_dpo.model_evaluation.statistics
import python_dpo.models
import python_dpo.models.qwen
import python_dpo.packaging
import python_dpo.packaging.compare
import python_dpo.packaging.errors
import python_dpo.packaging.inference
import python_dpo.packaging.merge
import python_dpo.packaging.package
import python_dpo.packaging.pipeline_stage
import python_dpo.packaging.registry
import python_dpo.packaging.verify
import python_dpo.pipeline
import python_dpo.pipeline.archive
import python_dpo.pipeline.artifacts
import python_dpo.pipeline.cache
import python_dpo.pipeline.config
import python_dpo.pipeline.cost
import python_dpo.pipeline.environment
import python_dpo.pipeline.errors
import python_dpo.pipeline.gitinfo
import python_dpo.pipeline.hashing
import python_dpo.pipeline.manifest
import python_dpo.pipeline.report
import python_dpo.pipeline.repository
import python_dpo.pipeline.reproduce
import python_dpo.pipeline.resources
import python_dpo.pipeline.stages
import python_dpo.pipeline.stages._context
import python_dpo.pipeline.stages.candidate_evaluation
import python_dpo.pipeline.stages.candidate_execution
import python_dpo.pipeline.stages.candidate_generation
import python_dpo.pipeline.stages.dpo_training
import python_dpo.pipeline.stages.error_analysis
import python_dpo.pipeline.stages.model_evaluation
import python_dpo.pipeline.stages.packaging
import python_dpo.pipeline.stages.preference_generation
import python_dpo.pipeline.stages.problem_dataset
import python_dpo.pipeline.state
import python_dpo.training
import python_dpo.training.callbacks
import python_dpo.training.config
import python_dpo.training.dataset
import python_dpo.training.hardware
import python_dpo.training.lengths
import python_dpo.training.loader
import python_dpo.training.run_repository
import python_dpo.training.trainer
import python_dpo.training.verify
import python_dpo.training.versions

heavy = [name for name in {heavy!r} if name in sys.modules]
print(",".join(heavy))
"""


def test_importing_the_package_does_not_load_a_model_backend():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(heavy=HEAVY_MODULES)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    loaded = result.stdout.strip()
    assert loaded == "", f"these were imported eagerly: {loaded}"
