# Python DPO Data Generation Pipeline

## Step 12 — End-to-End Pipeline Orchestration, Model Packaging, Deployment and Productionization

**Specification Version:** 1.0
**Status:** Implementation Specification
**Step:** 12 of 12
**Depends On:** Steps 1–11

---

# 1. Objective

Transform the Python DPO training system from a collection of individual stages into a reproducible, versioned, end-to-end machine-learning pipeline.

The final system must allow the user to execute:

```text
Problem Generation
        ↓
Candidate Generation
        ↓
Candidate Execution
        ↓
Candidate Evaluation
        ↓
Preference Generation
        ↓
DPO/QLoRA Training
        ↓
Held-Out Evaluation
        ↓
Error Analysis
        ↓
Next Experiment
```

through a consistent CLI and configuration system.

The system must also support packaging the resulting Qwen + LoRA model for inference.

---

# 2. Primary Goal

The final system must make it possible to execute an experiment such as:

```bash
python -m python_dpo experiment run \
    --config configs/experiments/qwen_python_dpo_v1.yaml
```

and produce a complete experiment artifact containing:

* source problem dataset
* generated candidates
* execution results
* preference dataset
* training configuration
* trained adapter
* evaluation results
* statistical analysis
* error analysis
* experiment recommendation
* final model artifact
* complete lineage

---

# 3. Design Principle

The pipeline must be:

```
reproducible
configurable
observable
resumable
versioned
testable
auditable
```

The system must NOT rely on:

```
manually executed undocumented commands
manually edited intermediate files
hidden environment variables
implicit dataset versions
```

---

# 4. End-to-End Architecture

The final architecture should be:

```text
                    ┌───────────────────┐
                    │ Experiment Config │
                    └─────────┬─────────┘
                              │
                              ▼
                     ┌────────────────┐
                     │ Problem Dataset│
                     └───────┬────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Candidate Generation │
                  └──────────┬───────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │ Docker/pytest│
                     └──────┬───────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Preference Builder │
                  └──────────┬─────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ DPO / QLoRA    │
                    └───────┬────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Held-out Evaluation│
                  └──────────┬─────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Error Analysis │
                    └───────┬────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Experiment Report  │
                  └────────────────────┘
```

---

# 5. Pipeline Stages

The orchestrator must understand these stages:

```text
1. problem_generation
2. candidate_generation
3. candidate_execution
4. candidate_evaluation
5. preference_generation
6. dpo_training
7. model_evaluation
8. error_analysis
9. packaging
```

The exact internal names may differ, but the stages must remain individually addressable.

---

# 6. Stage Independence

Each stage must be executable independently.

Example:

```bash
python -m python_dpo pipeline run-stage \
    --stage candidate-generation
```

The stage must not depend on the orchestrator being used.

This allows:

* debugging
* retries
* experimentation
* local development

---

# 7. Pipeline Orchestrator

Implement:

```text
PipelineOrchestrator
```

Responsibilities:

* resolve configuration
* validate dependencies
* determine execution order
* create run IDs
* execute stages
* persist stage status
* handle failures
* resume incomplete runs
* produce final manifest

---

# 8. Pipeline Configuration

Create:

```text
configs/experiments/qwen_python_dpo_v1.yaml
```

Example:

```yaml
experiment:
  name: qwen-python-dpo-v1
  seed: 42

problem_generation:
  enabled: true
  problem_count: 1000

candidate_generation:
  enabled: true
  candidates_per_problem: 8

candidate_execution:
  enabled: true

candidate_evaluation:
  enabled: true

preference_generation:
  enabled: true
  policy: strict

dpo_training:
  enabled: true
  config: configs/training/dpo_qlora.yaml

model_evaluation:
  enabled: true
  benchmark: python_eval_v1
  num_samples: 10

error_analysis:
  enabled: true

packaging:
  enabled: true
```

---

# 9. Configuration Hierarchy

Support:

```text
global configuration
        ↓
experiment configuration
        ↓
stage configuration
        ↓
CLI overrides
```

CLI overrides must have highest priority.

---

# 10. Configuration Immutability

Once an experiment starts, create an immutable copy:

```text
runs/<experiment_run_id>/resolved_config.yaml
```

The experiment must use this resolved configuration even if the source configuration is later modified.

---

# 11. Experiment Run ID

Every complete experiment receives:

```text
experiment_run_id
```

Example:

```text
exp_20260818_234500_a92f
```

---

# 12. Experiment Directory

Use:

```text
data/experiments/runs/<experiment_run_id>/
```

Structure:

```text
<experiment_run_id>/
    manifest.json
    resolved_config.yaml
    environment.json
    stages/
    datasets/
    candidates/
    preferences/
    training/
    evaluations/
    analysis/
    model/
    reports/
    logs/
```

---

# 13. Stage State

Every stage must have a state:

```text
PENDING
RUNNING
COMPLETED
FAILED
SKIPPED
CANCELLED
```

---

# 14. Stage Manifest

Each stage must create:

```text
stage_manifest.json
```

containing:

```text
stage_name
stage_run_id
status
start_time
end_time
input_artifacts
output_artifacts
configuration_hash
code_version
error
```

---

# 15. Stage Dependencies

The orchestrator must understand:

```text
candidate_generation
    requires problem_generation

candidate_execution
    requires candidate_generation

candidate_evaluation
    requires candidate_execution

preference_generation
    requires candidate_evaluation

dpo_training
    requires preference_generation

model_evaluation
    requires dpo_training

error_analysis
    requires model_evaluation

packaging
    requires dpo_training
```

---

# 16. Dependency Validation

If a required artifact does not exist:

```text
FAIL
```

with a clear error.

Do not attempt to reconstruct missing artifacts silently.

---

# 17. Stage Reuse

If a stage has already completed successfully and its input hashes match:

```text
reuse existing stage
```

instead of executing it again.

---

# 18. Stage Cache

The cache key must include:

```text
stage
input artifact hashes
configuration hash
code version
model version
```

---

# 19. Cache Invalidation

Invalidate a stage if any relevant input changes.

Example:

```text
preference dataset changes
        ↓
DPO training cache invalidated
        ↓
model evaluation invalidated
        ↓
error analysis invalidated
```

But:

```text
DPO hyperparameter changes
```

must NOT invalidate:

```text
problem generation
candidate generation
```

unless those stages explicitly depend on the changed parameter.

---

# 20. Pipeline Resume

Support:

```bash
python -m python_dpo experiment resume \
    --experiment-run-id EXP_RUN_ID
```

The orchestrator should determine:

* completed stages
* failed stages
* missing stages
* stale stages

and resume from the earliest invalid stage.

---

# 21. Stage Retry

Support:

```bash
python -m python_dpo pipeline retry \
    --experiment-run-id EXP_RUN_ID \
    --stage dpo-training
```

Only failed or invalid stages may be retried.

---

# 22. Force Re-run

Support:

```bash
--force
```

This explicitly invalidates the requested stage and all dependent stages.

Example:

```text
force candidate-evaluation

        ↓

candidate-evaluation rerun
preference-generation rerun
dpo-training rerun
model-evaluation rerun
error-analysis rerun
```

---

# 23. Dry Run

Support:

```bash
python -m python_dpo experiment run \
    --config configs/experiments/qwen_python_dpo_v1.yaml \
    --dry-run
```

Output:

```text
Experiment:
    qwen-python-dpo-v1

Stages:

    [1] Problem Generation
    [2] Candidate Generation
    [3] Candidate Execution
    [4] Candidate Evaluation
    [5] Preference Generation
    [6] DPO Training
    [7] Model Evaluation
    [8] Error Analysis
    [9] Packaging
```

No stage should execute.

---

# 24. Smoke Experiment

Support:

```bash
python -m python_dpo experiment run \
    --config configs/experiments/qwen_python_dpo_v1.yaml \
    --smoke-test
```

The smoke test must reduce:

* problem count
* candidates/problem
* training steps
* evaluation problems

while preserving the complete pipeline.

---

# 25. Smoke Experiment Objective

The smoke test is not intended to measure model quality.

It verifies:

```text
problem → candidate → pytest → preference
→ DPO → evaluation → analysis → package
```

works end-to-end.

---

# 26. Production Experiment

A full experiment should only run after:

```text
all stage tests pass
AND
smoke test passes
```

---

# 27. Artifact Lineage

Every artifact must be traceable.

Example:

```text
Model Adapter
     │
     └── training_run_id
             │
             └── preference_run_id
                     │
                     └── evaluation_run_id
                             │
                             └── problem_dataset_version
```

---

# 28. Global Experiment Manifest

Create:

```text
manifest.json
```

containing:

```text
experiment_run_id
experiment_name
status
start_time
end_time
git_commit
configuration_hash
dataset_versions
model_versions
stage_runs
final_model
final_evaluation
recommendation
```

---

# 29. Git Version

Capture:

```text
git commit SHA
git branch
git dirty status
```

If the working tree is dirty:

```text
warning
```

or:

```text
fail
```

depending on configuration.

---

# 30. Environment Capture

Capture:

```text
Python version
OS
CUDA
NVIDIA driver
GPU
VRAM
PyTorch
Transformers
TRL
PEFT
bitsandbytes
Accelerate
Datasets
pytest
Docker
```

---

# 31. Dependency Lock

Create a reproducible dependency specification.

Preferred:

```text
requirements.lock
```

or an equivalent environment lock.

Do not depend on unconstrained:

```text
pip install package
```

for reproducibility.

---

# 32. Containerization

Provide a Docker image for the training/evaluation environment.

Example:

```text
docker/Dockerfile
```

The image should contain:

* Python
* PyTorch
* Transformers
* TRL
* PEFT
* bitsandbytes
* Accelerate
* Datasets
* pytest
* project package

---

# 33. Docker Separation

The system should maintain two conceptual environments:

```text
Training/Inference Container
        +
Untrusted Python Execution Sandbox
```

The candidate execution environment must remain isolated from the model-training environment.

---

# 34. Candidate Sandbox Security

Generated Python code must continue to execute only inside the Step 5 sandbox.

Step 12 must not weaken sandbox restrictions.

---

# 35. Model Artifact

The final model artifact should contain:

```text
base model identifier
base model revision
LoRA adapter
adapter configuration
tokenizer configuration
training metadata
```

---

# 36. Adapter Packaging

Create:

```text
model/
    adapter/
    tokenizer/
    manifest.json
```

The base model itself should not necessarily be duplicated.

---

# 37. Model Package Manifest

Example:

```json
{
  "model_name": "Qwen/...",
  "model_revision": "...",
  "adapter_type": "LoRA",
  "training_method": "DPO",
  "quantization": "4-bit NF4",
  "training_run_id": "...",
  "evaluation_run_id": "...",
  "benchmark": "python_eval_v1"
}
```

---

# 38. Model Verification

Before packaging:

1. Load base model.
2. Load adapter.
3. Load tokenizer.
4. Generate Python code.
5. Run through sandbox.
6. Verify successful execution.

Packaging must fail if this verification fails.

---

# 39. Inference CLI

Add:

```bash
python -m python_dpo model generate \
    --model-package MODEL_PACKAGE \
    --prompt "Write a Python function..."
```

---

# 40. Batch Inference

Support:

```bash
python -m python_dpo model generate-batch \
    --model-package MODEL_PACKAGE \
    --input problems.jsonl \
    --output predictions.jsonl
```

---

# 41. Model Evaluation CLI

Allow a packaged model to be evaluated independently:

```bash
python -m python_dpo model evaluate \
    --model-package MODEL_PACKAGE \
    --benchmark python_eval_v1
```

---

# 42. Model Package Independence

The packaged model must be usable without the full DPO training pipeline.

Inference should require only:

```text
model package
+
inference environment
```

---

# 43. Optional LoRA Merge

Support:

```bash
python -m python_dpo model merge \
    --model-package MODEL_PACKAGE
```

The merge operation must produce:

```text
merged_model/
```

and must not delete:

```text
adapter/
```

---

# 44. Quantized Merge Warning

If the base model is loaded in 4-bit mode, merging adapters may require an appropriate higher-precision loading path.

The implementation must not assume that:

```text
4-bit model
+
LoRA
=
directly mergeable artifact
```

If the selected stack does not support the operation safely:

```text
FAIL with explanation
```

---

# 45. Model Registry

Implement a lightweight local model registry.

Example:

```text
models/registry.json
```

Each model entry should contain:

```text
model_id
training_run_id
adapter_path
benchmark
pass_at_1
pass_at_5
pass_at_10
status
created_at
```

---

# 46. Model Status

Supported states:

```text
EXPERIMENTAL
VALIDATED
RECOMMENDED
RETIRED
REJECTED
```

---

# 47. Model Promotion

A model can become:

```text
RECOMMENDED
```

only when Step 10 success criteria are satisfied.

Promotion must be explicit.

---

# 48. No Automatic Promotion

The pipeline must not automatically deploy or promote a model merely because training completed.

---

# 49. Model Comparison

Support:

```bash
python -m python_dpo model compare \
    --models MODEL_A,MODEL_B
```

Report:

```text
pass@1
pass@5
pass@10
syntax success
timeout rate
latency
memory
```

---

# 50. Experiment Dashboard Data

Produce machine-readable data suitable for visualization.

Create:

```text
reports/experiment_metrics.json
```

Include:

```text
training metrics
evaluation metrics
error metrics
resource metrics
```

---

# 51. Resource Monitoring

Capture:

```text
GPU utilization
GPU memory
CPU utilization
RAM
training duration
inference duration
tokens/second
```

where available.

---

# 52. Cost Tracking

The system should optionally calculate approximate experiment cost.

For local RTX 3060:

```text
GPU hours
```

may be the primary cost metric.

For cloud environments:

```text
GPU-hours
CPU-hours
storage
LLM API calls
```

may be included.

---

# 53. LLM API Cost

If Claude or another API is used for:

* problem generation
* analysis
* synthetic data generation

record:

```text
provider
model
input tokens
output tokens
estimated cost
```

---

# 54. Claude Code Compatibility

The project must be straightforward to operate using Claude Code.

All implementation tasks must be represented by:

```text
specs/
```

and:

```text
configs/
```

rather than relying on undocumented interactive instructions.

---

# 55. Claude Code Workflow

Recommended workflow:

```text
Read specification
        ↓
Inspect current implementation
        ↓
Implement stage
        ↓
Run unit tests
        ↓
Run smoke test
        ↓
Inspect generated artifacts
        ↓
Commit changes
```

---

# 56. Claude Code Guardrail

Claude Code must NOT modify:

```text
evaluation benchmark
```

without explicit instruction.

---

# 57. Claude Code Guardrail

Claude Code must NOT silently modify:

```text
training dataset
preference policy
DPO configuration
```

to make an experiment pass.

---

# 58. Experiment Specification

Every experiment should have:

```text
hypothesis
dataset
model
training configuration
evaluation benchmark
success criteria
```

---

# 59. Experiment Template

Create:

```text
configs/experiments/template.yaml
```

Example:

```yaml
experiment:
  name: ""

hypothesis:
  description: ""

data:
  problem_dataset: ""
  preference_dataset: ""

model:
  base_model: ""

training:
  config: ""

evaluation:
  benchmark: ""

success_criteria:
  pass_at_1_delta: 0.02
```

---

# 60. Experiment Validation

Before execution validate:

```text
dataset exists
model exists
benchmark exists
training config exists
evaluation config exists
GPU available
Docker available
```

---

# 61. Preflight Command

Add:

```bash
python -m python_dpo experiment preflight \
    --config configs/experiments/qwen_python_dpo_v1.yaml
```

Output:

```text
[PASS] GPU
[PASS] CUDA
[PASS] Docker
[PASS] Model
[PASS] Dataset
[PASS] Benchmark
[PASS] Training configuration
[PASS] Evaluation configuration
```

---

# 62. Pipeline Graph

Provide:

```bash
python -m python_dpo experiment graph \
    --config configs/experiments/qwen_python_dpo_v1.yaml
```

Output:

```text
Problems
   ↓
Candidates
   ↓
Execution
   ↓
Evaluation
   ↓
Preferences
   ↓
DPO
   ↓
Benchmark
   ↓
Analysis
   ↓
Package
```

---

# 63. Stage Status Command

Add:

```bash
python -m python_dpo experiment status \
    --experiment-run-id EXP_RUN_ID
```

Example:

```text
Problem Generation       COMPLETE
Candidate Generation     COMPLETE
Candidate Execution      COMPLETE
Candidate Evaluation     COMPLETE
Preference Generation    COMPLETE
DPO Training             COMPLETE
Model Evaluation         RUNNING
Error Analysis           PENDING
Packaging                PENDING
```

---

# 64. Logs

Each stage must have:

```text
logs/<stage>.log
```

The global experiment should have:

```text
logs/experiment.log
```

---

# 65. Error Handling

Errors must contain:

```text
stage
error_type
message
stack_trace
timestamp
input_artifacts
```

---

# 66. Failure Recovery

If a stage fails:

```text
stage = FAILED
```

Downstream stages must become:

```text
BLOCKED
```

or:

```text
PENDING
```

They must not execute against incomplete artifacts.

---

# 67. Interrupt Handling

If the process receives:

```text
SIGINT
SIGTERM
```

the orchestrator should:

1. record interruption
2. persist stage state
3. flush logs
4. preserve completed artifacts
5. mark current stage appropriately

---

# 68. Atomic Artifacts

Where practical, write artifacts to:

```text
temporary path
```

then atomically rename to the final location.

This prevents partially written manifests from appearing valid.

---

# 69. Data Integrity

Every major artifact should have:

```text
SHA-256 hash
```

Examples:

```text
problem dataset
candidate dataset
evaluation results
preference dataset
adapter
benchmark
final report
```

---

# 70. Artifact Manifest

Create:

```text
artifacts.json
```

Example:

```json
{
  "preference_dataset": {
    "path": "...",
    "sha256": "..."
  },
  "adapter": {
    "path": "...",
    "sha256": "..."
  }
}
```

---

# 71. Reproducibility Command

Add:

```bash
python -m python_dpo experiment reproduce \
    --experiment-run-id EXP_RUN_ID
```

This should display the commands/configuration required to recreate the experiment.

---

# 72. Reproduction Mode

Support:

```bash
--verify-only
```

which checks:

```text
model revision
dataset hash
configuration
environment
```

without running training.

---

# 73. Experiment Archive

Support exporting an experiment:

```bash
python -m python_dpo experiment archive \
    --experiment-run-id EXP_RUN_ID
```

Create:

```text
experiment_<id>.tar.gz
```

The archive should include metadata and artifacts required for reproducibility.

Do not unnecessarily duplicate the base model.

---

# 74. Experiment Import

Optionally support:

```bash
python -m python_dpo experiment inspect \
    --archive experiment_<id>.tar.gz
```

The initial implementation does not need to support automatic restoration.

---

# 75. Security

Do not allow generated Python code to execute outside the Docker sandbox.

Do not allow generated code to:

* access host filesystem
* access host Docker socket
* access cloud credentials
* access SSH keys
* access arbitrary network resources

---

# 76. Secrets

Never store:

```text
API keys
tokens
passwords
cloud credentials
```

inside:

```text
experiment manifests
logs
configs
artifacts
```

---

# 77. Environment Variables

Secrets must be supplied through:

```text
environment variables
```

or:

```text
secret manager
```

and excluded from experiment manifests.

---

# 78. PII

Generated problem datasets and experiment logs should not contain unnecessary personal information.

---

# 79. Determinism

All stochastic stages must support:

```text
seed
```

and record it.

---

# 80. Randomness Sources

Record seeds for:

```text
problem generation
candidate generation
training
evaluation
bootstrap analysis
```

---

# 81. Versioning

Version:

```text
problem dataset
candidate dataset
preference dataset
training run
model adapter
benchmark
evaluation run
analysis run
experiment
```

---

# 82. Semantic Versioning

Use versions such as:

```text
python_problem_v1
candidate_schema_v1
preference_schema_v1
benchmark_v1
```

Changes to schemas should increment versions.

---

# 83. Schema Validation

Use explicit schemas for:

```text
problem
candidate
evaluation
preference
training manifest
evaluation manifest
analysis result
experiment manifest
```

---

# 84. Backward Compatibility

If practical, readers should support previous schema versions.

If not:

```text
fail with explicit schema incompatibility
```

rather than silently interpreting old data.

---

# 85. Testing Strategy

The final project must have:

```text
unit tests
integration tests
smoke tests
pipeline tests
```

---

# 86. Unit Test Coverage

Cover:

* configuration loading
* hashing
* manifests
* stage dependency resolution
* cache keys
* state transitions
* model packaging
* artifact validation

---

# 87. Integration Test

Use a tiny dataset:

```text
3 problems
2 candidates/problem
1 DPO training step
2 evaluation problems
```

Run the entire pipeline.

Expected:

```text
SUCCESS
```

---

# 88. Failure Integration Test

Force a stage failure.

Verify:

```text
current stage = FAILED
downstream stages = BLOCKED/PENDING
```

---

# 89. Resume Test

Interrupt after:

```text
preference generation
```

Then resume.

Verify:

```text
previous stages reused
remaining stages executed
```

---

# 90. Cache Test

Run the same experiment twice.

Expected:

```text
second execution reuses compatible artifacts
```

---

# 91. Cache Invalidation Test

Change:

```text
DPO beta
```

Expected:

```text
problem generation remains cached
candidate generation remains cached
DPO training reruns
model evaluation reruns
analysis reruns
```

---

# 92. Benchmark Protection Test

Attempt to add a held-out benchmark problem to training.

Expected:

```text
FAIL
```

---

# 93. Model Packaging Test

Package a trained adapter.

Then:

```text
load package
generate code
execute code
```

Expected:

```text
SUCCESS
```

---

# 94. Final Pipeline Test

The complete test must execute:

```text
Step 1
 ↓
Step 2
 ↓
Step 3
 ↓
Step 4
 ↓
Step 5
 ↓
Step 6
 ↓
Step 7
 ↓
Step 8
 ↓
Step 9
 ↓
Step 10
 ↓
Step 11
 ↓
Step 12
```

using a small synthetic dataset.

---

# 95. Final CLI

The project should expose:

```bash
python -m python_dpo experiment preflight
python -m python_dpo experiment run
python -m python_dpo experiment resume
python -m python_dpo experiment status
python -m python_dpo experiment graph
python -m python_dpo experiment archive
python -m python_dpo experiment reproduce

python -m python_dpo model generate
python -m python_dpo model evaluate
python -m python_dpo model compare
python -m python_dpo model package
python -m python_dpo model merge
```

---

# 96. Final Project Structure

The expected final project should approximately be:

```text
python-dpo/
│
├── specs/
│   ├── 01_project_skeleton.md
│   ├── 02_problem_dataset.md
│   ├── 03_candidate_generator.md
│   ├── 04_candidate_persistence.md
│   ├── 05_docker_sandbox.md
│   ├── 06_candidate_executor.md
│   ├── 07_candidate_evaluation.md
│   ├── 08_preference_generation.md
│   ├── 09_dpo_qlora_training.md
│   ├── 10_model_evaluation.md
│   ├── 11_error_analysis_and_iteration.md
│   └── 12_pipeline_orchestration_and_productionization.md
│
├── configs/
│   ├── experiments/
│   ├── training/
│   ├── evaluation/
│   └── generation/
│
├── src/
│   └── python_dpo/
│       ├── cli/
│       ├── config/
│       ├── problems/
│       ├── candidates/
│       ├── sandbox/
│       ├── evaluation/
│       ├── preferences/
│       ├── training/
│       ├── analysis/
│       ├── pipeline/
│       ├── models/
│       ├── artifacts/
│       └── utils/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── smoke/
│   └── pipeline/
│
├── docker/
│   ├── Dockerfile
│   └── sandbox/
│
├── data/
│   ├── problems/
│   ├── candidates/
│   ├── evaluations/
│   ├── preferences/
│   ├── training/
│   ├── model_evaluations/
│   ├── analysis/
│   └── experiments/
│
├── models/
│   └── registry.json
│
├── scripts/
│
├── pyproject.toml
├── README.md
└── .gitignore
```

---

# 97. Final Experiment Lifecycle

The final system should support:

```text
CREATE EXPERIMENT
        │
        ▼
PREFLIGHT
        │
        ▼
RUN
        │
        ▼
MONITOR
        │
        ▼
EVALUATE
        │
        ▼
ANALYZE
        │
        ▼
PACKAGE
        │
        ▼
RECOMMEND NEXT EXPERIMENT
```

---

# 98. Model Lifecycle

A trained adapter moves through:

```text
TRAINED
   ↓
VERIFIED
   ↓
EVALUATED
   ↓
ANALYZED
   ↓
EXPERIMENTAL
   ↓
RECOMMENDED
   ↓
RETIRED
```

---

# 99. Final Success Criteria

Step 12 is complete only when:

* [ ] Complete pipeline can be executed from one experiment configuration.
* [ ] Every stage remains independently executable.
* [ ] Stage dependencies are enforced.
* [ ] Stage state is persisted.
* [ ] Pipeline can resume after failure.
* [ ] Compatible artifacts are cached.
* [ ] Cache invalidation works.
* [ ] Experiment configuration is immutable after start.
* [ ] Dataset versions are preserved.
* [ ] Model versions are preserved.
* [ ] Experiment lineage is preserved.
* [ ] Git version is recorded.
* [ ] Environment is recorded.
* [ ] Dependencies are reproducible.
* [ ] Training artifact can be packaged.
* [ ] Packaged adapter can be loaded.
* [ ] Packaged model can generate Python.
* [ ] Generated Python can be evaluated in the sandbox.
* [ ] Model registry exists.
* [ ] Model comparison works.
* [ ] Experiment archive works.
* [ ] Experiment reproduction metadata exists.
* [ ] Security boundaries remain intact.
* [ ] Secrets are not stored in artifacts.
* [ ] Unit tests pass.
* [ ] Integration tests pass.
* [ ] Smoke test passes.
* [ ] Full end-to-end test passes.

---

# 100. Verification Procedure

## 100.1 Preflight

Run:

```bash
python -m python_dpo experiment preflight \
    --config configs/experiments/qwen_python_dpo_v1.yaml
```

Expected:

```text
All preflight checks passed.
```

---

## 100.2 Graph

Run:

```bash
python -m python_dpo experiment graph \
    --config configs/experiments/qwen_python_dpo_v1.yaml
```

Verify:

```text
Problems
→ Candidates
→ Execution
→ Evaluation
→ Preferences
→ DPO
→ Benchmark
→ Analysis
→ Packaging
```

---

## 100.3 Smoke Experiment

Run:

```bash
python -m python_dpo experiment run \
    --config configs/experiments/qwen_python_dpo_v1.yaml \
    --smoke-test
```

Expected:

```text
Experiment completed successfully.
```

---

## 100.4 Status

Run:

```bash
python -m python_dpo experiment status \
    --experiment-run-id EXP_RUN_ID
```

Expected:

```text
All stages COMPLETE
```

---

## 100.5 Model Verification

Run:

```bash
python -m python_dpo model evaluate \
    --model-package MODEL_PACKAGE \
    --benchmark python_eval_v1
```

---

## 100.6 Archive

Run:

```bash
python -m python_dpo experiment archive \
    --experiment-run-id EXP_RUN_ID
```

Expected:

```text
Experiment archive created.
```

---

## 100.7 Reproduction Verification

Run:

```bash
python -m python_dpo experiment reproduce \
    --experiment-run-id EXP_RUN_ID \
    --verify-only
```

Expected:

```text
Experiment reproducibility checks passed.
```

---

# 101. Expected Final Artifacts

The completed experiment should produce:

```text
data/experiments/runs/<experiment_run_id>/
│
├── manifest.json
├── resolved_config.yaml
├── environment.json
│
├── stages/
│   ├── problem_generation/
│   ├── candidate_generation/
│   ├── candidate_execution/
│   ├── candidate_evaluation/
│   ├── preference_generation/
│   ├── dpo_training/
│   ├── model_evaluation/
│   ├── error_analysis/
│   └── packaging/
│
├── training/
│   └── adapter/
│
├── evaluations/
│   └── base_vs_dpo/
│
├── analysis/
│   └── final_analysis.md
│
├── model/
│   ├── adapter/
│   ├── tokenizer/
│   └── manifest.json
│
├── reports/
│   ├── experiment_summary.md
│   ├── model_comparison.md
│   └── next_experiment.md
│
└── logs/
    └── experiment.log
```

---

# 102. Final Experiment Summary

The system must generate a final report similar to:

```text
==================================================
QWEN PYTHON DPO EXPERIMENT
==================================================

Experiment:
    qwen-python-dpo-v1

Base Model:
    Qwen/...

Training:
    DPO + QLoRA

Training Examples:
    4,000

Held-out Problems:
    500

--------------------------------------------------
MODEL PERFORMANCE
--------------------------------------------------

             Base       DPO       Delta

pass@1       42.0%      48.0%     +6.0 pp
pass@5       58.4%      63.1%     +4.7 pp
pass@10      65.2%      69.0%     +3.8 pp

Syntax       97.2%      98.4%     +1.2 pp

Timeout       1.8%       1.5%     -0.3 pp

--------------------------------------------------
COMPARISON
--------------------------------------------------

DPO wins:      83
Ties:         367
Base wins:     50

--------------------------------------------------
ANALYSIS
--------------------------------------------------

Major improvement:
    Edge-case handling

Major regression:
    Dynamic programming

Potential data gap:
    Dynamic programming

--------------------------------------------------
DECISION
--------------------------------------------------

Model status:
    EXPERIMENTAL

Recommendation:
    Generate additional DP-focused preference data.

Next hypothesis:
    Improved DP coverage will increase held-out
    pass@1 without causing regressions.

==================================================
```

The numbers above are illustrative only.

---

# 103. Final System Definition

After Step 12, the project is no longer merely:

```text
"Fine-tune Qwen with DPO."
```

It becomes:

```text
A reproducible Python-code preference-learning system
that can generate data, train Qwen, objectively evaluate
the resulting model, analyze failures, and create the
next controlled experiment.
```

---

# 104. Final Closed-Loop Architecture

```text
                         ┌───────────────────────┐
                         │ Python Problem Pool   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Candidate Generation  │
                         │       Qwen            │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Docker + pytest       │
                         │ Objective Evaluation  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Preference Generation │
                         │ chosen / rejected     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ DPO + QLoRA Training   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Held-out Benchmark    │
                         │ Base vs DPO           │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Error Analysis        │
                         │ Data Gap Analysis     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Next Experiment       │
                         │ Hypothesis            │
                         └───────────┬───────────┘
                                     │
                                     ▼
                            Generate New Data
                                     │
                                     └──────────────┐
                                                    │
                                                    ▼
                                             DPO Training
```

---

# 105. Definition of Done

Step 12 is complete when a new user can clone the repository, configure a Qwen model and run:

```bash
python -m python_dpo experiment preflight \
    --config configs/experiments/qwen_python_dpo_v1.yaml

python -m python_dpo experiment run \
    --config configs/experiments/qwen_python_dpo_v1.yaml \
    --smoke-test
```

and the system automatically performs the complete workflow:

```text
Problem
   ↓
Candidate
   ↓
Docker
   ↓
pytest
   ↓
Preference
   ↓
DPO
   ↓
Held-out Evaluation
   ↓
Error Analysis
   ↓
Model Package
   ↓
Experiment Report
```

with every artifact, configuration, model, dataset, metric and result traceable to a unique experiment ID.

The implementation must not automatically start another training iteration after completing Step 12.

The next experiment must be explicitly approved or invoked by the user.
