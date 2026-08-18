# Python DPO Data Generation Pipeline

## Step 10 — Base vs DPO Model Evaluation

**Specification Version:** 1.0
**Status:** Implementation Specification
**Step:** 10 of 12
**Depends On:** Step 1 — Project Skeleton
**Depends On:** Step 2 — Python Problem Dataset
**Depends On:** Step 3 — Qwen Candidate Generator
**Depends On:** Step 4 — Candidate Persistence
**Depends On:** Step 5 — Docker Sandbox
**Depends On:** Step 6 — Candidate Test Executor
**Depends On:** Step 7 — Candidate Evaluation and Ranking
**Depends On:** Step 8 — DPO Preference Pair Generation
**Depends On:** Step 9 — DPO/QLoRA Training

---

# 1. Objective

Evaluate whether the DPO-trained Qwen Coder model performs better at Python programming than the original base Qwen model.

The evaluation must compare:

```
Base Qwen
    VS
DPO-trained Qwen + LoRA
```

using:

* the same held-out problems
* the same prompts
* the same generation parameters
* the same number of samples
* the same execution sandbox
* the same pytest test suite
* the same evaluation metrics

The primary objective is to measure:

> Whether DPO improves Python programming correctness on previously unseen problems.

---

# 2. Core Experimental Principle

The evaluation must answer:

```
Did DPO improve Python coding performance?
```

It must NOT merely answer:

```
Did DPO training loss decrease?
```

Therefore the primary evaluation signal must come from:

```
generated Python code
    ↓
Docker sandbox
    ↓
pytest
    ↓
objective test results
```

---

# 3. Experimental Design

The basic experiment is:

```text
                 HELD-OUT PROBLEMS
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
          Base Qwen           DPO Qwen
              │                   │
              ▼                   ▼
        Generate code       Generate code
              │                   │
              ▼                   ▼
          Candidate           Candidate
              │                   │
              └─────────┬─────────┘
                        ▼
                  Same pytest tests
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
         Base results        DPO results
              │                   │
              └─────────┬─────────┘
                        ▼
                  Statistical
                    comparison
```

---

# 4. Scope

This stage MUST implement:

1. Held-out benchmark loading.
2. Base-model inference.
3. DPO-model inference.
4. Identical generation configuration.
5. Multiple-sample generation.
6. Candidate persistence.
7. Candidate evaluation.
8. pass@1.
9. pass@k.
10. syntax-success rate.
11. runtime-success rate.
12. timeout rate.
13. test pass rate.
14. problem-level comparison.
15. Base-vs-DPO comparison.
16. aggregate metrics.
17. bootstrap confidence intervals.
18. statistical significance analysis.
19. experiment reproducibility.
20. evaluation report.
21. CLI commands.
22. Unit tests.
23. Integration tests.

This stage MUST NOT:

* modify the DPO model
* modify the base model
* regenerate the training dataset
* tune DPO hyperparameters
* tune generation parameters based on the test set
* retrain the model
* use the test set for training
* use an LLM judge as the primary correctness metric

---

# 5. Critical Dataset Separation

The evaluation benchmark must be completely independent of the problems used to create DPO preference pairs.

There must be:

```
training problems
validation problems
held-out test problems
```

The model must NOT have seen the held-out problem prompts during:

* candidate generation
* preference generation
* DPO training
* hyperparameter selection

---

# 6. Evaluation Dataset

The evaluation dataset should contain:

```
problem_id
prompt
function_signature
tests
```

The reference solution may be retained for benchmark validation.

The model must NOT receive:

```
reference_solution
```

during inference.

---

# 7. Test Leakage Prevention

Before evaluation, validate:

```
evaluation_problem_ids ∩ training_problem_ids == empty
```

and:

```
evaluation_problem_ids ∩ validation_problem_ids == empty
```

If overlap exists:

```
FAIL evaluation
```

Do not continue with a contaminated benchmark.

---

# 8. Evaluation Dataset Version

Create a benchmark version.

Example:

```
python_eval_v1
```

Persist:

```
benchmark_version
```

in every evaluation run.

---

# 9. Benchmark Manifest

Create:

```
benchmarks/python_eval_v1/manifest.json
```

containing:

```
benchmark_version
problem_ids
problem_count
creation_date
dataset_hash
source_dataset_version
```

---

# 10. Benchmark Hash

Calculate SHA-256 for the benchmark dataset.

Store:

```
benchmark_hash
```

This prevents accidental changes to the evaluation set.

---

# 11. Model Variants

Step 10 must support at least:

```
base
dpo
```

Example:

```
base:
    Qwen/<model>

dpo:
    Qwen/<model> + adapter/<adapter>
```

---

# 12. Base Model

The base model must be loaded exactly according to the model revision used during training.

Record:

```
model_name
model_revision
```

---

# 13. DPO Model

The DPO model must use:

```
same base model
+
trained LoRA adapter
```

Record:

```
base_model
base_revision
adapter_path
training_run_id
```

---

# 14. Base Model Integrity

The evaluation system must verify that the base model used for DPO evaluation is the same base model used during Step 9.

If not:

```
FAIL
```

unless explicitly configured for a separate experiment.

---

# 15. Adapter Integrity

Verify:

```
adapter exists
adapter_config exists
adapter weights exist
adapter references compatible base model
```

If adapter loading fails:

```
FAIL evaluation
```

Do not silently fall back to the base model.

---

# 16. Inference Architecture

Create:

```
ModelEvaluator
```

with conceptually:

```
evaluate_model(
    model,
    benchmark
)
```

The evaluator should be model-agnostic.

Implement:

```
BaseModelRunner
```

and:

```
AdapterModelRunner
```

or an equivalent abstraction.

---

# 17. Model Loading

The model should be loaded once per evaluation process.

Do not reload the model for every problem.

Conceptually:

```
load model
   ↓
generate problem 1
generate problem 2
generate problem 3
   ...
   ↓
unload model
```

---

# 18. GPU Usage

Inference should use the GPU where available.

Record:

```
GPU
VRAM
CUDA
PyTorch
```

---

# 19. Inference Quantization

Use the same quantization approach for:

```
Base Qwen
DPO Qwen
```

where practical.

For example:

```
4-bit NF4
```

This ensures that model comparisons are not affected by different inference precision.

---

# 20. Generation Configuration

Base and DPO models MUST use identical:

```
temperature
top_p
top_k
max_new_tokens
repetition_penalty
stop configuration
```

unless a specific experiment explicitly changes them.

---

# 21. Generation Configuration Example

Initial configuration:

```
generation:
  temperature: 0.2
  top_p: 0.95
  max_new_tokens: 512
  do_sample: true
```

All values must be configurable.

---

# 22. Deterministic Comparison

For deterministic evaluation:

```
temperature = 0
```

may be used.

However, programming tasks benefit from pass@k sampling.

Therefore the benchmark must support both:

```
deterministic mode
```

and:

```
sampling mode
```

---

# 23. Recommended Primary Evaluation

Use:

```
temperature > 0
```

and:

```
n > 1
```

to measure pass@k.

This is more informative than measuring only one deterministic response.

---

# 24. Number of Samples

Support:

```
k ∈ {1, 5, 10}
```

Example:

```
num_samples = 10
```

For each problem:

```
Base → 10 candidates
DPO  → 10 candidates
```

---

# 25. Same Random Seeds

Base and DPO evaluations should use the same seed schedule where practical.

For example:

```
problem 001:
    seeds = [1001 ... 1010]
```

Both models receive the same seed sequence.

This reduces random sampling variance.

---

# 26. Seed Recording

Persist:

```
global_seed
per_problem_seed
sample_index
```

Example:

```
problem_id = p101
sample_index = 3
seed = 100103
```

---

# 27. Prompt Identity

Base and DPO models must receive exactly the same prompt.

Calculate:

```
prompt_sha256
```

and verify equality.

---

# 28. Prompt Formatting

Use the same tokenizer/chat-template handling for both models.

The only difference between model runs should be:

```
base weights
```

versus:

```
base + DPO adapter
```

---

# 29. No Prompt Engineering Difference

Do not use:

```
better prompt for DPO
different system prompt
different instruction format
different examples
```

The comparison must isolate the effect of DPO.

---

# 30. Candidate Generation

For every:

```
problem_id
model_variant
sample_index
```

generate one candidate.

Example:

```
p101
  base_001
  base_002
  ...
  base_010
```

and:

```
p101
  dpo_001
  dpo_002
  ...
  dpo_010
```

---

# 31. Candidate Persistence

Reuse the Step 4 candidate persistence mechanism.

Every evaluation candidate should record:

```
evaluation_run_id
model_variant
problem_id
sample_index
seed
prompt_hash
generated_code
```

---

# 32. Candidate Provenance

Each candidate must be traceable to:

```
benchmark_version
evaluation_run_id
model_variant
model_name
model_revision
adapter
generation_config
seed
```

---

# 33. Code Extraction

If the model returns:

````
```python
def solution(...):
    ...
````

extract the Python code.

If the model returns plain Python:

```
def solution(...):
    ...
```

use it directly.

---

# 34. Code Extraction Consistency

Use the same extraction function for:

```
Base Qwen
DPO Qwen
```

Do not use model-specific extraction heuristics unless necessary.

---

# 35. Code Extraction Failure

If code cannot be extracted:

```
status = generation_error
```

Do not execute malformed natural-language output.

Record:

```
raw_response
```

for debugging.

---

# 36. Raw Response

Persist raw model response separately from:

```
extracted_code
```

This allows debugging extraction failures.

Do not include raw responses in the final benchmark metrics unless needed.

---

# 37. Candidate Evaluation

Every generated candidate must be evaluated using:

```
Step 5 Docker Sandbox
+
Step 6 Candidate Test Executor
```

Do not implement a separate execution mechanism for Step 10.

---

# 38. Evaluation Consistency

The same test suite must be used for:

```
Base candidate
DPO candidate
```

No test modification is permitted between models.

---

# 39. Test Result

For each candidate record:

```
tests_total
tests_passed
tests_failed
tests_error
timeout
status
```

---

# 40. Candidate Correctness

A generated candidate is:

```
correct
```

only if:

```
all tests pass
```

A candidate with:

```
9/10
```

is:

```
incorrect
```

for exact-correctness purposes.

---

# 41. pass@1

For each problem:

```
pass@1
```

is the probability that one generated solution passes all tests.

For deterministic single-sample evaluation:

```
pass@1 = number_correct / number_problems
```

---

# 42. pass@k

For each problem, generate:

```
n
```

samples.

If:

```
c
```

of those samples are correct, estimate pass@k using the standard unbiased estimator:

```
pass@k =
    1 - C(n-c, k) / C(n, k)
```

when:

```
n >= k
```

and:

```
c <= n
```

If:

```
c >= n - k + 1
```

then:

```
pass@k = 1
```

---

# 43. pass@k Requirements

Support:

```
pass@1
pass@5
pass@10
```

provided:

```
n >= 10
```

for pass@10.

---

# 44. Do Not Approximate pass@k

Do not calculate:

```
c / n
```

and call it:

```
pass@k
```

The pass@k metric must use the proper estimator.

---

# 45. Problem-Level pass@k

Compute pass@k separately for each problem.

Then aggregate across problems.

Do not simply treat all generated samples as independent observations without preserving problem boundaries.

---

# 46. Aggregate pass@k

Report:

```
mean_problem_pass_at_1
mean_problem_pass_at_5
mean_problem_pass_at_10
```

The exact aggregation method must be documented.

---

# 47. Test Pass Rate

Also report:

```
mean_test_pass_rate
```

where:

```
test_pass_rate =
    tests_passed / tests_total
```

This provides a more granular signal than pass/fail.

---

# 48. Exact Problem Solve Rate

Report:

```
problems_solved
```

and:

```
problems_total
```

Therefore:

```
solve_rate =
    problems_solved / problems_total
```

This is equivalent to deterministic pass@1 when exactly one sample is generated.

---

# 49. Syntax Success Rate

Report:

```
syntax_success_rate
```

Definition:

```
syntactically_valid_candidates /
total_generated_candidates
```

---

# 50. Execution Success Rate

Report:

```
execution_success_rate
```

Definition:

```
candidates_that_execute_without_runtime_error /
total_candidates
```

---

# 51. Timeout Rate

Report:

```
timeout_rate
```

Definition:

```
timed_out_candidates /
total_candidates
```

---

# 52. Generation Failure Rate

Report:

```
generation_failure_rate
```

Definition:

```
code_extraction_or_generation_failures /
total_generation_attempts
```

---

# 53. Test Failure Distribution

Report:

```
0 tests passed
1–20%
20–40%
40–60%
60–80%
80–99%
100%
```

This helps identify whether DPO improves partial correctness even when full pass@1 does not improve.

---

# 54. Base vs DPO Comparison

For every problem calculate:

```
base_pass
dpo_pass
```

and:

```
improvement
```

Possible values:

```
+1
 0
-1
```

where:

```
+1 = DPO solved, base failed
 0 = both same
-1 = base solved, DPO failed
```

---

# 55. Win/Tie/Loss

Report:

```
DPO wins
ties
DPO losses
```

Example:

```
DPO wins:   31
ties:       52
losses:     17
```

---

# 56. Win Rate

Calculate:

```
dpo_win_rate =
    dpo_wins /
    (dpo_wins + dpo_losses)
```

Do not include ties in the denominator.

---

# 57. Problem-Level Test Improvement

For each problem calculate:

```
dpo_test_pass_rate
base_test_pass_rate
```

and:

```
delta =
    dpo_test_pass_rate -
    base_test_pass_rate
```

This can reveal improvement even when both models fail the complete problem.

---

# 58. Candidate-Level Comparison

For each problem and sample index:

```
base_sample_i
dpo_sample_i
```

can be compared.

However, the primary metric should remain problem-level because sampled generations are not fully independent.

---

# 59. Statistical Confidence

Use bootstrap confidence intervals for aggregate metrics.

At minimum calculate:

```
pass@1
pass@5
pass@10
```

confidence intervals.

Default:

```
95%
```

---

# 60. Bootstrap Unit

Bootstrap at the:

```
problem level
```

not:

```
candidate level
```

because each problem produces multiple candidate samples.

---

# 61. Bootstrap Procedure

For:

```
N problems
```

sample:

```
N problems
```

with replacement.

Calculate the metric.

Repeat:

```
1000
```

or more iterations.

Calculate:

```
2.5th percentile
97.5th percentile
```

for a 95% interval.

---

# 62. Paired Comparison

Because Base and DPO are evaluated on the same problems, use paired comparisons.

At minimum support:

```
paired bootstrap
```

for:

```
DPO pass@1 - Base pass@1
```

---

# 63. Statistical Test

Optionally report:

```
McNemar's test
```

for paired binary problem-level outcomes:

```
Base solved / DPO solved
```

This is particularly useful for pass@1 with one deterministic sample.

---

# 64. Statistical Significance

Report:

```
effect_size
confidence_interval
p_value
```

where applicable.

Do not claim improvement solely because:

```
DPO score > Base score
```

---

# 65. Practical Significance

Statistical significance alone is insufficient.

Report:

```
absolute improvement
relative improvement
```

Example:

```
Base pass@1 = 42%
DPO pass@1  = 48%
```

Absolute:

```
+6 percentage points
```

Relative:

```
+14.3%
```

---

# 66. Confidence Interval Example

Example:

```
Base pass@1:
    0.42
    95% CI [0.36, 0.48]

DPO pass@1:
    0.48
    95% CI [0.42, 0.54]

Difference:
    +0.06
    95% CI [0.01, 0.11]
```

---

# 67. Benchmark Categories

If the problem dataset contains categories, report metrics by:

```
category
```

Examples:

```
algorithms
data structures
strings
recursion
dynamic programming
file processing
numerical
OOP
```

Do not invent categories if the benchmark does not define them.

---

# 68. Difficulty Buckets

If the problem dataset contains difficulty metadata, report:

```
easy
medium
hard
```

separately.

If difficulty is not available, do not infer it using the evaluation result.

---

# 69. Generalization Analysis

Compare:

```
training problem performance
held-out problem performance
```

The primary success criterion must be:

```
held-out performance
```

Do not report training-problem performance as evidence of generalization.

---

# 70. Overfitting Detection

Flag potential overfitting if:

```
training performance increases
```

while:

```
held-out performance decreases
```

This should produce a warning.

---

# 71. Catastrophic Regression

A DPO model must not be considered successful if Python performance improves in one metric while another major metric collapses.

Example:

```
pass@1 +5%
syntax errors +30%
```

This should trigger a regression warning.

---

# 72. Regression Metrics

Compare:

```
syntax success
execution success
timeout rate
test pass rate
pass@1
pass@5
pass@10
```

between:

```
Base
DPO
```

---

# 73. Generation Length

Report:

```
average generated tokens
p50
p90
p95
maximum
```

for:

```
Base
DPO
```

This can reveal whether DPO causes excessive verbosity.

---

# 74. Code Length

Report:

```
average lines of code
p50
p90
```

for:

```
Base
DPO
```

Do not use code length as a correctness metric.

---

# 75. Inference Latency

Measure:

```
generation latency
```

for:

```
Base
DPO
```

Report:

```
mean
p50
p95
```

The comparison must use identical hardware and generation configuration.

---

# 76. Token Throughput

Report:

```
generated_tokens / second
```

for:

```
Base
DPO
```

This is secondary.

The primary objective remains coding correctness.

---

# 77. Memory Usage

Record:

```
peak GPU memory
```

for:

```
Base
DPO
```

This verifies that the adapter does not unexpectedly alter inference memory behavior.

---

# 78. Evaluation Environment

Persist:

```
GPU
VRAM
CUDA
PyTorch
Transformers
PEFT
model revision
adapter revision
```

---

# 79. Evaluation Run ID

Create:

```
evaluation_run_id
```

Example:

```
benchmark_20260818_140000_a123
```

This is distinct from:

```
training_run_id
```

and:

```
preference_run_id
```

---

# 80. Evaluation Directory

Use:

```
data/model_evaluations/runs/<evaluation_run_id>/
```

Structure:

```
<evaluation_run_id>/
    manifest.json
    config.yaml
    benchmark_manifest.json
    model_base/
    model_dpo/
    generations/
    evaluations/
    metrics/
    reports/
    logs/
```

The model directories may contain metadata rather than duplicated model weights.

---

# 81. Generation Artifacts

Persist generated candidates as JSONL:

```
generations/base.jsonl
generations/dpo.jsonl
```

Each record should contain:

```
problem_id
sample_index
seed
prompt_hash
raw_response
extracted_code
generation_time_ms
```

---

# 82. Evaluation Artifacts

Persist:

```
evaluations/base.jsonl
evaluations/dpo.jsonl
```

Each record should contain:

```
problem_id
sample_index
tests_total
tests_passed
tests_failed
tests_error
timeout
status
duration_ms
```

---

# 83. Metrics

Persist:

```
metrics/summary.json
```

containing:

```
base_pass_at_1
dpo_pass_at_1
base_pass_at_5
dpo_pass_at_5
base_pass_at_10
dpo_pass_at_10
base_test_pass_rate
dpo_test_pass_rate
base_timeout_rate
dpo_timeout_rate
base_syntax_success
dpo_syntax_success
```

---

# 84. Comparison Report

Create:

```
reports/base_vs_dpo.json
```

and:

```
reports/base_vs_dpo.md
```

The Markdown report should contain:

```
Executive Summary
Benchmark Description
Model Configuration
Generation Configuration
pass@k Results
Test Pass Rate
Win/Tie/Loss
Statistical Analysis
Regression Analysis
Latency
Memory
Failure Analysis
Conclusions
```

---

# 85. Executive Summary

The report should produce a statement like:

```
DPO improved pass@1 from 42.0% to 48.0%,
an absolute improvement of 6.0 percentage points.
```

or:

```
DPO did not improve pass@1.
Base: 42.0%
DPO: 41.0%
```

Do not make unsupported causal claims.

---

# 86. Success Criteria

Define configurable success criteria.

Initial recommendation:

```
DPO is successful if:

1. pass@1 improves
2. pass@5 does not regress materially
3. syntax success does not regress materially
4. timeout rate does not increase materially
5. paired confidence interval supports the improvement
```

---

# 87. Minimum Improvement Threshold

Do not declare success for trivial differences.

Initial configurable threshold:

```
minimum_pass_at_1_improvement = 0.02
```

meaning:

```
+2 percentage points
```

This is a configurable experimental criterion, not a universal statistical standard.

---

# 88. Regression Threshold

Example:

```
maximum_allowed_regression = 0.02
```

If DPO causes more than a 2-point regression in another important metric:

```
flag evaluation
```

---

# 89. No Automatic Model Promotion

Step 10 must NOT automatically declare:

```
DPO model = production model
```

The evaluation should produce evidence.

A later model-selection or deployment stage can decide whether to promote it.

---

# 90. Baseline Reproducibility

The Base Qwen evaluation should be persisted.

Do not rely on a previously reported baseline number.

Every experiment should run:

```
Base
+
DPO
```

under the same evaluation conditions whenever practical.

---

# 91. Benchmark Stability

The benchmark itself must not change between model comparisons.

If the benchmark changes:

```
create a new benchmark_version
```

Do not compare:

```
Base on benchmark A
```

against:

```
DPO on benchmark B
```

---

# 92. Generation Cache

Support generation caching.

If:

```
same model
same model revision
same prompt
same generation config
same seed
```

then the existing generation may be reused.

The cache key must include:

```
model identity
adapter identity
prompt hash
generation config
seed
```

---

# 93. Evaluation Cache

Similarly cache candidate evaluation results.

The cache key should include:

```
candidate hash
problem ID
test suite hash
evaluator version
sandbox version
```

This avoids unnecessary Docker execution.

---

# 94. No Cross-Model Cache Collision

Base and DPO candidates must never share a cache entry merely because:

```
prompt is identical
```

The model identity must be part of the cache key.

---

# 95. CLI

Add:

```
python -m python_dpo evaluate-model \
    --benchmark python_eval_v1 \
    --training-run-id TRAINING_RUN_ID
```

This runs:

```
Base
+
DPO
```

---

# 96. Base-Only Evaluation

Support:

```
--model base
```

for debugging.

---

# 97. DPO-Only Evaluation

Support:

```
--model dpo
```

for debugging.

---

# 98. Number of Samples

Support:

```
--num-samples 10
```

Example:

```
python -m python_dpo evaluate-model \
    --benchmark python_eval_v1 \
    --training-run-id TRAINING_RUN_ID \
    --num-samples 10
```

---

# 99. Problem Limit

Support:

```
--limit
```

Example:

```
--limit 5
```

This is useful for debugging.

---

# 100. Smoke Evaluation

Support:

```
--smoke-test
```

The smoke test should:

* use 1–3 problems
* generate 1 sample
* evaluate Base
* evaluate DPO
* produce a comparison report

---

# 101. Full Evaluation

Only after the smoke evaluation succeeds should the full benchmark be run.

Example:

```
python -m python_dpo evaluate-model \
    --benchmark python_eval_v1 \
    --training-run-id TRAINING_RUN_ID \
    --num-samples 10
```

---

# 102. Evaluation Validation CLI

Add:

```
python -m python_dpo evaluate-model validate \
    --evaluation-run-id EVAL_RUN_ID
```

Validate:

* benchmark integrity
* model identity
* adapter identity
* prompt equality
* generation configuration equality
* candidate counts
* test counts
* metric consistency

---

# 103. Report CLI

Add:

```
python -m python_dpo evaluate-model report \
    --evaluation-run-id EVAL_RUN_ID
```

Generate:

```
base_vs_dpo.md
base_vs_dpo.json
```

---

# 104. Statistical Analysis CLI

Add:

```
python -m python_dpo evaluate-model stats \
    --evaluation-run-id EVAL_RUN_ID
```

Report:

```
pass@k
confidence intervals
win/tie/loss
paired comparison
p-values where applicable
```

---

# 105. Statistical Reproducibility

The bootstrap procedure must use a fixed seed.

Example:

```
bootstrap_seed: 42
```

Persist:

```
bootstrap_iterations
bootstrap_seed
```

---

# 106. Bootstrap Iterations

Default:

```
1000
```

Make configurable.

For final experiments, support:

```
5000
10000
```

if computationally practical.

---

# 107. Paired Bootstrap

For every bootstrap sample:

```
sample problems with replacement
```

Then calculate:

```
DPO metric
Base metric
DPO - Base
```

Store the distribution of differences.

---

# 108. Confidence Interval

Report:

```
lower
upper
```

for:

```
Base
DPO
Difference
```

---

# 109. Problem-Level Win Analysis

For each problem:

```
Base:
    best_pass_rate

DPO:
    best_pass_rate
```

Calculate:

```
DPO better
equal
Base better
```

---

# 110. Sample-Level vs Problem-Level

Do not confuse:

```
sample-level accuracy
```

with:

```
problem-level solve rate
```

A problem with:

```
10 samples
1 successful
```

contributes:

```
pass@10
```

rather than:

```
10% problem accuracy
```

---

# 111. pass@k Validation

The implementation must test the pass@k estimator against known values.

Example:

```
n = 10
c = 10
```

Expected:

```
pass@1 = 1
pass@5 = 1
pass@10 = 1
```

Example:

```
n = 10
c = 0
```

Expected:

```
pass@k = 0
```

for:

```
k <= 10
```

---

# 112. Model Comparison Test

Create a mock benchmark:

```
10 problems
```

Base:

```
4 solved
```

DPO:

```
6 solved
```

Expected:

```
Base pass@1 = 0.4
DPO pass@1 = 0.6
Improvement = +0.2
```

---

# 113. Regression Test

Create a mock case:

```
Base:
    8/10 problems solved

DPO:
    6/10 solved
```

Expected:

```
DPO regression = -0.2
```

The report must clearly identify the regression.

---

# 114. Pairing Test

Verify that Base and DPO receive:

```
identical problem IDs
```

and:

```
identical prompts
```

in the same benchmark.

---

# 115. Generation Configuration Test

Verify:

```
base.generation_config == dpo.generation_config
```

unless explicitly configured otherwise.

---

# 116. Seed Test

Verify that:

```
base.seed[i] == dpo.seed[i]
```

for corresponding sample indices.

---

# 117. Sandbox Reuse Test

Verify that Base and DPO candidates are evaluated through the same:

```
sandbox configuration
evaluator version
pytest version
test suite
```

---

# 118. Adapter Isolation Test

Verify that:

```
Base evaluation
```

does not accidentally load the DPO adapter.

And:

```
DPO evaluation
```

does not silently fall back to Base.

---

# 119. Adapter Effect Test

For a smoke prompt:

```
Base response
```

and:

```
DPO response
```

should be independently recorded.

They may be identical.

Identical output does not constitute an error.

---

# 120. Evaluation Failure Handling

If one candidate fails infrastructure evaluation:

```
do not automatically count it as incorrect
```

Record:

```
infrastructure_error
```

and exclude it from candidate-level correctness statistics.

At the problem-level benchmark analysis, report the missing/invalid evaluation.

---

# 121. Incomplete Benchmark

If:

```
Base evaluated 100 problems
```

but:

```
DPO evaluated 98
```

do not silently calculate a comparison over mismatched sets.

Either:

1. rerun the missing evaluations, or
2. explicitly create a paired subset and report it.

---

# 122. Paired Evaluation Set

The primary Base-vs-DPO comparison must use:

```
problems evaluated successfully for both models
```

Call this:

```
paired_problem_set
```

Report its size.

---

# 123. Benchmark Completeness

The report must contain:

```
benchmark_problems
base_successfully_evaluated
dpo_successfully_evaluated
paired_problems
```

---

# 124. Failure Analysis

Generate:

```
reports/failure_analysis.json
```

Categorize failures:

```
generation_error
syntax_error
import_error
runtime_error
assertion_failure
timeout
infrastructure_error
```

---

# 125. Failure Distribution

Compare Base vs DPO:

```
syntax errors
runtime errors
assertion failures
timeouts
```

This may reveal how DPO changes failure modes.

---

# 126. Error Clustering

Where practical, group failures by:

```
error_type
```

For example:

```
TypeError
IndexError
AttributeError
AssertionError
```

Do not use an LLM to classify errors in Step 10.

Simple deterministic classification is sufficient.

---

# 127. Test-Level Analysis

For each problem, compare:

```
base tests passed
dpo tests passed
```

Calculate:

```
test_pass_delta
```

This reveals partial improvement.

---

# 128. Problem Difficulty Analysis

If difficulty metadata exists, report:

```
pass@1 by difficulty
```

Example:

```
Easy:
    Base 72%
    DPO 78%

Medium:
    Base 45%
    DPO 52%

Hard:
    Base 18%
    DPO 20%
```

---

# 129. Category Analysis

If categories exist:

```
pass@1 by category
```

This may reveal whether DPO improves:

```
algorithms
data structures
recursion
strings
etc.
```

---

# 130. Regression Problem List

Generate:

```
reports/regressions.jsonl
```

containing problems where:

```
Base solved
DPO failed
```

Include:

```
problem_id
prompt
base_code
dpo_code
base_test_result
dpo_test_result
```

---

# 131. Improvement Problem List

Generate:

```
reports/improvements.jsonl
```

containing problems where:

```
Base failed
DPO solved
```

This is particularly valuable for qualitative analysis.

---

# 132. Tie Problem List

Generate:

```
reports/ties.jsonl
```

where:

```
Base and DPO have equivalent correctness outcomes
```

---

# 133. Qualitative Analysis

Do not use an LLM automatically in Step 10.

However, preserve artifacts so that a later human/LLM analysis can inspect:

```
improved examples
regressions
partial improvements
```

---

# 134. Important Benchmark Principle

The evaluation benchmark must not be used to modify:

```
DPO beta
LoRA rank
learning rate
epochs
preference policy
```

If these are changed based on benchmark performance, create a new training experiment and treat the benchmark as having been consulted.

---

# 135. Hyperparameter Search

Do NOT implement automatic hyperparameter optimization in Step 10.

If hyperparameter experiments are needed:

```
use a separate validation benchmark
```

The held-out test benchmark should remain untouched.

---

# 136. Multiple Training Runs

Support comparing:

```
training_run_A
training_run_B
training_run_C
```

against the same:

```
Base Qwen
```

This allows experiments such as:

```
strict DPO
margin DPO
different beta
different LoRA rank
```

---

# 137. Multi-Experiment Comparison

Support:

```
python -m python_dpo evaluate-model compare \
    --runs RUN_A,RUN_B,RUN_C
```

Report:

```
pass@1
pass@5
pass@10
win/tie/loss
```

for each run.

---

# 138. Baseline Caching

The Base Qwen benchmark results may be cached and reused for multiple DPO experiments if:

```
base model identity
benchmark
generation config
seeds
evaluator version
```

are identical.

---

# 139. Cache Integrity

Never reuse Base results if:

```
model revision differs
```

or:

```
generation configuration differs
```

or:

```
benchmark differs
```

or:

```
seed differs
```

---

# 140. Evaluation Configuration

Create:

```
configs/evaluation/python_eval.yaml
```

Example:

```
benchmark:
  name: python_eval_v1

generation:
  temperature: 0.2
  top_p: 0.95
  max_new_tokens: 512
  num_samples: 10

statistics:
  bootstrap_iterations: 1000
  bootstrap_seed: 42
  confidence_level: 0.95
```

---

# 141. Final Evaluation Report

The final report must answer:

### 1. Did DPO improve pass@1?

### 2. Did DPO improve pass@5?

### 3. Did DPO improve pass@10?

### 4. Did DPO reduce syntax errors?

### 5. Did DPO reduce runtime failures?

### 6. Did DPO increase timeout rate?

### 7. How many problems did DPO solve that Base did not?

### 8. How many problems did Base solve that DPO did not?

### 9. Are the differences statistically meaningful?

### 10. Does DPO generalize to unseen Python problems?

---

# 142. Final Model Decision

The report may provide:

```
recommended:
    yes/no
```

but this must be based on configurable criteria.

Do not automatically promote the model.

---

# 143. Recommended Default Success Rule

For v1:

```
DPO_SUCCESS if:

DPO pass@1 - Base pass@1 >= 0.02

AND

DPO pass@5 >= Base pass@5 - 0.02

AND

DPO syntax_success >= Base syntax_success - 0.02

AND

DPO timeout_rate <= Base timeout_rate + 0.02

AND

paired confidence interval for pass@1 improvement
does not strongly support a regression
```

These are experimental gates, not universal standards.

---

# 144. Important Limitation

If the benchmark contains only:

```
10 problems
```

do not interpret the result as statistically meaningful.

The benchmark is suitable for:

```
pipeline validation
```

but not:

```
reliable model-performance conclusions.
```

---

# 145. Recommended Benchmark Size

For serious experimentation, target:

```
hundreds of held-out Python problems
```

and preferably:

```
500–1,000+
```

for more stable aggregate estimates.

The exact size depends on problem diversity and difficulty.

---

# 146. Acceptance Criteria

Step 10 is complete only when:

* [ ] Held-out benchmark exists.
* [ ] Training/evaluation leakage check exists.
* [ ] Benchmark is versioned.
* [ ] Benchmark is hashed.
* [ ] Base model evaluation works.
* [ ] DPO model evaluation works.
* [ ] Same prompts are used.
* [ ] Same generation configuration is used.
* [ ] Same seeds are used.
* [ ] Same test suites are used.
* [ ] Same Docker sandbox is used.
* [ ] Candidate generations are persisted.
* [ ] Candidate evaluations are persisted.
* [ ] pass@1 is implemented.
* [ ] pass@5 is implemented.
* [ ] pass@10 is implemented.
* [ ] pass-rate metrics are implemented.
* [ ] syntax metrics are implemented.
* [ ] timeout metrics are implemented.
* [ ] Base-vs-DPO win/tie/loss is implemented.
* [ ] Paired comparison is implemented.
* [ ] Bootstrap confidence intervals are implemented.
* [ ] Failure analysis exists.
* [ ] Improvement examples are persisted.
* [ ] Regression examples are persisted.
* [ ] Benchmark leakage is prevented.
* [ ] Evaluation is reproducible.
* [ ] Evaluation results are versioned.
* [ ] Base model identity is validated.
* [ ] DPO adapter identity is validated.
* [ ] Adapter reload is validated.
* [ ] No test-set hyperparameter tuning is implemented.
* [ ] No automatic model promotion occurs.
* [ ] Smoke evaluation passes.
* [ ] Full evaluation passes.

---

# 147. Verification Procedure

## 147.1 Validate benchmark

Run:

```
python -m python_dpo benchmark validate \
    --benchmark python_eval_v1
```

Expected:

```
Benchmark validation passed.
```

---

## 147.2 Verify training/evaluation separation

Run:

```
python -m python_dpo benchmark check-leakage \
    --benchmark python_eval_v1 \
    --preference-run-id PREF_RUN_ID
```

Expected:

```
No problem leakage detected.
```

---

## 147.3 Hardware check

Run:

```
python -m python_dpo train hardware-check
```

---

## 147.4 Smoke evaluation

Run:

```
python -m python_dpo evaluate-model \
    --benchmark python_eval_v1 \
    --training-run-id TRAINING_RUN_ID \
    --smoke-test
```

Verify:

```
Base generated
DPO generated
Base evaluated
DPO evaluated
comparison generated
```

---

## 147.5 Full evaluation

Run:

```
python -m python_dpo evaluate-model \
    --benchmark python_eval_v1 \
    --training-run-id TRAINING_RUN_ID \
    --num-samples 10
```

---

## 147.6 Validate results

Run:

```
python -m python_dpo evaluate-model validate \
    --evaluation-run-id EVAL_RUN_ID
```

---

## 147.7 Generate report

Run:

```
python -m python_dpo evaluate-model report \
    --evaluation-run-id EVAL_RUN_ID
```

---

## 147.8 Generate statistics

Run:

```
python -m python_dpo evaluate-model stats \
    --evaluation-run-id EVAL_RUN_ID
```

---

# 148. Expected Artifacts

After Step 10:

```
data/
└── model_evaluations/
    └── runs/
        └── <evaluation_run_id>/
            ├── manifest.json
            ├── config.yaml
            ├── benchmark_manifest.json
            ├── generations/
            │   ├── base.jsonl
            │   └── dpo.jsonl
            ├── evaluations/
            │   ├── base.jsonl
            │   └── dpo.jsonl
            ├── metrics/
            │   ├── summary.json
            │   ├── pass_at_k.json
            │   └── bootstrap.json
            ├── reports/
            │   ├── base_vs_dpo.md
            │   ├── base_vs_dpo.json
            │   ├── improvements.jsonl
            │   ├── regressions.jsonl
            │   ├── ties.jsonl
            │   └── failure_analysis.json
            └── logs/
                └── evaluation.log
```

---

# 149. Example Final Report

The report should be capable of producing:

```
Python DPO Evaluation
======================

Benchmark:
    python_eval_v1
    Problems: 500

Base Qwen:
    pass@1: 42.0%
    pass@5: 58.4%
    pass@10: 65.2%

DPO Qwen:
    pass@1: 48.0%
    pass@5: 63.1%
    pass@10: 69.0%

Improvement:
    pass@1: +6.0 pp
    pass@5: +4.7 pp
    pass@10: +3.8 pp

Problem-level comparison:
    DPO wins: 83
    Ties:     367
    Base wins: 50

Syntax success:
    Base: 97.2%
    DPO:  98.4%

Timeout rate:
    Base: 1.8%
    DPO:  1.5%

Conclusion:
    DPO shows positive improvement on the held-out
    Python benchmark.
```

The actual conclusion must be calculated from real results.

---

# 150. What Step 10 Produces

Step 10 produces the evidence needed to answer:

> Did DPO actually make Qwen better at Python programming?

The complete experimental pipeline is now:

```
Problems
   │
   ▼
Qwen candidates
   │
   ▼
Docker + pytest
   │
   ▼
Objective preferences
   │
   ▼
DPO / QLoRA
   │
   ▼
Base Qwen ─────────┐
                   │
DPO Qwen ──────────┤
                   ▼
            Held-out problems
                   │
                   ▼
             Docker + pytest
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
     Base metrics        DPO metrics
         │                   │
         └─────────┬─────────┘
                   ▼
             Statistical
              comparison
                   │
                   ▼
            DPO effectiveness
```

```

---

# 151. Final Implementation Report

After implementation, report:

1. Benchmark version.
2. Benchmark hash.
3. Number of benchmark problems.
4. Base model.
5. DPO model.
6. Adapter.
7. Model revisions.
8. Generation configuration.
9. Number of samples.
10. Random seeds.
11. Base pass@1.
12. DPO pass@1.
13. Base pass@5.
14. DPO pass@5.
15. Base pass@10.
16. DPO pass@10.
17. Test pass rates.
18. Syntax success rates.
19. Timeout rates.
20. Runtime error rates.
21. Generation failure rates.
22. DPO wins.
23. Ties.
24. Base wins.
25. Absolute improvement.
26. Relative improvement.
27. Bootstrap confidence intervals.
28. Statistical test results.
29. Improvement examples.
30. Regression examples.
31. Failure analysis.
32. Peak GPU memory.
33. Inference latency.
34. Tokens/second.
35. Dataset leakage result.
36. Evaluation reproducibility information.
37. Final recommendation.
38. Files created/modified.
39. Dependencies added.
40. Deviations from specification.
41. Known limitations.

Do NOT implement Step 11 automatically.

Wait for explicit approval before implementing the next pipeline stage.
```
