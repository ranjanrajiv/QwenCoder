# Python DPO Data Generation Pipeline

## Step 11 — Error Analysis, Preference Refinement and Iterative Improvement

**Specification Version:** 1.0
**Status:** Implementation Specification
**Step:** 11 of 12
**Depends On:** Steps 1–10

---

# 1. Objective

Analyze the results produced by Step 10 and determine:

1. Where Base Qwen fails.
2. Where DPO Qwen fails.
3. Where DPO improves Base Qwen.
4. Where DPO causes regressions.
5. What classes of Python problems remain difficult.
6. What types of preference data are underrepresented.
7. Whether the preference-generation strategy should be modified.
8. Whether another DPO training iteration should be created.

The output of Step 11 is an **experiment-analysis package** and, optionally, a new refined preference dataset for the next training iteration.

---

# 2. Core Principle

Step 11 must NOT blindly generate more training data.

The process must be:

```text
Step 10 Evaluation
       │
       ▼
Failure Analysis
       │
       ├──────────────┐
       ▼              ▼
DPO Improvements   DPO Regressions
       │              │
       └───────┬──────┘
               ▼
       Error Classification
               │
               ▼
       Data Gap Analysis
               │
               ▼
      Preference Refinement
               │
               ▼
        New Experiment
```

---

# 3. Main Question

The system must answer:

> What should we change in the next DPO iteration to improve held-out Python performance?

Possible answers include:

* generate more difficult Python problems
* generate more candidates per problem
* improve candidate diversity
* add edge-case-heavy examples
* increase preference margin
* add partial-vs-partial preferences
* remove noisy preference pairs
* change generation strategies
* change LoRA configuration
* change DPO beta
* increase dataset size
* change problem distribution

The system must provide evidence before recommending a change.

---

# 4. Scope

Step 11 MUST implement:

1. Evaluation-result ingestion.
2. Base-vs-DPO error comparison.
3. Problem-level error classification.
4. Test-level failure analysis.
5. Error-type aggregation.
6. Improvement analysis.
7. Regression analysis.
8. Category analysis.
9. Difficulty analysis.
10. Preference-data coverage analysis.
11. Candidate diversity analysis.
12. Preference-quality analysis.
13. DPO failure-pattern detection.
14. Data-gap identification.
15. Iteration recommendations.
16. Optional hard-example dataset creation.
17. Optional refined preference dataset creation.
18. Experiment lineage.
19. Analysis reports.
20. CLI.
21. Unit tests.
22. Integration tests.

Step 11 MUST NOT automatically retrain the model.

---

# 5. No Automatic Training

The system may produce:

```
recommended_next_experiment
```

but must NOT automatically execute:

```
Step 9 training
```

unless explicitly requested.

---

# 6. Input Artifacts

Step 11 consumes:

```
evaluation_run_id
```

from Step 10.

It may also consume:

```
preference_run_id
training_run_id
ranking_run_id
candidate_run_id
```

to trace failures back through the pipeline.

---

# 7. Experiment Lineage

The analysis must preserve:

```text
Problem
   │
   ▼
Candidate Run
   │
   ▼
Evaluation Run
   │
   ▼
Ranking Run
   │
   ▼
Preference Run
   │
   ▼
Training Run
   │
   ▼
Model Evaluation Run
   │
   ▼
Step 11 Analysis
```

This lineage is mandatory.

---

# 8. Analysis Run

Create:

```
analysis_run_id
```

Example:

```
analysis_20260818_220000_a123
```

---

# 9. Analysis Directory

Use:

```
data/analysis/runs/<analysis_run_id>/
```

Structure:

```
<analysis_run_id>/
    manifest.json
    config.yaml
    summary.json
    reports/
    classifications/
    improvements/
    regressions/
    data_gaps/
    recommendations/
    refined_dataset/
    logs/
```

---

# 10. Error Classification

Every failed generated solution should receive a deterministic error classification where possible.

Supported initial categories:

```
generation_failure
code_extraction_failure
syntax_error
import_error
runtime_error
assertion_failure
timeout
memory_error
infrastructure_error
```

---

# 11. Error Hierarchy

The error taxonomy should support hierarchical classification.

Example:

```text
runtime_error
    ├── TypeError
    ├── IndexError
    ├── KeyError
    ├── AttributeError
    ├── ValueError
    └── other
```

Similarly:

```text
syntax_error
    ├── invalid_syntax
    ├── indentation_error
    └── other
```

---

# 12. Deterministic Classification

Prefer:

```
pytest result
Python traceback
exception type
exit code
timeout status
```

over an LLM.

Do not introduce an LLM judge as the primary classifier.

---

# 13. LLM-Based Analysis

An optional LLM-based analysis module may be supported later.

If implemented, it must be:

```
secondary analysis
```

and must never replace:

```
pytest-based correctness
```

The implementation must clearly distinguish:

```
objective classification
```

from:

```
semantic interpretation.
```

---

# 14. Base Error Profile

Generate:

```
base_error_profile.json
```

containing:

```
total_candidates
correct_candidates
syntax_failures
runtime_failures
assertion_failures
timeouts
generation_failures
```

---

# 15. DPO Error Profile

Generate:

```
dpo_error_profile.json
```

with the same structure.

This allows direct comparison.

---

# 16. Error Rate Comparison

For every error category calculate:

```
base_rate
dpo_rate
delta
relative_delta
```

Example:

```text
Syntax errors:
    Base = 4.2%
    DPO  = 2.8%
    Delta = -1.4 pp
```

---

# 17. Improvement Problems

An improvement problem is:

```
Base failed
AND
DPO succeeded
```

Persist:

```
improvements.jsonl
```

Each record should contain:

```
problem_id
category
difficulty
base_result
dpo_result
base_code
dpo_code
tests_passed_base
tests_passed_dpo
```

---

# 18. Regression Problems

A regression problem is:

```
Base succeeded
AND
DPO failed
```

Persist:

```
regressions.jsonl
```

Include:

```
problem_id
category
difficulty
base_code
dpo_code
base_test_result
dpo_test_result
failure_type
```

---

# 19. Partial Improvement

A partial improvement occurs when:

```
Base test_pass_rate < DPO test_pass_rate
```

but:

```
DPO does not pass all tests.
```

Example:

```
Base = 4/10
DPO  = 8/10
```

This should be classified separately from:

```
complete improvement.
```

---

# 20. Partial Regression

Similarly:

```
Base = 10/10
DPO  = 7/10
```

must be classified as:

```
partial_regression
```

if DPO does not completely fail the problem.

---

# 21. Complete Improvement

Classify:

```
Base = 0 tests passed
DPO = all tests passed
```

as:

```
complete_improvement
```

---

# 22. Complete Regression

Classify:

```
Base = all tests passed
DPO = 0 tests passed
```

as:

```
complete_regression
```

---

# 23. No Change

If:

```
Base test result == DPO test result
```

classify:

```
unchanged
```

---

# 24. Problem-Level Analysis

For each problem calculate:

```
base_best_score
dpo_best_score

base_solved
dpo_solved

delta

error_category
```

---

# 25. Best Candidate Definition

For sampled evaluation:

```
best_score
```

is:

```
maximum test_pass_rate
```

across generated samples.

For exact correctness:

```
solved = any candidate passes all tests
```

---

# 26. Candidate-Level Analysis

Also preserve candidate-level analysis.

For every sample:

```
model_variant
problem_id
sample_index
test_pass_rate
correctness
error_type
```

This allows analysis of sampling behavior.

---

# 27. Sampling Analysis

For each model report:

```
number of samples
number correct
average test pass rate
best test pass rate
duplicate rate
```

This helps determine whether DPO improves:

```
individual solution quality
```

or:

```
diversity of sampled solutions.
```

---

# 28. Candidate Diversity

Calculate code diversity.

At minimum:

```
exact_code_duplicate_rate
```

For each problem:

```
unique_code_count
total_code_count
```

---

# 29. Diversity Metric

Define:

```
diversity =
    unique_candidates / total_candidates
```

Example:

```
8 unique candidates / 10 samples

diversity = 0.8
```

---

# 30. Diversity Comparison

Compare:

```
Base diversity
DPO diversity
```

This is important because DPO may improve correctness while reducing candidate diversity.

---

# 31. Duplicate Generation Analysis

If DPO generates the same solution repeatedly:

```
report potential mode collapse
```

Example:

```
Base:
    8 unique / 10

DPO:
    3 unique / 10
```

This should trigger a warning.

---

# 32. Preference Coverage

Analyze the original Step 8 preference dataset.

For every problem category calculate:

```
number of preference pairs
strong pairs
medium pairs
average score margin
```

---

# 33. Preference Distribution

Report:

```text
Problem category
    Number of problems
    Number of pairs
    Pairs/problem
    Average margin
    Strong pair percentage
```

---

# 34. Data Gap Detection

Identify categories where:

```
evaluation failures are high
```

but:

```
preference examples are low.
```

Example:

```text
Dynamic Programming

Evaluation:
    DPO pass@1 = 28%

Preference data:
    1.2 pairs/problem

Recommendation:
    increase DP preference examples
```

---

# 35. Difficulty Gap

Compare:

```
training preference distribution
```

against:

```
evaluation problem difficulty distribution.
```

Identify:

```
underrepresented difficulty levels.
```

Example:

```text
Training:
    Easy = 70%
    Medium = 25%
    Hard = 5%

Evaluation:
    Easy = 30%
    Medium = 45%
    Hard = 25%

Potential issue:
    insufficient hard examples
```

---

# 36. Category Gap

Compare:

```
training category distribution
```

against:

```
evaluation category distribution.
```

Calculate:

```
category_coverage_ratio
```

---

# 37. Coverage Ratio

Define:

```
coverage_ratio =
    training_percentage /
    evaluation_percentage
```

Interpretation:

```
~1.0 = balanced

<0.5 = underrepresented

>2.0 = potentially overrepresented
```

Thresholds must be configurable.

---

# 38. Error-to-Data Correlation

For each category calculate:

```
preference_density
DPO_error_rate
```

This helps identify whether poor performance correlates with insufficient training coverage.

Do NOT claim causality.

Use language such as:

```
"potential data gap"
```

rather than:

```
"DPO failed because of insufficient data."
```

---

# 39. Preference Strength Analysis

Compare DPO performance against:

```
strong preference percentage
medium preference percentage
average preference margin
```

This helps determine whether noisy preference data may be limiting performance.

---

# 40. Strict vs Margin Dataset Analysis

If both datasets exist:

```
strict_v1
margin_v1
```

compare their training/evaluation results.

Report:

```
preference_count
average_margin
DPO_pass_at_1
DPO_pass_at_5
DPO_pass_at_10
```

---

# 41. Training Strategy Analysis

If candidate-generation strategy metadata exists, compare:

```
chosen_strategy
rejected_strategy
```

Example:

```text
Strategy              Chosen %
--------------------------------
normal                   41%
edge_case                28%
optimized                20%
alternative              11%
```

---

# 42. Strategy Gap

Identify generation strategies that produce:

```
high-quality candidates
```

but have:

```
low representation in preference data.
```

Potential recommendation:

```
increase sampling from that strategy.
```

---

# 43. Error Pattern Analysis

Identify common patterns such as:

```
off-by-one errors
missing edge cases
incorrect recursion base case
mutable default arguments
incorrect sorting assumptions
incorrect dictionary handling
integer/float conversion errors
inefficient algorithms
incorrect time complexity
incorrect state management
```

The initial implementation should derive these from deterministic metadata where possible.

---

# 44. Semantic Error Classification

Some errors cannot be reliably identified from exception types.

For example:

```
wrong algorithm
incorrect edge-case handling
```

may result in:

```
assertion_failure
```

The system may record:

```
assertion_failure
```

as the objective category.

Do not infer the exact conceptual bug unless an optional analysis layer is used.

---

# 45. Test-Level Failure Analysis

For every failed test calculate:

```
problem_id
test_id
base_pass
dpo_pass
```

This allows identification of recurring edge cases.

---

# 46. Test Failure Frequency

Calculate:

```
failure_count_by_test
```

and:

```
failure_rate_by_test
```

This identifies difficult test cases.

---

# 47. Hard Test Identification

A test is potentially hard if:

```
high failure rate
```

across:

```
Base
and
DPO
```

This can be used to construct future hard examples.

---

# 48. DPO-Specific Hard Tests

A test is DPO-specific difficult if:

```
Base frequently passes
```

but:

```
DPO frequently fails.
```

These should be investigated as regression patterns.

---

# 49. Base-Specific Hard Tests

A test is Base-specific difficult if:

```
Base frequently fails
```

but:

```
DPO frequently passes.
```

These are evidence of successful DPO learning.

---

# 50. Regression Severity

Assign:

```
regression_severity
```

Possible values:

```
low
medium
high
```

Suggested logic:

### Low

small test-pass reduction

### Medium

multiple tests lost

### High

fully correct Base → completely incorrect DPO

---

# 51. Improvement Severity

Similarly:

```
improvement_severity
```

Possible values:

```
low
medium
high
```

A:

```
0/10 → 10/10
```

improvement is:

```
high
```

---

# 52. Regression Threshold

Make configurable:

```
regression_threshold
```

Example:

```
0.2
```

A reduction greater than:

```
20 percentage points
```

may be classified as significant.

---

# 53. Recommendation Engine

Create:

```
RecommendationEngine
```

It must convert observed evidence into potential next experiments.

Example:

```text
Observed:
    hard problems underrepresented

Recommendation:
    increase hard-problem generation

Observed:
    DPO regression on recursion

Recommendation:
    add recursion-focused preference examples

Observed:
    DPO diversity collapsed

Recommendation:
    increase candidate-generation diversity
```

---

# 54. Recommendation Confidence

Every recommendation must include:

```
confidence
```

Possible:

```
low
medium
high
```

---

# 55. Recommendation Evidence

Every recommendation must reference evidence.

Example:

```text
recommendation:
    Increase dynamic-programming examples.

evidence:
    DPO pass@1 = 22%
    DP preference density = 0.8 pairs/problem
    benchmark share = 24%
    training share = 5%

confidence:
    medium
```

---

# 56. No Unsupported Recommendations

Do not generate:

```
"Increase LoRA rank"
```

unless there is evidence suggesting model capacity may be limiting.

Data issues should be investigated before arbitrarily changing model architecture.

---

# 57. Recommended Iteration Priority

Rank recommendations by:

```
expected impact
evidence strength
implementation cost
```

Create:

```
recommendation_score
```

---

# 58. Recommendation Categories

Support:

```
add_data
improve_data_quality
improve_candidate_diversity
change_preference_policy
change_generation_strategy
adjust_dpo_hyperparameters
increase_problem_difficulty
rebalance_problem_categories
investigate_regression
investigate_mode_collapse
```

---

# 59. Hard Example Dataset

Support generating:

```
hard_examples.jsonl
```

containing problems where:

```
Base fails
AND
DPO fails
```

These are candidates for future training data.

---

# 60. Hard Example Selection

Do not automatically add every failure.

Filter based on:

```
reproducibility
problem validity
benchmark quality
error severity
```

---

# 61. Regression Dataset

Create:

```
regression_examples.jsonl
```

containing:

```
Base correct
DPO incorrect
```

These examples should be candidates for future preference generation.

---

# 62. Improvement Dataset

Create:

```
successful_dpo_examples.jsonl
```

containing:

```
Base incorrect
DPO correct
```

These examples are valuable for understanding what DPO learned.

---

# 63. Hard Example Provenance

Every generated hard-example record must contain:

```
source_evaluation_run_id
problem_id
model_variant
benchmark_version
```

Do not create orphaned examples.

---

# 64. Candidate Reuse

If future preference generation uses these examples, it must reference the original problem rather than duplicating the problem definition.

---

# 65. No Automatic Benchmark Contamination

Hard examples originating from the held-out test benchmark MUST NOT automatically be inserted into the next DPO training dataset.

This would contaminate the evaluation benchmark.

Instead:

```
identify the problem pattern
```

and generate:

```
new unseen problems
```

of similar type.

---

# 66. Critical Rule

Never do:

```text
held-out problem fails
       ↓
add held-out problem to training
       ↓
retrain
       ↓
evaluate on same problem
```

This invalidates the benchmark.

---

# 67. Synthetic Hard-Problem Generation

The system may optionally create a specification for generating new problems similar to observed failures.

Example:

```text
Observed:
    off-by-one errors in sliding window problems

Generate:
    100 new sliding-window Python problems
```

But the generated problems must be new problems.

---

# 68. Problem Generation Interface

Create an optional:

```
HardProblemGenerator
```

interface.

It should produce:

```
problem_specifications
```

rather than immediately adding problems to training.

---

# 69. Human Review Gate

New generated problems must pass:

```
problem validation
```

and preferably:

```
human review
```

before entering training.

---

# 70. Preference Refinement

The system may create a new preference-generation specification:

```
refined_preference_plan.json
```

Example:

```text
Focus:
    recursion
    dynamic programming
    edge cases

Target:
    500 additional problems

Candidates/problem:
    8

Preference policy:
    strict

Minimum margin:
    0.2
```

---

# 71. Preference Filtering

Support filtering out potentially noisy preference pairs.

Examples:

```
very small score margin
ambiguous evaluation
flaky tests
infrastructure errors
duplicated code
malformed prompts
```

---

# 72. Flaky Test Detection

If the same candidate produces inconsistent results across repeated executions:

```
mark:

flaky_evaluation
```

Do not use the candidate as preference evidence until resolved.

---

# 73. Flaky Test Threshold

Support:

```
repeated_evaluation_count
```

Example:

```
3
```

If results differ across repetitions:

```
flaky = true
```

---

# 74. Test Stability

A test suite used to create preferences should be stable.

If a test has:

```
nondeterministic behavior
```

the associated preference pairs should be flagged.

---

# 75. Preference Quality Score

Optionally calculate:

```
preference_quality_score
```

based on:

```
score_margin
test stability
correctness confidence
candidate uniqueness
```

This score is for analysis.

Do not automatically replace objective preference labels with it.

---

# 76. Dataset Refinement Policy

Support:

```
retain
remove
regenerate
```

for preference examples.

---

# 77. Refinement Output

Create:

```
refined_preferences.jsonl
```

but do not overwrite the original Step 8 dataset.

---

# 78. Dataset Versioning

If:

```
preference_run_001
```

produces:

```
refined_preferences_002
```

the new dataset must receive a new version.

Example:

```
dpo_preference_v2
```

---

# 79. Lineage

Record:

```
parent_preference_run_id
```

for every refined dataset.

---

# 80. Experiment Proposal

Create:

```
next_experiment.yaml
```

Example:

```yaml
parent_training_run: dpo_001

objective:
  improve:
    - recursion
    - dynamic_programming

data:
  additional_problems: 500
  candidates_per_problem: 8
  preference_policy: strict

training:
  dpo_beta: 0.1
  lora_rank: 16
  learning_rate: 1.0e-5

evaluation:
  benchmark: python_eval_v2
```

The system must not execute this automatically.

---

# 81. Experiment Comparison

Support comparing multiple Step 10 evaluation runs.

Example:

```
strict DPO
margin DPO
larger-data DPO
higher-rank DPO
```

Produce:

```
experiment_comparison.md
```

---

# 82. Experiment Matrix

Report:

| Experiment | Preference Policy | Pairs | pass@1 | pass@5 | pass@10 |
| ---------- | ----------------- | ----: | -----: | -----: | ------: |
| Base       | —                 |     — |    ... |    ... |     ... |
| DPO-Strict | strict            |   ... |    ... |    ... |     ... |
| DPO-Margin | margin            |   ... |    ... |    ... |     ... |

---

# 83. Best Experiment

Identify:

```
best_observed_experiment
```

based on:

```
primary_metric
```

Default:

```
held-out pass@1
```

Secondary:

```
pass@5
pass@10
syntax success
timeout rate
```

---

# 84. Do Not Overfit to One Metric

If:

```
pass@1 improves
```

but:

```
pass@10 collapses
```

the experiment should not automatically be considered better.

The report must highlight the tradeoff.

---

# 85. Training Curve Analysis

If training metrics are available, compare:

```
training loss
validation loss
reward margin
```

against:

```
Step 10 performance.
```

This may identify:

```
overtraining
undertraining
unstable DPO
```

---

# 86. Overtraining Detection

Flag:

```
training loss decreasing
```

while:

```
held-out performance decreases
```

as:

```
potential overtraining.
```

---

# 87. Preference Overfitting

Flag if:

```
DPO performance on preference-source problems increases strongly
```

while:

```
held-out performance remains unchanged or decreases.
```

This may indicate:

```
preference-data overfitting.
```

---

# 88. Mode Collapse Detection

Flag if:

```
DPO candidate diversity
```

drops significantly relative to Base.

Example threshold:

```
diversity reduction > 20%
```

Make configurable.

---

# 89. Regression Detection

Flag if:

```
DPO loses more problems than it gains.
```

Example:

```
DPO wins = 20
DPO losses = 35
```

This should produce:

```
regression_warning = true
```

---

# 90. Iteration Decision

The analysis must produce one of:

```
continue
refine_data
tune_training
investigate_regression
insufficient_evidence
```

---

# 91. Continue

Use when:

```
DPO improves
no major regressions
dataset appears healthy
```

---

# 92. Refine Data

Use when:

```
strong evidence of coverage/data-quality gaps.
```

---

# 93. Tune Training

Use only when:

```
dataset appears adequate
```

but:

```
training behavior suggests optimization issues.
```

---

# 94. Investigate Regression

Use when:

```
DPO introduces substantial regressions.
```

---

# 95. Insufficient Evidence

Use when:

```
benchmark too small
```

or:

```
evaluation incomplete
```

or:

```
confidence intervals too wide.
```

---

# 96. Final Analysis Summary

Create:

```
summary.json
```

containing:

```
evaluation_run_id
training_run_id
benchmark_version
base_pass_at_1
dpo_pass_at_1
delta_pass_at_1
base_pass_at_5
dpo_pass_at_5
delta_pass_at_5
dpo_wins
dpo_losses
dpo_ties
dominant_error_categories
dominant_regressions
dominant_improvements
data_gaps
recommendation
recommendation_confidence
```

---

# 97. Markdown Report

Create:

```
reports/analysis.md
```

Sections:

```
Executive Summary
Experiment Context
Benchmark
Overall Results
Error Analysis
Improvements
Regressions
Category Analysis
Difficulty Analysis
Preference Data Analysis
Diversity Analysis
Data Gaps
Recommendations
Proposed Next Experiment
Limitations
```

---

# 98. Improvement Report

Create:

```
reports/improvements.md
```

Include examples such as:

```text
Problem:
    Binary tree traversal

Base:
    4/8 tests

DPO:
    8/8 tests

Improvement:
    complete
```

---

# 99. Regression Report

Create:

```
reports/regressions.md
```

Example:

```text
Problem:
    Sliding window maximum

Base:
    10/10

DPO:
    6/10

Regression:
    partial

Likely failure:
    boundary handling
```

The phrase:

```
"Likely failure"
```

must only be used if supported by the analysis.

---

# 100. Data Gap Report

Create:

```
reports/data_gaps.md
```

Example:

```text
Potential gap:

Dynamic Programming

Benchmark representation:
    22%

Preference dataset representation:
    5%

DPO pass@1:
    24%

Recommendation:
    increase DP training examples
```

---

# 101. Next Experiment Report

Create:

```
recommendations/next_experiment.md
```

It must specify:

```
objective
hypothesis
proposed change
expected outcome
risks
evaluation plan
```

---

# 102. Hypothesis Format

Example:

```text
Hypothesis:

Increasing edge-case-heavy Python preference examples
will improve DPO performance on medium-difficulty
algorithmic problems.

Evidence:

Current benchmark shows high failure rates on boundary
conditions.

Experiment:

Generate 500 new edge-case-focused problems and train
with strict preferences.

Success criterion:

+2 percentage points pass@1 without >2 pp regression
in pass@5 or timeout rate.
```

---

# 103. Experiment Discipline

Every recommendation must have:

```
hypothesis
```

not simply:

```
change X
```

This keeps the project scientifically interpretable.

---

# 104. No Data Leakage

Any newly generated training data derived from Step 11 must not contain:

```
held-out benchmark solutions
held-out benchmark prompts
held-out benchmark tests
```

unless the benchmark is explicitly retired and replaced.

---

# 105. Benchmark Retirement

If the benchmark must be used for training for diagnostic reasons:

```
mark benchmark as:

retired
```

and create:

```
new held-out benchmark
```

before claiming a new generalization result.

---

# 106. CLI

Add:

```
python -m python_dpo analyze \
    --evaluation-run-id EVAL_RUN_ID
```

---

# 107. Error Analysis CLI

Support:

```
python -m python_dpo analyze errors \
    --evaluation-run-id EVAL_RUN_ID
```

---

# 108. Data Gap CLI

Support:

```
python -m python_dpo analyze data-gaps \
    --evaluation-run-id EVAL_RUN_ID \
    --preference-run-id PREF_RUN_ID
```

---

# 109. Recommendation CLI

Support:

```
python -m python_dpo analyze recommend \
    --evaluation-run-id EVAL_RUN_ID
```

---

# 110. Generate Refinement Plan

Support:

```
python -m python_dpo analyze refine \
    --evaluation-run-id EVAL_RUN_ID
```

Output:

```
next_experiment.yaml
```

Do not automatically train.

---

# 111. Compare Experiments

Support:

```
python -m python_dpo analyze compare \
    --evaluation-runs RUN_A,RUN_B,RUN_C
```

---

# 112. Smoke Analysis

Support:

```
--smoke-test
```

The smoke test should use:

```
2–5 problems
```

and verify:

* error classification
* improvement detection
* regression detection
* report generation
* recommendation generation

---

# 113. Unit Tests

Test:

### Error classification

```
syntax error → syntax_error

TypeError → runtime_error/TypeError

timeout → timeout
```

### Improvement

```
Base 0/10
DPO 10/10
```

→ complete improvement

### Regression

```
Base 10/10
DPO 0/10
```

→ complete regression

### Partial improvement

```
Base 3/10
DPO 7/10
```

→ partial improvement

### Unchanged

```
Base 5/10
DPO 5/10
```

→ unchanged

---

# 114. Diversity Tests

Test:

```
10 identical candidates
```

→ diversity = 0.1

Test:

```
10 unique candidates
```

→ diversity = 1.0

---

# 115. Coverage Tests

Given:

```
training category = 5%

evaluation category = 25%
```

calculate:

```
coverage_ratio = 0.2
```

Classify:

```
underrepresented
```

---

# 116. Recommendation Tests

Given:

```
high error rate
```

and:

```
low training coverage
```

the recommendation engine should produce:

```
add_data
```

---

# 117. Leakage Tests

Ensure a held-out problem cannot appear in:

```
refined_preferences.jsonl
```

---

# 118. Integration Test

Create a synthetic Step 10 evaluation:

```text
Problems:
    20

Base:
    8 solved

DPO:
    11 solved

Errors:
    recursion
    DP
    edge cases
```

Create preference distribution:

```text
recursion:
    2%

DP:
    5%

edge cases:
    8%
```

The analysis should identify these as potential data gaps if the benchmark contains significantly more of these categories.

---

# 119. Acceptance Criteria

Step 11 is complete only when:

* [ ] Evaluation results can be loaded.
* [ ] Experiment lineage is preserved.
* [ ] Error taxonomy exists.
* [ ] Base error profile exists.
* [ ] DPO error profile exists.
* [ ] Improvements are identified.
* [ ] Regressions are identified.
* [ ] Partial improvements are identified.
* [ ] Partial regressions are identified.
* [ ] Error categories are aggregated.
* [ ] Test-level failures are analyzed.
* [ ] Candidate diversity is measured.
* [ ] Preference coverage is analyzed.
* [ ] Category gaps are identified.
* [ ] Difficulty gaps are identified.
* [ ] Preference quality is analyzed.
* [ ] Hard examples can be identified.
* [ ] Regression examples can be identified.
* [ ] New training examples do not contain held-out benchmark problems.
* [ ] Recommendations include evidence.
* [ ] Recommendations include hypotheses.
* [ ] Next experiment plan can be generated.
* [ ] Original datasets are never overwritten.
* [ ] Experiment lineage is versioned.
* [ ] Reports are generated.
* [ ] CLI commands exist.
* [ ] Unit tests pass.
* [ ] Integration tests pass.
* [ ] No automatic retraining occurs.

---

# 120. Verification Procedure

Run:

```
python -m python_dpo analyze \
    --evaluation-run-id EVAL_RUN_ID
```

Then:

```
python -m python_dpo analyze errors \
    --evaluation-run-id EVAL_RUN_ID
```

Then:

```
python -m python_dpo analyze data-gaps \
    --evaluation-run-id EVAL_RUN_ID \
    --preference-run-id PREF_RUN_ID
```

Then:

```
python -m python_dpo analyze recommend \
    --evaluation-run-id EVAL_RUN_ID
```

Finally:

```
python -m python_dpo analyze refine \
    --evaluation-run-id EVAL_RUN_ID
```

Verify:

```
next_experiment.yaml
```

was generated but NOT executed.

---

# 121. Expected Artifacts

After Step 11:

```
data/
└── analysis/
    └── runs/
        └── <analysis_run_id>/
            ├── manifest.json
            ├── config.yaml
            ├── summary.json
            │
            ├── classifications/
            │   ├── base_errors.jsonl
            │   └── dpo_errors.jsonl
            │
            ├── improvements/
            │   └── improvements.jsonl
            │
            ├── regressions/
            │   └── regressions.jsonl
            │
            ├── data_gaps/
            │   ├── category_gaps.json
            │   └── difficulty_gaps.json
            │
            ├── refined_dataset/
            │   ├── hard_examples.jsonl
            │   ├── regression_examples.jsonl
            │   └── successful_dpo_examples.jsonl
            │
            ├── recommendations/
            │   ├── next_experiment.yaml
            │   └── next_experiment.md
            │
            ├── reports/
            │   ├── analysis.md
            │   ├── improvements.md
            │   ├── regressions.md
            │   └── data_gaps.md
            │
            └── logs/
                └── analysis.log
```

---

# 122. Example Final Analysis

The system should be capable of producing:

```text
DPO Iteration Analysis
======================

Benchmark:
    Python Eval v1

Base pass@1:
    42.0%

DPO pass@1:
    48.0%

Improvement:
    +6.0 percentage points

DPO wins:
    83

DPO losses:
    50

Ties:
    367
```

Then:

```text
Major DPO improvements:

1. Recursion
   Base: 31%
   DPO: 45%

2. Edge-case handling
   Base: 38%
   DPO: 52%

Major regressions:

1. Dynamic programming
   Base: 41%
   DPO: 37%

2. File processing
   Base: 65%
   DPO: 61%
```

Then:

```text
Potential data gap:

Dynamic programming

Benchmark:
    24%

Preference dataset:
    7%

Recommendation:
    Generate additional DP problems.

Confidence:
    Medium
```

---

# 123. Recommended Next Experiment

The system should produce a structured hypothesis:

```yaml
hypothesis:
  id: H002
  description: >
    Increasing dynamic-programming and edge-case-heavy
    preference examples will improve DPO performance
    on medium-difficulty Python problems.

evidence:
  dpo_pass_at_1: 0.48
  dp_pass_at_1: 0.37
  dp_training_share: 0.07
  dp_benchmark_share: 0.24

proposed_change:
  additional_problems: 500
  candidates_per_problem: 8
  focus:
    - dynamic_programming
    - edge_cases

training:
  preference_policy: strict
  beta: 0.1

success_criteria:
  pass_at_1_delta: ">= 0.02"
```

This becomes the input to the next iteration.

---

# 124. What Step 11 Produces

Step 11 transforms:

```
Evaluation Results
```

into:

```
Understanding
    +
Error Patterns
    +
Data Gaps
    +
Experimental Hypothesis
    +
Next Training Plan
```

The overall loop becomes:

```text
                    ┌───────────────────────┐
                    │    Python Problems    │
                    └───────────┬───────────┘
                                │
                                ▼
                    Candidate Generation
                                │
                                ▼
                         pytest Evaluation
                                │
                                ▼
                       Preference Creation
                                │
                                ▼
                           DPO / QLoRA
                                │
                                ▼
                       Held-out Evaluation
                                │
                                ▼
                        ┌───────────────┐
                        │ Step 11       │
                        │ Error         │
                        │ Analysis      │
                        └───────┬───────┘
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
             Improvements   Regressions    Data Gaps
                 │              │              │
                 └──────────────┼──────────────┘
                                ▼
                        Next Experiment
                                │
                                ▼
                        Generate New Data
                                │
                                ▼
                           DPO Again
```

---

# 125. Final Implementation Report

After implementation, report:

1. Analysis run ID.
2. Evaluation run ID.
3. Training run ID.
4. Preference run ID.
5. Benchmark version.
6. Base pass@1.
7. DPO pass@1.
8. Base pass@5.
9. DPO pass@5.
10. Base pass@10.
11. DPO pass@10.
12. DPO wins.
13. DPO losses.
14. DPO ties.
15. Major error categories.
16. Major improvements.
17. Major regressions.
18. Partial improvements.
19. Partial regressions.
20. Category gaps.
21. Difficulty gaps.
22. Preference-density analysis.
23. Candidate-diversity analysis.
24. Potential mode collapse.
25. Potential preference noise.
26. Hard examples identified.
27. Regression examples identified.
28. Recommended next experiment.
29. Recommendation confidence.
30. Hypothesis.
31. Files created/modified.
32. Dependencies added.
33. Test results.
34. Deviations from specification.
35. Known limitations.

Do NOT automatically execute the next training run.
