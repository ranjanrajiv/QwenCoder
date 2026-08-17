# Python DPO Data Generation Pipeline

## Step 7 — Candidate Evaluation, Scoring and Ranking

**Specification Version:** 1.0
**Status:** Implementation Specification
**Step:** 7 of 12
**Depends On:** Step 1 — Project Skeleton
**Depends On:** Step 2 — Python Problem Dataset
**Depends On:** Step 3 — Qwen Candidate Generator
**Depends On:** Step 4 — Candidate Persistence
**Depends On:** Step 5 — Docker Sandbox
**Depends On:** Step 6 — Candidate Test Executor

---

# 1. Objective

Implement the candidate-quality analysis layer.

Step 6 produces objective execution evidence:

```
candidate
    ↓
pytest
    ↓
tests passed
tests failed
errors
timeout
execution metadata
```

Step 7 converts that evidence into:

1. Correctness classification.
2. Test pass rate.
3. Candidate quality score.
4. Candidate ranking within each problem.
5. Candidate comparison metadata.
6. Ranking confidence/eligibility.
7. Evaluation summaries.

The output of this stage will be consumed by Step 8 to construct DPO preference pairs.

---

# 2. Core Principle

The ranking system must be primarily **objective and execution-based**.

For Python programming tasks:

```
executable correctness
    >
static heuristics
    >
subjective LLM judgment
```

Therefore the primary ranking signal must be:

```
test-case performance
```

Do not allow an LLM judge to override objective test results in this stage.

---

# 3. Scope

This stage MUST implement:

1. Candidate correctness classification.
2. Test pass-rate calculation.
3. Candidate scoring.
4. Deterministic ranking.
5. Tie-breaking.
6. Candidate eligibility classification.
7. Pairwise comparison.
8. Ranking metadata.
9. Ranking persistence.
10. Ranking statistics.
11. CLI commands.
12. Unit tests.
13. Integration tests.
14. Ranking validation.

This stage MUST NOT implement:

* DPO dataset generation.
* chosen/rejected JSONL generation.
* model fine-tuning.
* LoRA/QLoRA.
* reward model training.
* LLM-as-a-judge as a primary ranking mechanism.
* candidate regeneration.
* mutation testing.
* distributed ranking.

---

# 4. Architecture

The pipeline must be:

```
EvaluationResult
      │
      ▼
CorrectnessClassifier
      │
      ▼
CandidateScorer
      │
      ▼
CandidateComparator
      │
      ▼
CandidateRanker
      │
      ▼
RankingResult
      │
      ▼
RankingRepository
      │
      ▼
ranking.jsonl
```

---

# 5. Important Separation

Step 7 must not modify:

```
Candidate
```

or:

```
EvaluationResult
```

Those are historical artifacts.

Instead create new derived artifacts:

```
CandidateAssessment
```

and:

```
RankingResult
```

This preserves the distinction between:

```
generated artifact
execution evidence
derived ranking
```

---

# 6. Package Structure

Create:

```
src/python_dpo/ranking/
```

Suggested modules:

```
__init__.py
classifier.py
scorer.py
comparator.py
ranker.py
models.py
repository.py
statistics.py
```

Tests:

```
tests/ranking/
    __init__.py
    test_classifier.py
    test_scorer.py
    test_comparator.py
    test_ranker.py
    test_repository.py
    test_statistics.py
    test_integration.py
```

The exact module names may differ, but responsibilities must remain separated.

---

# 7. Input

The ranking stage receives:

```
EvaluationResult
```

for candidates belonging to the same:

```
problem_id
```

and preferably the same:

```
evaluation_run_id
```

The ranker must not mix candidates from unrelated problems.

---

# 8. Candidate Grouping

Candidates must be ranked independently per problem.

Example:

```
p001:
    candidate A
    candidate B
    candidate C

p002:
    candidate D
    candidate E
    candidate F
```

Candidate A must never be ranked directly against candidate D.

The ranking unit is:

```
problem_id
```

---

# 9. Correctness Classification

Create:

```
CorrectnessClassifier
```

It converts objective test results into a classification.

Minimum classifications:

```
correct
incorrect
indeterminate
```

---

# 10. Correctness Rules

A candidate is:

```
correct
```

only when:

```
tests_total > 0
```

and:

```
tests_passed == tests_total
```

and:

```
tests_failed == 0
```

and:

```
tests_error == 0
```

and:

```
tests_skipped == 0
```

and:

```
timeout == false
```

and:

```
infrastructure_error == false
```

---

# 11. Incorrect Classification

A candidate is:

```
incorrect
```

when the candidate executes sufficiently to establish that one or more required tests fail.

Examples:

```
8 tests
7 passed
1 failed
```

or:

```
8 tests
5 passed
3 failed
```

Classification:

```
incorrect
```

---

# 12. Indeterminate Classification

A candidate is:

```
indeterminate
```

when there is insufficient evidence to determine correctness.

Examples:

```
Docker failure
evaluator infrastructure failure
missing test results
evaluation corruption
incomplete evaluation
```

A timeout caused by the candidate itself is NOT indeterminate.

It is evidence of an incorrect/failed candidate.

---

# 13. Timeout Classification

A candidate that times out during evaluation must be classified:

```
incorrect
```

provided the timeout is attributable to candidate execution.

Example:

```
while True:
    pass
```

Result:

```
incorrect
```

Reason:

```
timeout
```

Do not classify candidate-caused timeout as infrastructure failure.

---

# 14. Infrastructure Failure

If:

```
Docker unavailable
```

or:

```
evaluator infrastructure failed
```

then:

```
correctness = indeterminate
```

Do not generate a preference from an indeterminate evaluation.

This is critical.

---

# 15. Test Pass Rate

Calculate:

```
pass_rate = tests_passed / tests_total
```

Example:

```
8 / 10 = 0.8
```

Store:

```
pass_rate
```

as a floating-point value.

Use a consistent precision in serialized output.

---

# 16. Basic Score

The primary candidate score must be based on test performance.

Recommended:

```
correctness_score = tests_passed / tests_total
```

Therefore:

```
10/10 → 1.0
9/10  → 0.9
5/10  → 0.5
0/10  → 0.0
```

This is intentionally simple.

Do not introduce arbitrary weighting between individual tests in Step 7.

---

# 17. Hard Correctness Signal

Store an explicit binary signal:

```
all_tests_passed
```

Values:

```
true
false
```

This must remain separate from:

```
pass_rate
```

For example:

```
pass_rate = 0.95
all_tests_passed = false
```

This distinction will be important for DPO pair generation.

---

# 18. Candidate Quality Score

Create a `CandidateScore` model.

It should contain:

```
pass_rate
all_tests_passed
correctness
score
```

Initially:

```
score = pass_rate
```

Do not add complexity merely for the sake of creating a sophisticated score.

---

# 19. Why Score Should Start Simple

The purpose of this project is to generate high-quality preference data for DPO.

The strongest initial preference signal is:

```
candidate A passes more tests than candidate B
```

rather than:

```
candidate A received an arbitrary composite score
```

Keep the initial ranking objective interpretable.

---

# 20. Secondary Signals

The architecture may record secondary signals for future use:

```
execution_time_ms
syntax_valid
generation_strategy
code_length
token_count if available
```

But these must NOT affect the primary correctness score initially.

---

# 21. Execution Time

Record:

```
duration_ms
```

but do not automatically reward faster candidates.

For example:

```
Candidate A:
    10/10
    200 ms

Candidate B:
    10/10
    100 ms
```

Both are initially:

```
score = 1.0
```

Do not automatically rank B above A solely because it is faster.

Performance optimization should be a separately designed objective.

---

# 22. Code Length

Record candidate code length if useful:

```
code_chars
code_lines
```

Do not use code length as a correctness signal.

A shorter implementation is not necessarily better.

---

# 23. Syntax

Syntax validity may be retained as metadata:

```
syntax_valid
```

but correctness should ultimately come from sandbox execution.

Do not assign a special arbitrary score such as:

```
syntax_valid = 0.5
```

The test execution result is stronger evidence.

---

# 24. Candidate Assessment Model

Create:

```
CandidateAssessment
```

Recommended fields:

```
evaluation_run_id
candidate_run_id
candidate_id
problem_id
correctness
tests_total
tests_passed
tests_failed
tests_error
tests_skipped
pass_rate
all_tests_passed
score
timeout
infrastructure_error
execution_duration_ms
created_at
```

---

# 25. Ranking Result

Create:

```
RankingResult
```

Recommended fields:

```
ranking_run_id
evaluation_run_id
problem_id
candidate_id
rank
score
correctness
pass_rate
all_tests_passed
tie_group
eligible_for_preference
created_at
```

---

# 26. Ranking Run

Ranking must have its own run ID.

Example:

```
rank_20260817_180500_a91c
```

This allows ranking logic to evolve without modifying historical evaluations.

---

# 27. Ranking Manifest

Persist:

```
manifest.json
```

containing:

```
ranking_run_id
evaluation_run_id
ranking_version
scoring_version
comparator_version
created_at
scoring_configuration
```

Example:

```
{
  "ranking_version": "v1",
  "scoring_version": "v1",
  "comparator_version": "v1"
}
```

---

# 28. Ranking Algorithm

Candidates within a problem should initially be ordered by:

```
1. correctness classification
2. pass_rate
3. deterministic tie-breaker
```

Recommended ordering:

```
correct candidates
    ↓
incorrect candidates
    ↓
indeterminate candidates
```

Within:

```
correct
```

all candidates have:

```
pass_rate = 1.0
```

Therefore they form a tie unless another explicit objective is introduced.

Within:

```
incorrect
```

rank by:

```
pass_rate descending
```

Indeterminate candidates should not participate in preference generation.

---

# 29. Important Ranking Rule

Do NOT artificially rank two fully correct candidates.

If:

```
candidate A = 10/10

candidate B = 10/10
```

then:

```
A and B are tied
```

unless an additional objective is explicitly enabled.

Do NOT choose a winner arbitrarily.

---

# 30. Tie Handling

Create:

```
tie_group
```

Example:

```
candidate A:
    rank = 1
    tie_group = "tg001"

candidate B:
    rank = 1
    tie_group = "tg001"
```

Both:

```
10/10
```

Candidate C:

```
8/10
```

becomes:

```
rank = 3
```

or another documented ranking convention.

Use a consistent ranking policy.

---

# 31. Deterministic Tie-Breaking

For candidates with exactly the same score and same correctness status, do NOT use random ordering.

Use:

```
candidate_id
```

as a deterministic final ordering only for presentation.

However, the system must preserve:

```
tied = true
```

and must not use this arbitrary ordering to create a DPO preference.

This distinction is critical.

---

# 32. Preference Eligibility

Create:

```
eligible_for_preference
```

Rules:

```
correct → true

incorrect → true

indeterminate → false
```

However, not every correct/incorrect pair should automatically become a preference.

Step 8 will apply the preference-pair rules.

---

# 33. Pairwise Comparison

Implement:

```
compare(candidate_a, candidate_b)
```

It must return something conceptually like:

```
A_BETTER
B_BETTER
TIE
INDETERMINATE
```

Rules:

### A has more passed tests

```
A_BETTER
```

### B has more passed tests

```
B_BETTER
```

### Both pass all tests

```
TIE
```

### Both have identical pass rate

```
TIE
```

### Either evaluation is indeterminate

```
INDETERMINATE
```

---

# 34. Pairwise Comparison Example

Given:

```
A = 10/10
B = 8/10
```

Return:

```
A_BETTER
```

Given:

```
A = 8/10
B = 5/10
```

Return:

```
A_BETTER
```

Given:

```
A = 10/10
B = 10/10
```

Return:

```
TIE
```

Given:

```
A = infrastructure_error
B = 8/10
```

Return:

```
INDETERMINATE
```

---

# 35. No Arbitrary Preference

Never create:

```
A chosen
B rejected
```

if:

```
A and B are tied
```

This prevents noisy DPO labels.

---

# 36. Correct vs Incorrect

A fully correct candidate should always outrank a partially correct candidate.

Example:

```
A = 10/10
B = 9/10
```

Result:

```
A > B
```

This is one of the strongest preference relationships available in the dataset.

---

# 37. Incorrect vs Incorrect

If:

```
A = 8/10
B = 5/10
```

then:

```
A > B
```

because A satisfies more explicit behavioral requirements.

This comparison may be useful for DPO.

However, Step 8 must decide whether to include partial-vs-partial pairs.

Step 7 should simply expose the objective ordering.

---

# 38. Correct vs Correct

If:

```
A = 10/10
B = 10/10
```

then:

```
TIE
```

Do not rank by:

```
code length
execution time
strategy
candidate ID
```

unless an explicit secondary objective is introduced later.

---

# 39. Indeterminate Candidates

Indeterminate candidates must be:

```
excluded from ranking
```

or:

```
placed in a separate indeterminate group
```

They must never be used as:

```
chosen
```

or:

```
rejected
```

in DPO data.

---

# 40. Ranking Statistics

Persist:

```
statistics.json
```

with:

```
problems
candidates
correct
incorrect
indeterminate
fully_correct
partially_correct
zero_test_pass
tied_candidates
preference_eligible_candidates
```

Do not calculate actual preference-pair counts yet.

---

# 41. Candidate Distribution

For each problem report:

```
total candidates
fully correct
partially correct
completely failing
indeterminate
```

Example:

```
p001:
    total: 5
    correct: 2
    partial: 2
    zero-pass: 1
    indeterminate: 0
```

---

# 42. Ranking Persistence

Create:

```
data/rankings/runs/<ranking_run_id>/
```

with:

```
manifest.json
assessments.jsonl
rankings.jsonl
statistics.json
```

Optional:

```
comparisons.jsonl
```

if pairwise comparisons are persisted.

---

# 43. Assessment Persistence

Each candidate should have exactly one assessment per ranking run.

Example:

```
{
  "candidate_id": "p001_c002",
  "problem_id": "p001",
  "correctness": "incorrect",
  "tests_total": 10,
  "tests_passed": 7,
  "tests_failed": 3,
  "pass_rate": 0.7,
  "score": 0.7,
  "all_tests_passed": false
}
```

---

# 44. Ranking Persistence

Example:

```
{
  "candidate_id": "p001_c001",
  "problem_id": "p001",
  "rank": 1,
  "score": 1.0,
  "correctness": "correct",
  "pass_rate": 1.0,
  "tie_group": "tg001",
  "eligible_for_preference": true
}
```

---

# 45. Ranking Version

Use:

```
ranking_version
```

Example:

```
"v1"
```

Any material change to:

* correctness rules
* score
* comparator
* tie handling

must increment the relevant version.

Historical ranking artifacts must remain unchanged.

---

# 46. Determinism Requirement

Given identical:

```
EvaluationResult
```

and:

```
ranking configuration
```

the ranking output must be identical.

There must be:

```
no random ranking
no LLM calls
no timestamps influencing ordering
no nondeterministic sorting
```

---

# 47. CLI

Add:

```
python -m python_dpo rank run \
    --evaluation-run-id EVAL_RUN_ID
```

Support:

```
--problem-id
--limit
--force
```

Example:

```
python -m python_dpo rank run \
    --evaluation-run-id eval_001
```

---

# 48. Rank One Problem

Support:

```
python -m python_dpo rank run \
    --evaluation-run-id eval_001 \
    --problem-id p001
```

This should rank only candidates belonging to p001.

---

# 49. Ranking Inspection

Add:

```
python -m python_dpo rankings list RANKING_RUN_ID
```

and:

```
python -m python_dpo rankings show RANKING_RUN_ID p001
```

The output should show:

```
rank
candidate
pass rate
correctness
tie group
preference eligibility
```

Example:

```
Rank  Candidate    Tests    Score    Status
1     p001_c001    10/10    1.00     correct
1     p001_c004    10/10    1.00     correct
3     p001_c002     8/10    0.80     incorrect
4     p001_c003     5/10    0.50     incorrect
5     p001_c005     0/10    0.00     incorrect
```

---

# 50. Ranking Validation

Add:

```
python -m python_dpo rankings validate RANKING_RUN_ID
```

Validate:

* every candidate belongs to the specified evaluation run
* every candidate belongs to a valid problem
* every candidate has exactly one assessment
* score matches test results
* pass rate is correct
* correctness classification is correct
* tied candidates are not artificially preferred
* indeterminate candidates are not preference eligible
* ranks are consistent

---

# 51. Score Validation

The validator must independently recalculate:

```
tests_passed / tests_total
```

and compare it with:

```
stored pass_rate
```

If inconsistent:

```
ranking validation fails.
```

Do not trust the stored score blindly.

---

# 52. Correctness Validation

The validator must independently verify:

```
all_tests_passed
```

and:

```
correctness
```

from the evaluation results.

This prevents accidental corruption of ranking metadata.

---

# 53. Repository

Create:

```
RankingRepository
```

Support:

```
save_assessment()

save_ranking()

get_assessment()

get_ranking()

list_problem_rankings()

list_all_rankings()

count()
```

Do not modify historical records.

---

# 54. Resume Ranking

Ranking should be resumable.

If ranking a large evaluation run is interrupted:

```
python -m python_dpo rank run \
    --evaluation-run-id eval_001 \
    --resume RANKING_RUN_ID
```

should continue without recomputing completed problem groups unnecessarily.

---

# 55. Force Ranking

If:

```
--force
```

is specified:

```
create a new ranking run
```

Do not modify an existing ranking run.

This allows future scoring algorithms to be compared.

---

# 56. No LLM Judge

Step 7 MUST NOT call:

* Qwen
* Claude
* GPT
* any external LLM

for ranking.

The ranking must be completely deterministic and based on Step 6 execution results.

This is deliberate.

A future optional evaluator may use an LLM for:

* code quality
* style
* explanation quality
* maintainability

but that should be a separate signal and not silently replace execution-based correctness.

---

# 57. Optional Secondary Metrics

The architecture may store but not use:

```
execution_duration_ms
code_lines
code_chars
generation_strategy
```

These can support future experiments.

Do not allow them to affect the primary ranking in v1.

---

# 58. Pairwise Matrix

Optionally support generating an in-memory pairwise comparison matrix:

```
A vs B
A vs C
A vs D
B vs C
...
```

Do not persist all pairwise comparisons unless useful.

For:

```
N = 5
```

there are only:

```
N(N-1)/2 = 10
```

pairwise comparisons per problem.

This is small enough for the initial implementation.

---

# 59. Transitivity

The ranking should normally be transitive because it is based on:

```
pass_rate
```

Example:

```
A = 10/10
B = 8/10
C = 5/10
```

Then:

```
A > B
B > C
A > C
```

The implementation should verify this property in tests.

---

# 60. Partial Correctness

Partial correctness is explicitly retained.

Example:

```
Candidate A = 9/10
Candidate B = 7/10
```

A is objectively better according to the available tests.

Record:

```
A > B
```

Do not call A fully correct.

A remains:

```
incorrect
```

because:

```
all_tests_passed = false
```

---

# 61. Zero-Pass Candidate

A candidate with:

```
0/10
```

must be:

```
incorrect
```

with:

```
score = 0.0
```

It can potentially serve as a rejected example later.

---

# 62. All-Correct Problem

If all five candidates are:

```
10/10
```

then the problem has:

```
5 correct candidates
```

but:

```
0 preference relationships
```

from correctness alone.

This is expected.

Do NOT invent preferences.

---

# 63. All-Wrong Problem

If:

```
5 candidates
```

are all incorrect but have different pass rates:

```
9/10
8/10
6/10
4/10
1/10
```

the ranking should preserve that ordering.

Step 8 can decide which pairs are sufficiently strong for DPO.

---

# 64. Indeterminate Problem

If all candidates have infrastructure errors:

```
no preference data can be generated
```

from that evaluation run.

The ranking should record:

```
indeterminate
```

and produce zero preference-eligible candidates.

---

# 65. Important DPO Preparation Rule

The ranking layer should expose enough information for Step 8 to implement multiple preference policies.

For example:

```
Policy A:
    only fully-correct vs incorrect

Policy B:
    any higher-pass-rate vs lower-pass-rate

Policy C:
    require score margin >= 0.2

Policy D:
    correct vs partial only
```

Therefore Step 7 must not hard-code one DPO preference policy.

---

# 66. Candidate Comparison API

Expose an API conceptually similar to:

```
compare(
    assessment_a,
    assessment_b
)
```

Return:

```
ComparisonResult(
    winner=...,
    loser=...,
    relation=...,
    score_margin=...
)
```

Example:

```
A = 1.0
B = 0.7
```

Result:

```
relation = "A_BETTER"
score_margin = 0.3
```

---

# 67. Comparison Result

Recommended fields:

```
candidate_a
candidate_b
relation
score_a
score_b
score_margin
correctness_a
correctness_b
preference_eligible
```

Do not call the result:

```
chosen/rejected
```

yet.

Use neutral terminology:

```
better
worse
tie
indeterminate
```

---

# 68. Score Margin

Calculate:

```
score_margin = abs(score_a - score_b)
```

Example:

```
A = 1.0
B = 0.8
```

margin:

```
0.2
```

This will later allow Step 8 to require a minimum confidence/margin.

---

# 69. Ranking Quality Checks

Implement checks for:

### Duplicate scores

Expected.

### Duplicate code

Possible.

Do not automatically treat duplicate code as an error.

### Missing candidates

Error.

### Missing evaluation

Error or indeterminate.

### Infrastructure failures

Excluded from preference eligibility.

---

# 70. Candidate Count Consistency

The ranker must verify that every candidate being ranked has a corresponding evaluation result.

If not:

```
ranking cannot assume the missing candidate is bad.
```

The missing evaluation should be:

```
indeterminate
```

or cause the ranking run to fail validation, depending on configuration.

---

# 71. No Silent Data Loss

Never silently skip:

* missing candidates
* malformed evaluations
* invalid problems
* infrastructure failures

Every excluded artifact must have a reason.

---

# 72. Testing Requirements

Create unit tests for:

## CorrectnessClassifier

Test:

* all tests pass
* one test fails
* all tests fail
* timeout
* infrastructure error
* skipped tests
* zero tests

## Scorer

Test:

* 10/10
* 5/10
* 0/10
* fractional rates

## Comparator

Test:

* 10 vs 8
* 8 vs 5
* 10 vs 10
* indeterminate
* timeout

## Ranker

Test:

* unique scores
* ties
* all correct
* all incorrect
* mixed
* indeterminate

## Repository

Test:

* persistence
* retrieval
* validation
* duplicate prevention

---

# 73. Ranking Integration Test

Use:

```
1 problem
5 candidates
```

with simulated evaluation results:

```
candidate A = 10/10
candidate B = 8/10
candidate C = 10/10
candidate D = 5/10
candidate E = 0/10
```

Expected:

```
A = rank 1
C = rank 1
B = rank 3
D = rank 4
E = rank 5
```

A and C must belong to the same tie group.

Neither A nor C should be declared better than the other.

---

# 74. Pairwise Integration Test

For:

```
A = 10/10
B = 8/10
C = 10/10
D = 5/10
E = 0/10
```

Expected:

```
A > B
A = C
A > D
A > E

B < A
B < C
B > D
B > E

C > D
C > E

D > E
```

No pair involving an indeterminate candidate should produce a winner.

---

# 75. Reproducibility Test

Given identical evaluation artifacts and ranking configuration:

```
ranking_run_A
```

and:

```
ranking_run_B
```

must produce identical:

```
correctness classifications
scores
ranks
tie groups
pairwise comparisons
```

Run IDs and timestamps may differ.

---

# 76. Versioning Test

Change:

```
scoring_version
```

and verify:

```
new ranking run
```

is created.

Existing ranking artifacts must remain unchanged.

---

# 77. Acceptance Criteria

Step 7 is complete only when:

* [ ] Correctness classifier exists.
* [ ] Correctness is derived from objective test results.
* [ ] Infrastructure failures are classified as indeterminate.
* [ ] Candidate-caused timeout is classified as incorrect.
* [ ] Pass rate is calculated.
* [ ] Score is deterministic.
* [ ] Candidate assessments are persisted.
* [ ] Candidates are ranked independently per problem.
* [ ] Fully correct candidates are tied.
* [ ] Partial candidates are ordered by pass rate.
* [ ] Indeterminate candidates are excluded from preference eligibility.
* [ ] Pairwise comparison exists.
* [ ] Score margin is calculated.
* [ ] Ranking runs are versioned.
* [ ] Ranking is resumable.
* [ ] `--force` creates a new ranking run.
* [ ] Ranking validation exists.
* [ ] Ranking statistics exist.
* [ ] No LLM judge is used.
* [ ] No DPO pairs are generated.
* [ ] No chosen/rejected labels are persisted.
* [ ] No model fine-tuning is implemented.
* [ ] All unit tests pass.
* [ ] Integration tests pass.
* [ ] Results are deterministic.

---

# 78. Verification Procedure

Run:

```
pytest -q
```

Then identify an evaluation run:

```
python -m python_dpo evaluations list
```

Run ranking:

```
python -m python_dpo rank run \
    --evaluation-run-id EVAL_RUN_ID
```

Inspect:

```
python -m python_dpo rankings list RANKING_RUN_ID
```

Inspect one problem:

```
python -m python_dpo rankings show \
    RANKING_RUN_ID \
    p001
```

Validate:

```
python -m python_dpo rankings validate \
    RANKING_RUN_ID
```

---

# 79. Expected Output

For a problem with:

```
p001_c001 = 10/10
p001_c002 = 8/10
p001_c003 = 10/10
p001_c004 = 5/10
p001_c005 = 0/10
```

the ranking should conceptually be:

```
Rank  Candidate    Tests    Score    Correctness
-------------------------------------------------
1     p001_c001    10/10    1.00     correct
1     p001_c003    10/10    1.00     correct
3     p001_c002     8/10    0.80     incorrect
4     p001_c004     5/10    0.50     incorrect
5     p001_c005     0/10    0.00     incorrect
```

The two 10/10 candidates are tied.

---

# 80. Expected Artifacts

After Step 7:

```
data/
└── rankings/
    └── runs/
        └── <ranking_run_id>/
            ├── manifest.json
            ├── assessments.jsonl
            ├── rankings.jsonl
            └── statistics.json
```

The ranking artifact must reference the original:

```
evaluation_run_id
```

and therefore remain traceable to the original candidate-generation run.

---

# 81. What Step 7 Produces

Step 7 produces:

```
objective candidate assessments
+
deterministic rankings
+
pairwise comparison information
```

It does NOT produce:

```
chosen
rejected
DPO JSONL
preference pairs
```

Those belong to Step 8.

---

# 82. Example End-to-End State

After Steps 1–7:

```
Problem
   │
   ▼
Qwen Candidate
   │
   ▼
Candidate Persistence
   │
   ▼
Docker Sandbox
   │
   ▼
pytest Evaluation
   │
   ▼
EvaluationResult
   │
   ▼
CandidateAssessment
   │
   ▼
RankingResult
   │
   ├── Candidate A: 10/10
   ├── Candidate B: 8/10
   ├── Candidate C: 10/10
   ├── Candidate D: 5/10
   └── Candidate E: 0/10
             │
             ▼
      Ready for Step 8
```

```

---

# 83. Final Implementation Report

After implementation, report:

1. Correctness-classification rules.
2. Scoring formula.
3. Ranking algorithm.
4. Tie-handling policy.
5. Pairwise comparison policy.
6. Indeterminate handling.
7. Ranking schema.
8. Ranking-run architecture.
9. Versioning strategy.
10. CLI commands added.
11. Ranking statistics.
12. Unit-test results.
13. Integration-test results.
14. Determinism-test results.
15. Example ranking output.
16. Files created/modified.
17. Dependencies added.
18. Any deviations from this specification.
19. Known limitations.

Do NOT implement Step 8 automatically.

Wait for explicit approval before implementing DPO preference-pair generation.
```
