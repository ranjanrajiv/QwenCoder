# Python DPO Data Generation Pipeline

## Step 6 — Candidate Test Executor

**Specification Version:** 1.0
**Status:** Implementation Specification
**Step:** 6 of 12
**Depends On:** Step 1 — Project Skeleton
**Depends On:** Step 2 — Python Problem Dataset
**Depends On:** Step 3 — Qwen Candidate Generator
**Depends On:** Step 4 — Candidate Persistence
**Depends On:** Step 5 — Docker Sandbox

---

# 1. Objective

Implement the candidate test-execution layer.

The purpose of this stage is to determine whether an individual Qwen-generated Python candidate satisfies the test cases defined for its corresponding problem.

The execution flow must be:

```
Problem
   │
   ├── Reference behavior / expected outputs
   │
   └── Test cases
            │
            ▼
      Test Generator
            │
            ▼
      pytest test suite
            │
            ▼
    Docker Sandbox
            │
            ▼
     pytest execution
            │
            ▼
    Test Execution Result
            │
            ▼
    Structured Evaluation Data
```

The system must execute candidate code ONLY inside the Step 5 Docker sandbox.

---

# 2. Core Principle

The Step 5 sandbox answers:

> "Can this Python program execute safely?"

Step 6 answers:

> "What happened when this candidate was executed against the problem's tests?"

Step 6 must NOT yet answer:

> "Is this the best candidate?"

and must NOT generate:

```
chosen
rejected
```

preference pairs.

Preference generation belongs to a later stage.

---

# 3. Scope

This stage MUST implement:

1. Candidate loading.
2. Problem loading.
3. Test-case generation.
4. Temporary pytest job creation.
5. Candidate/test integration.
6. pytest execution through Docker.
7. pytest result parsing.
8. Structured test results.
9. Test-level results.
10. Candidate-level execution results.
11. Timeout/error classification.
12. Evaluation artifact persistence.
13. CLI commands.
14. Unit tests.
15. Docker integration tests.
16. End-to-end tests using the Step 5 sandbox.

This stage MUST NOT implement:

* preference generation
* DPO
* candidate ranking
* reward modeling
* LLM judging
* performance ranking
* mutation testing
* Qwen generation
* candidate regeneration
* distributed evaluation
* parallel execution

---

# 4. Architecture

Implement:

```
CandidateRepository
        │
        ▼
    Candidate
        │
        │
ProblemRepository
        │
        ▼
     Problem
        │
        ▼
   TestGenerator
        │
        ▼
┌───────────────────┐
│ pytest job        │
│                   │
│ candidate.py      │
│ test_candidate.py │
└─────────┬─────────┘
          │
          ▼
   SandboxExecutor
          │
          ▼
      Docker
          │
          ▼
       pytest
          │
          ▼
   PytestResultParser
          │
          ▼
   EvaluationResult
          │
          ▼
 EvaluationRepository
```

---

# 5. Package Structure

Create:

```
src/python_dpo/evaluation/
```

Suggested modules:

```
__init__.py
test_generator.py
executor.py
pytest_runner.py
result_parser.py
models.py
repository.py
errors.py
```

Tests:

```
tests/evaluation/
    __init__.py
    test_models.py
    test_test_generator.py
    test_result_parser.py
    test_executor.py
    test_repository.py
    test_integration.py
```

The exact module names may differ, but responsibilities must remain separated.

---

# 6. Candidate Input

The executor must accept a persisted `Candidate`.

The candidate must contain:

```
run_id
candidate_id
problem_id
code
```

The evaluator must retrieve the corresponding `Problem` using:

```
candidate.problem_id
```

Do not allow the caller to evaluate a candidate against an arbitrary unrelated problem without explicit validation.

---

# 7. Problem Input

The evaluator must load the problem from the Step 2 problem repository.

The problem provides:

```
problem_id
prompt
signature
tests
reference_solution
```

The evaluator should use the problem's declared tests.

Do NOT ask Qwen to generate tests.

---

# 8. Test Oracle

The expected outputs defined in the `Problem` test cases are the authoritative expected results.

For example:

```
TestCase:
    input = [1, 2, 3, 4]
    expected = 6
```

The candidate must produce:

```
6
```

for that input.

Do not use the candidate itself to determine the expected answer.

---

# 9. Reference Solution

The reference solution is a trusted development artifact.

The evaluation framework must NOT execute the reference solution for every candidate execution unless required for validation.

The expected results in the dataset are authoritative.

The reference solution exists primarily to:

* validate the problem dataset
* establish expected behavior
* help create/verify tests

Do not dynamically calculate expected values from the reference solution during candidate evaluation.

---

# 10. Test Case Representation

Each `TestCase` must contain enough information to generate a deterministic pytest test.

For example:

```
{
  "id": "p001_tc001",
  "input": [1, 2, 3, 4],
  "expected": 6
}
```

The test generator must transform this into a pytest test without executing Python source on the host.

---

# 11. Test Generation

Create:

```
TestGenerator
```

It must convert:

```
Problem + Candidate
```

into an isolated test job.

The job should contain at least:

```
candidate.py
test_candidate.py
```

Example:

```
/workspace/
    candidate.py
    test_candidate.py
```

---

# 12. Candidate File

The generated candidate source must be written exactly as persisted.

Do not modify the candidate's algorithm.

Do not automatically repair:

* syntax
* imports
* function names
* logic

If the candidate is invalid, the evaluator must report that fact.

---

# 13. Test File

Generate a pytest file that imports the candidate function.

For example:

```
from candidate import sum_even

def test_p001_tc001():
    result = sum_even([1, 2, 3, 4])
    assert result == 6
```

The actual generated test code must be deterministic.

---

# 14. Function Signature

The test generator must use the problem's declared function signature/function name.

For example:

```
def sum_even(values):
```

The candidate must provide the expected function.

If it does not, pytest should report an import/attribute failure.

The evaluator must not silently substitute another function.

---

# 15. Input Handling

The test generator must correctly pass JSON-compatible test inputs to Python functions.

Support at least:

* integers
* floats
* strings
* booleans
* null/None
* lists
* dictionaries
* nested combinations of these

Do not construct test inputs using unsafe string interpolation.

---

# 16. Test Data Safety

Test values originate from the trusted problem dataset.

Nevertheless, avoid generating tests through unsafe string interpolation.

Prefer generating Python literals using a safe serialization mechanism such as:

```
repr()
```

for controlled Python values, or another validated approach.

Do not use:

```
eval()
```

to reconstruct test data.

---

# 17. Expected Value Handling

Expected outputs may contain:

* integers
* floats
* strings
* lists
* tuples
* dictionaries
* sets where supported by the schema
* None
* booleans

The schema should remain deterministic.

If the dataset currently supports only JSON-compatible values, preserve that restriction.

Do not expand the schema unnecessarily in Step 6.

---

# 18. Floating-Point Tests

If the dataset contains floating-point expected values, do not automatically use exact equality.

Support an explicit comparison mode such as:

```
exact
approx
```

For example:

```
expected:
  value: 0.3
  comparison: approx
  abs_tol: 1e-9
```

Do not add approximate comparison to every test by default.

---

# 19. Test Metadata

Each generated pytest test should be traceable to the original test case.

Use pytest IDs where practical.

Example:

```
@pytest.mark.parametrize(..., ids=["p001_tc001"])
```

or:

```
def test_p001_tc001():
```

This allows pytest output to identify exactly which dataset test failed.

---

# 20. One Test Case = One Logical Test

Each problem test case should map to one logical test.

Example:

```
p001_tc001
p001_tc002
p001_tc003
p001_tc004
```

This makes failures directly traceable.

Avoid combining ten unrelated assertions into one test.

---

# 21. Test Failure Information

For each test case capture:

```
test_case_id
status
duration
error_type
error_message
```

Possible test-level statuses:

```
passed
failed
error
skipped
```

A candidate-level timeout is handled separately.

---

# 22. Candidate-Level Result

Create a typed `EvaluationResult`.

It must contain at least:

```
run_id
candidate_id
problem_id
status
tests_total
tests_passed
tests_failed
tests_error
tests_skipped
duration_ms
```

Recommended fields:

```
syntax_error
runtime_error
timeout
infrastructure_error
stdout
stderr
exit_code
sandbox_container_id
evaluation_timestamp
```

---

# 23. Candidate-Level Status

Use:

```
passed
failed
timeout
syntax_error
infrastructure_error
```

The exact classification must be deterministic.

---

# 24. Meaning of `passed`

A candidate receives:

```
status = passed
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

Do not mark a candidate as passed merely because the Python process exited with code 0.

---

# 25. Meaning of `failed`

A candidate receives:

```
status = failed
```

when pytest executes successfully but one or more tests fail.

Example:

```
tests_total = 8
tests_passed = 6
tests_failed = 2
```

Result:

```
failed
```

---

# 26. Syntax Error

A candidate containing invalid Python should be classified as:

```
syntax_error
```

if the failure can be reliably identified as a syntax error.

Do not execute candidate syntax validation on the host as a substitute for sandbox execution.

The Step 3 AST validation may already indicate syntax validity, but Step 6 must still treat the sandbox execution result as authoritative.

---

# 27. Runtime Error

If the candidate imports successfully but raises an exception during a test:

```
status = failed
```

At the test level:

```
status = error
```

Capture the exception type/message in the test result.

Do not classify candidate runtime exceptions as infrastructure errors.

---

# 28. Timeout

If the Step 5 sandbox terminates execution because of a timeout:

```
status = timeout
```

Set:

```
timeout = true
```

Do not mark the candidate as merely failed.

Timeout is a distinct failure mode.

---

# 29. Infrastructure Error

If Docker itself fails:

```
status = infrastructure_error
```

Examples:

* Docker daemon unavailable
* container cannot start
* sandbox image unavailable
* workspace creation failure
* Docker API failure

Do NOT treat infrastructure failures as candidate failures.

This distinction will be important when creating DPO preferences later.

---

# 30. Test-Level Result Model

Create:

```
TestCaseResult
```

with:

```
test_case_id
status
duration_ms
error_type
error_message
stdout
stderr
```

Optional:

```
traceback
```

The complete traceback should be retained in evaluation artifacts where practical.

---

# 31. Pytest Result Parsing

Create:

```
PytestResultParser
```

It must parse pytest output into structured results.

Do not rely only on parsing human-readable stdout if a more structured pytest reporting mechanism can be used.

Prefer a machine-readable mechanism where practical.

The parser must determine:

```
total
passed
failed
errors
skipped
```

and map failures to:

```
test_case_id
```

---

# 32. Machine-Readable Pytest Output

Prefer using pytest's structured reporting mechanisms.

Possible approaches include:

* pytest JSON report plugin
* pytest hooks
* JUnit XML
* a custom lightweight pytest plugin

Choose the simplest reliable mechanism.

Do not add a large dependency merely for parsing.

If an additional dependency is required, document why.

---

# 33. Pytest Dependency

The sandbox image must contain the required pytest version for evaluation.

Do not rely on the host's pytest installation.

The container should use a pinned pytest version.

For example:

```
pytest==<pinned-version>
```

The exact version should be selected and documented.

---

# 34. Evaluation Image

Step 5 used:

```
python:3.12-slim
```

Step 6 may require a dedicated evaluation image containing pytest.

For example:

```
python-dpo-evaluator:<version>
```

The image should contain:

* pinned Python version
* pinned pytest version
* only required dependencies

Do not install unnecessary packages.

---

# 35. Evaluation Docker Image

Create a Dockerfile such as:

```
docker/evaluator/Dockerfile
```

It should build an image containing:

```
Python
pytest
```

Do not copy the entire host project into the image unless necessary.

Candidate and test files should be supplied through the isolated workspace.

---

# 36. Network

The evaluator container MUST continue to use:

```
network_mode = none
```

Candidate tests must not access the network.

Do not relax Step 5 network isolation simply because pytest is now being used.

All dependencies required by the evaluator image must be installed when the image is built.

---

# 37. Filesystem

The evaluator must retain all Step 5 filesystem isolation properties:

* no project-directory mount
* no home-directory mount
* no Docker socket
* no credentials
* no SSH keys
* no host Python environment

Only the evaluation workspace should be exposed.

---

# 38. Evaluation Workspace

A job should look like:

```
evaluation-job/
    candidate.py
    test_candidate.py
```

Optional:

```
    pytest.ini
```

Do not include unnecessary files.

---

# 39. pytest Configuration

If needed, generate a minimal:

```
pytest.ini
```

or:

```
pyproject.toml
```

inside the evaluation workspace.

Do not expose the host project's pytest configuration.

The evaluator must be isolated from host pytest configuration.

---

# 40. Import Isolation

The generated candidate must be imported from the isolated workspace.

The test process must not accidentally import:

```
src/python_dpo/
```

or another host module.

Ensure:

```
PYTHONPATH
```

is controlled.

Do not inherit the host's Python path.

---

# 41. Test Environment

The test environment must be deterministic.

Do not inherit:

* host environment variables
* current working directory assumptions
* host locale unless explicitly required
* host PYTHONPATH

Set only the environment required by the evaluator.

---

# 42. Test Execution Command

Use a fixed command similar to:

```
pytest
    -q
    test_candidate.py
```

The exact arguments may differ.

Do not construct shell commands by concatenating candidate code.

---

# 43. No Shell Injection

Candidate source must never be interpolated into:

```
shell=True
```

commands.

Do not run:

```
sh -c ...
```

with candidate content.

Use fixed executable/argument arrays or equivalent Docker API commands.

---

# 44. Test Generation Security

Problem inputs and expected outputs must not be treated as executable source.

Do not generate tests using:

```
eval(input_string)
```

Do not use:

```
exec()
```

to create tests.

The test generator must produce ordinary Python source from trusted structured data.

---

# 45. Empty Test Suite

A problem must have at least one test.

If a problem has zero tests:

```
status = infrastructure_error
```

or:

```
invalid_problem
```

Do not mark the candidate as passed.

---

# 46. Test Case Count

The evaluator must verify:

```
actual_test_count == expected_test_count
```

where possible.

If the generated pytest file accidentally contains fewer tests than expected:

```
infrastructure_error
```

or:

```
test_generation_error
```

Do not silently accept missing tests.

---

# 47. Test Generation Validation

Before sending the job to Docker, validate the generated test file structurally.

At minimum verify:

* file exists
* expected number of tests generated
* candidate import name is correct
* test IDs exist
* no unexpected test count discrepancy

Do not execute the generated test file on the host.

---

# 48. Evaluation Persistence

Create:

```
data/evaluations/
```

Use:

```
data/evaluations/runs/<evaluation_run_id>/
```

Each evaluation run should contain:

```
manifest.json
evaluations.jsonl
test_results.jsonl
failures.jsonl
statistics.json
```

The exact structure may reuse the Step 4 run architecture.

---

# 49. Evaluation Run ID

An evaluation run should have its own identifier.

Example:

```
eval_20260817_154500_a12f
```

Record:

```
evaluation_run_id
```

in every evaluation record.

---

# 50. Evaluation Provenance

Each evaluation record must reference:

```
candidate_run_id
candidate_id
problem_id
evaluation_run_id
```

This allows:

```
evaluation
   ↓
candidate
   ↓
generation run
   ↓
model
   ↓
prompt
```

to be reconstructed.

---

# 51. Sandbox Configuration Snapshot

Persist the sandbox configuration used for the evaluation.

Include:

```
image
image_digest
Python version
pytest version
network mode
CPU limit
memory limit
timeout
PID limit
output limit
```

This is important for reproducibility.

---

# 52. Evaluation Repository

Create:

```
EvaluationRepository
```

Support:

```
save(result)

get(evaluation_run_id, candidate_id)

list(evaluation_run_id)

find_by_candidate(candidate_run_id, candidate_id)

find_by_problem(evaluation_run_id, problem_id)

count(evaluation_run_id)
```

Do not expose filesystem details outside the repository.

---

# 53. CLI

Add:

```
python -m python_dpo evaluate candidate \
    --run-id RUN_ID \
    --candidate-id p001_c001
```

This evaluates one candidate.

Also support:

```
python -m python_dpo evaluate run \
    --run-id RUN_ID
```

This evaluates all candidates in a generation run.

---

# 54. Problem Selection

Support:

```
--problem-id
```

Example:

```
python -m python_dpo evaluate run \
    --run-id RUN_ID \
    --problem-id p001
```

Only candidates belonging to that problem should be evaluated.

---

# 55. Limit

Support:

```
--limit
```

Example:

```
python -m python_dpo evaluate run \
    --run-id RUN_ID \
    --limit 5
```

This is useful for controlled testing.

---

# 56. Resume Evaluation

Evaluation must be resumable.

If:

```
10 candidates
```

exist and:

```
4 evaluations
```

have already been persisted, restarting must evaluate only the remaining six.

Use persisted evaluation records to determine completed work.

Do not rely only on in-memory state.

---

# 57. Force Evaluation

Support:

```
--force
```

Do not overwrite historical evaluation results.

Instead create a new evaluation run.

Example:

```
evaluation_run_A
       ↓
    --force
       ↓
evaluation_run_B
```

This preserves historical evaluation results.

---

# 58. Evaluation Statistics

Persist:

```
statistics.json
```

with at least:

```
candidates_requested
candidates_evaluated
passed
failed
syntax_errors
timeouts
infrastructure_errors
tests_total
tests_passed
tests_failed
tests_errors
```

Do not yet calculate:

```
reward
preference score
ranking
```

---

# 59. Example Evaluation Result

A successful candidate:

```
{
  "evaluation_run_id": "eval_001",
  "candidate_run_id": "run_001",
  "candidate_id": "p001_c001",
  "problem_id": "p001",
  "status": "passed",
  "tests_total": 8,
  "tests_passed": 8,
  "tests_failed": 0,
  "tests_error": 0,
  "tests_skipped": 0,
  "timeout": false,
  "duration_ms": 142
}
```

A failed candidate:

```
{
  "evaluation_run_id": "eval_001",
  "candidate_run_id": "run_001",
  "candidate_id": "p001_c002",
  "problem_id": "p001",
  "status": "failed",
  "tests_total": 8,
  "tests_passed": 5,
  "tests_failed": 3,
  "tests_error": 0,
  "tests_skipped": 0,
  "timeout": false,
  "duration_ms": 138
}
```

---

# 60. Test-Level Example

```
{
  "evaluation_run_id": "eval_001",
  "candidate_id": "p001_c002",
  "problem_id": "p001",
  "test_case_id": "p001_tc006",
  "status": "failed",
  "duration_ms": 3,
  "error_type": "AssertionError",
  "error_message": "Expected 10 but received 12"
}
```

The exact error format may differ.

---

# 61. Pass Rate

Calculate:

```
pass_rate = tests_passed / tests_total
```

Example:

```
5 / 8 = 0.625
```

Store this value.

However:

> Pass rate is an evaluation metric, not yet a preference score.

Do not use it to generate DPO pairs in Step 6.

---

# 62. Correctness Classification

Do NOT introduce the final:

```
correct / incorrect
```

semantic classification here if it belongs to the next evaluation/ranking stage.

The evaluator should primarily produce objective execution facts:

```
tests_passed
tests_failed
errors
timeout
etc.
```

A later stage may transform these into:

```
correct
incorrect
```

This keeps the evaluation layer reusable.

---

# 63. Candidate Syntax Metadata

Step 3 already records:

```
syntax_valid
```

Step 6 should compare this with actual sandbox execution.

If Step 3 says:

```
syntax_valid = true
```

but pytest reports a syntax error:

record the discrepancy.

Do not silently overwrite historical generation metadata.

---

# 64. Discrepancy Reporting

If candidate metadata conflicts with execution:

```
generated_syntax_valid = true
execution_result = syntax_error
```

record:

```
metadata_discrepancy = true
```

and the reason.

This is valuable for debugging the candidate-generation pipeline.

---

# 65. Test Result Parser Tests

Create parser fixtures for:

### All tests pass

```
8 passed
```

### Partial failure

```
5 passed, 3 failed
```

### Runtime error

```
1 error
```

### Syntax error

```
collection error
```

### Timeout

```
sandbox-level timeout
```

### Skipped tests

```
7 passed, 1 skipped
```

The parser must distinguish these cases.

---

# 66. Integration Test Cases

At minimum create these candidate fixtures:

### Candidate A — Correct

Should pass every test.

### Candidate B — Wrong result

Should fail one or more tests.

### Candidate C — Syntax error

Should produce syntax error.

### Candidate D — Runtime exception

Should produce test error/failure.

### Candidate E — Infinite loop

Should produce sandbox timeout.

### Candidate F — Network attempt

Should fail because network is disabled.

---

# 67. End-to-End Test

Use one existing problem from Step 2.

Generate or use a known candidate.

Run:

```
Candidate
   ↓
TestGenerator
   ↓
Docker Sandbox
   ↓
pytest
   ↓
EvaluationResult
```

Verify the final result matches the expected behavior.

---

# 68. Multiple Test Cases

Run the evaluator against a candidate that:

* passes some tests
* fails some tests

Verify:

```
tests_total
```

equals the number of problem test cases.

Verify:

```
tests_passed + tests_failed + tests_error + tests_skipped
```

equals:

```
tests_total
```

unless pytest infrastructure behavior requires a documented exception.

---

# 69. No Candidate Modification

The evaluator must never modify the candidate source.

After evaluation:

```
candidate_before == candidate_after
```

must remain true.

The persisted candidate artifact must be immutable.

---

# 70. No Problem Modification

The evaluator must not modify:

```
problems.jsonl
```

or any problem record.

Problems are inputs.

---

# 71. Evaluation Immutability

Historical evaluation results must not be modified.

If an evaluation needs to be rerun:

```
create a new evaluation run.
```

This allows comparison of:

```
evaluator version A
evaluator version B
```

later.

---

# 72. Evaluator Version

Record:

```
evaluator_version
```

in the evaluation manifest.

Example:

```
evaluator_version: "v1"
```

This should change whenever evaluation semantics change materially.

---

# 73. Test Generator Version

Record:

```
test_generator_version
```

in the evaluation manifest.

This allows historical evaluation behavior to be reconstructed.

---

# 74. Pytest Version

Record the actual pytest version used.

Example:

```
pytest_version: "8.x.x"
```

Do not rely on the host's pytest version.

---

# 75. Evaluation Environment

The evaluation environment must be reproducible.

Record:

```
Python version
pytest version
Docker image
Docker image digest
sandbox configuration
evaluator version
test-generator version
```

---

# 76. Logging

Log:

```
evaluation run started
candidate started
problem loaded
test suite generated
sandbox started
pytest completed
result parsed
evaluation persisted
```

Example:

```
INFO | Evaluating p001_c001
INFO | Generated 8 pytest tests
INFO | Sandbox execution completed
INFO | p001_c001 | 8/8 tests passed
INFO | Evaluation persisted
```

Do not log complete candidate source at INFO level.

---

# 77. Error Logging

For failures, log:

```
candidate_id
problem_id
error category
concise message
```

The full traceback should be stored in evaluation artifacts where appropriate.

Do not log secrets.

---

# 78. Security Requirements

All Step 5 security requirements remain mandatory.

Step 6 must NOT weaken:

* network isolation
* filesystem isolation
* non-root execution
* resource limits
* PID limits
* timeout
* output limits
* Docker socket isolation
* environment isolation

Adding pytest must not change these properties.

---

# 79. No Host Test Execution

Do NOT run:

```
pytest test_candidate.py
```

on the host.

The generated test suite and candidate code must execute inside Docker.

The host may:

* generate the test file
* inspect structured metadata
* persist results

but may not execute generated candidate/test code.

---

# 80. No `exec()` or `eval()`

The evaluator must not use:

```
exec()
```

or:

```
eval()
```

to execute candidate code or test input.

This is a mandatory security requirement.

---

# 81. Acceptance Criteria

Step 6 is complete only when:

* [ ] Candidate loader works.
* [ ] Problem loader works.
* [ ] Test generator exists.
* [ ] pytest test files are generated deterministically.
* [ ] Candidate code is copied unchanged into the sandbox.
* [ ] Test files execute only inside Docker.
* [ ] Evaluation uses the Step 5 sandbox.
* [ ] pytest is installed in a pinned evaluation image.
* [ ] pytest output is parsed into structured results.
* [ ] Test-level results are persisted.
* [ ] Candidate-level results are persisted.
* [ ] Pass/fail/error counts are correct.
* [ ] Timeout is distinguished from test failure.
* [ ] Infrastructure errors are distinguished from candidate failures.
* [ ] Evaluation runs are versioned.
* [ ] Evaluation is resumable.
* [ ] `--force` creates a new evaluation run.
* [ ] Evaluation statistics are persisted.
* [ ] Candidate source is never modified.
* [ ] Problem source is never modified.
* [ ] No candidate code executes on the host.
* [ ] No preference generation exists.
* [ ] No ranking exists.
* [ ] No DPO code exists.
* [ ] Security tests pass.
* [ ] End-to-end evaluation tests pass.
* [ ] All applicable tests pass.

---

# 82. Verification Procedure

Run unit tests:

```
pytest -q
```

Build the evaluation image:

```
docker build -t python-dpo-evaluator:1.0 docker/evaluator/
```

Run sandbox health:

```
python -m python_dpo sandbox health
```

Select one candidate:

```
python -m python_dpo candidates list RUN_ID
```

Evaluate it:

```
python -m python_dpo evaluate candidate \
    --run-id RUN_ID \
    --candidate-id p001_c001
```

Inspect the evaluation:

```
python -m python_dpo evaluations list EVALUATION_RUN_ID
```

Inspect statistics:

```
python -m python_dpo evaluations stats EVALUATION_RUN_ID
```

---

# 83. Mandatory Functional Test

Use one known-good candidate.

Expected:

```
tests_total > 0
tests_passed == tests_total
tests_failed == 0
tests_error == 0
status == passed
```

---

# 84. Mandatory Negative Test

Use a deliberately incorrect candidate.

Expected:

```
tests_total > 0
tests_passed < tests_total
status == failed
```

The system must identify which test cases failed.

---

# 85. Mandatory Timeout Test

Use:

```
while True:
    pass
```

Expected:

```
status == timeout
```

and:

```
container cleaned up
```

---

# 86. Mandatory Infrastructure Test

Temporarily make Docker unavailable.

Expected:

```
status == infrastructure_error
```

The candidate must NOT be classified as failed because the infrastructure was unavailable.

---

# 87. Mandatory Security Test

Use a candidate attempting:

```
import socket
socket.create_connection(("8.8.8.8", 53))
```

Expected:

```
network operation fails
```

The evaluator must continue to classify the result correctly without exposing network access.

---

# 88. Expected Artifacts

After Step 6:

```
data/
├── problems/
│   └── problems.jsonl
│
├── candidates/
│   └── runs/
│       └── <run_id>/
│           └── candidates.jsonl
│
└── evaluations/
    └── runs/
        └── <evaluation_run_id>/
            ├── manifest.json
            ├── evaluations.jsonl
            ├── test_results.jsonl
            ├── failures.jsonl
            └── statistics.json
```

---

# 89. What Step 6 Does NOT Do

The output of Step 6 is:

```
objective execution evidence
```

It is NOT:

```
chosen
rejected
reward
preference
ranking
```

For example:

```
candidate A:
    10/10 tests passed

candidate B:
    7/10 tests passed
```

Step 6 records those facts.

A later stage decides how to convert those facts into preference pairs.

---

# 90. Final Implementation Report

After implementation, report:

1. Test-generation architecture.
2. Evaluation architecture.
3. Docker evaluation image.
4. Python version.
5. pytest version.
6. Test-generation format.
7. Result-parser design.
8. Evaluation-result schema.
9. Evaluation-run architecture.
10. Resume behavior.
11. Failure classification.
12. Security properties preserved from Step 5.
13. Unit-test results.
14. Docker integration-test results.
15. End-to-end test results.
16. Files created/modified.
17. Dependencies added.
18. Any deviations from this specification.
19. Known limitations.

Do NOT implement Step 7 automatically.

Wait for explicit approval before implementing candidate evaluation/ranking and correctness classification.
