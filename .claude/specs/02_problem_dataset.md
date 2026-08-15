# Python DPO Data Generation Pipeline

## Stage 2 — Python Problem Dataset and Ground Truth

**Specification Version:** 1.0
**Status:** Implementation Specification
**Stage:** 2 of 12
**Depends On:** Stage 1 — Project Skeleton

---

## 1. Objective

Implement the problem-data and ground-truth layer of the Python DPO preference-data generation pipeline.

This stage must create a reliable collection of Python programming problems where every problem has:

1. A well-defined programming task.
2. A Python function signature.
3. A trusted reference implementation.
4. Executable test cases.
5. Metadata describing category and difficulty.
6. A machine-readable representation.
7. Validation ensuring that the reference implementation passes all tests.

The output of this stage will be consumed by the candidate-generation stage.

The candidate generator must NOT be implemented in this stage.

---

# 2. Scope

Implement only:

* Problem schema
* Test-case schema
* Reference-solution schema
* Problem validation
* JSONL persistence
* Problem loading
* 10 manually curated Python problems
* Reference-solution validation
* Test execution against reference solutions
* Dataset integrity checks
* Unit tests
* Integration tests

Do NOT implement:

* Qwen inference
* Candidate generation
* Docker sandbox
* Generated-code execution
* DPO
* QLoRA
* TRL
* Hugging Face model loading
* LLM evaluation
* Benchmarking
* Ruff/mypy evaluation
* Dataset downloading from external sources

Those belong to later stages.

---

# 3. Design Principle

The problem dataset is the ground truth layer of the system.

The architecture is:

```
Problem
   |
   +---- Prompt
   |
   +---- Function Signature
   |
   +---- Reference Solution
   |
   +---- Test Cases
   |
   +---- Metadata
   |
   v
Validated Problem
   |
   v
Candidate Generation
   |
   v
Candidate Evaluation
```

A problem must NOT be considered valid unless its reference solution passes all of its tests.

---

# 4. Data Model

Implement strongly typed Python models.

Use either:

* Python dataclasses with explicit validation, or
* Pydantic models.

Prefer Pydantic if it is already consistent with the Stage 1 architecture.

Do not introduce a large framework.

---

# 5. Problem Model

Create a model equivalent to:

```
Problem
```

Required fields:

```
id
prompt
signature
category
difficulty
reference_solution
tests
```

Optional fields:

```
description
tags
source
metadata
```

Example:

```
{
    "id": "p001",
    "prompt": "Return the sum of all even numbers in a list.",
    "signature": "def sum_even(numbers):",
    "category": "lists",
    "difficulty": "easy",
    "reference_solution": "...",
    "tests": [...]
}
```

---

# 6. Problem ID

Problem IDs must be:

* unique
* stable
* human-readable
* suitable for filenames and dataset joins

Use a convention such as:

```
p001
p002
p003
```

Do not use random UUIDs for the primary problem ID.

The problem ID will later be used to join:

```
problems
   |
   +---- candidates
   |
   +---- evaluations
   |
   +---- preferences
```

---

# 7. Categories

The initial dataset must contain the following categories.

Use these exact category names:

```
lists
dictionaries
strings
sets
sorting
recursion
edge_cases
exceptions
generators
async
```

One problem should correspond to each category.

The category should describe the primary Python concept being tested.

---

# 8. Difficulty

Support:

```
easy
medium
hard
```

The initial dataset should contain:

```
5 easy
4 medium
1 hard
```

The difficulty should reflect the programming complexity rather than the amount of text in the prompt.

---

# 9. TestCase Model

Create a strongly typed test-case representation.

A test case must contain enough information to execute the target function.

Recommended fields:

```
id
input
expected
```

Example:

```
{
    "id": "t001",
    "input": "[1, 2, 3, 4]",
    "expected": "6"
}
```

The exact representation may use structured Python values rather than strings if the implementation determines that this is safer.

Prefer structured representations over serialized Python expressions where practical.

For example, prefer:

```
{
    "input": {
        "numbers": [1, 2, 3, 4]
    },
    "expected": 6
}
```

over:

```
{
    "input": "[1, 2, 3, 4]",
    "expected": "6"
}
```

However, the final representation must remain simple enough to serialize reliably to JSONL.

---

# 10. Function Invocation

The problem schema must explicitly identify the function that candidates are expected to implement.

The signature field should contain the expected function declaration.

Example:

```
def sum_even(numbers):
```

The evaluator implemented in a later stage must be able to determine how to invoke the candidate function from the problem definition.

Do not require generated candidates to contain a complete executable Python program.

The expected candidate format is:

```
def function_name(...):
    ...
```

The candidate generator in Stage 3 will be responsible for producing this implementation.

---

# 11. Reference Solutions

Every problem must contain a reference implementation.

Reference implementations must:

* be valid Python
* implement the requested function
* follow the specified function signature
* be deterministic
* not use network access
* not read or write files
* not depend on external services
* not depend on environment variables
* not require third-party packages unless explicitly justified
* pass every test

Reference solutions are trusted code.

They are NOT generated by Qwen in this stage.

They may be manually authored.

---

# 12. Reference Solution Quality

Reference implementations should prioritize:

1. Correctness
2. Clarity
3. Determinism
4. Simplicity

Do not intentionally make the reference implementation inefficient merely to create a future preference example.

Performance-based preference generation will be introduced later.

The reference implementation is the correctness oracle, not the preferred DPO answer.

---

# 13. Initial Ten Problems

Create exactly 10 initial problems.

Use the following conceptual problem definitions.

### P001 — Lists

Category:

```
lists
```

Task:

Return the sum of all even integers in a list.

Required behavior:

* empty list returns 0
* negative numbers must work
* duplicate values must work

---

### P002 — Dictionaries

Category:

```
dictionaries
```

Task:

Return the key whose value occurs most frequently in a dictionary.

The problem must clearly define tie behavior.

Include tests for:

* normal case
* one key
* ties
* empty dictionary

---

### P003 — Strings

Category:

```
strings
```

Task:

Return the first non-repeating character in a string.

Return `None` if every character occurs more than once.

Include tests for:

* normal strings
* repeated characters
* empty string
* one-character string
* all characters repeated

---

### P004 — Sets

Category:

```
sets
```

Task:

Given two collections, return the elements that occur in both collections without duplicates.

Define the expected output ordering explicitly.

Do not leave ordering ambiguous.

Include:

* normal case
* duplicates
* empty collections
* no intersection

---

### P005 — Sorting

Category:

```
sorting
```

Task:

Return the `k` largest values from a list.

Define:

* whether duplicates are retained
* expected ordering
* behavior when k is zero
* behavior when k exceeds the list size

Include appropriate edge cases.

---

### P006 — Recursion

Category:

```
recursion
```

Task:

Implement a recursive solution for a simple problem such as computing the factorial of a non-negative integer.

Define behavior for:

* 0
* 1
* normal values
* invalid negative input

The expected exception behavior must be explicit.

---

### P007 — Edge Cases

Category:

```
edge_cases
```

Task:

Implement a function that safely retrieves an element from a list by index and returns a specified fallback when the index is invalid.

Tests must emphasize:

* empty list
* first element
* last element
* negative index
* index beyond the list
* non-integer index if applicable

The expected semantics must be clearly defined.

---

### P008 — Exceptions

Category:

```
exceptions
```

Task:

Implement a function that converts a string to an integer and returns a default value when conversion fails.

Define expected behavior for:

* valid integer
* negative integer
* whitespace
* invalid text
* empty string
* None if the function accepts it

The exception behavior must be explicit.

---

### P009 — Generators

Category:

```
generators
```

Task:

Implement a generator that yields chunks of a sequence of a specified size.

Tests must cover:

* normal input
* chunk size of one
* chunk size larger than input
* empty input
* invalid chunk size

The expected generator behavior must be explicitly defined.

---

### P010 — Async

Category:

```
async
```

Task:

Implement an asynchronous function that concurrently obtains results from multiple asynchronous operations and returns the results in the required order.

The problem must use a deterministic test setup.

Do not use real network calls.

The tests should use local async functions or controlled delays.

The problem should test correct use of:

```
async
await
asyncio.gather
```

or an equivalent correct async mechanism.

---

# 14. Important Requirement for P002

The dictionary problem must not contain ambiguous semantics.

For example, if multiple keys have the same maximum frequency, explicitly define one of:

* return all tied keys
* return the first key according to insertion order
* return the lexicographically smallest key

Choose one and document it in the problem prompt.

Do not leave the behavior to the implementation.

---

# 15. Important Requirement for P004

Set operations do not inherently guarantee the desired output order.

Therefore, explicitly define ordering.

For example:

```
Return the common elements in the order in which they first
appear in the first input collection.
```

The reference implementation and tests must follow that rule.

---

# 16. Important Requirement for P005

The "k largest" problem must explicitly define duplicate behavior.

For example:

```
Return the k largest elements, preserving duplicates, in
descending order.
```

Then:

```
[5, 1, 5, 3], k=3
```

could produce:

```
[5, 5, 3]
```

The exact semantics should be documented in the problem.

---

# 17. Important Requirement for P006

Define invalid-input behavior explicitly.

For example:

```
factorial(-1)
```

must raise:

```
ValueError
```

Do not allow the reference implementation and tests to make an implicit decision.

---

# 18. Important Requirement for P009

Define the return type.

For example:

```
list(chunk_generator([1,2,3,4,5], 2))
```

must produce:

```
[[1,2], [3,4], [5]]
```

The generator must not return all chunks as a list directly.

The test suite must verify generator behavior.

---

# 19. Important Requirement for P010

Do not use:

* internet
* external APIs
* real HTTP requests
* external services

The asynchronous behavior must be completely deterministic.

Use local coroutines such as:

```
async def simulated_operation(value, delay):
    ...
```

The tests must verify the resulting values and ordering.

---

# 20. Test Design

Each problem must have at least 5 tests.

Prefer 7–10 tests where meaningful.

Tests should include a mixture of:

### Normal cases

Typical valid inputs.

### Boundary cases

Minimum valid input.

### Empty cases

Where the function accepts collections or strings.

### Degenerate cases

Repeated values, duplicate values, etc.

### Invalid cases

Where the specification explicitly defines invalid-input behavior.

---

# 21. Test Quality Principle

Tests must be capable of distinguishing common incorrect implementations.

Do not create tests merely to achieve a high test count.

For each problem, ask:

> What is the most likely mistake a generated Python solution could make?

Then create a test that catches that mistake.

Examples:

### Off-by-one

Add a test around the first/last index.

### Empty input

Add an empty input.

### Duplicate handling

Add repeated values.

### Exception handling

Add invalid input.

### Ordering

Add a case where multiple output orders are possible.

### Async ordering

Use operations completing at different times.

---

# 22. Reference Validation

Create a validator that checks every problem.

Conceptually:

```
for problem in problems:
    validate_schema(problem)
    validate_reference_solution(problem)
    run_all_tests(problem)
```

A problem is valid only when:

```
schema_valid == True
AND
reference_compiles == True
AND
all_tests_pass == True
```

---

# 23. Reference Validation Execution

Because reference solutions are trusted manually authored code, they may be executed directly during this stage if necessary.

However, isolate the execution mechanism so that the Stage 3 candidate evaluator can later replace it with the Docker sandbox.

Do not build the Docker sandbox in Stage 2.

The architecture should make it possible to change:

```
ReferenceExecutor
```

later without changing:

```
ProblemValidator
```

---

# 24. Reference Execution API

Create an abstraction similar to:

```
class ReferenceExecutor:

    def run(
        self,
        problem: Problem,
        test_case: TestCase
    ) -> TestResult:
        ...
```

The exact implementation may differ.

The important requirement is that validation code should not be tightly coupled to the execution mechanism.

---

# 25. Test Result Model

Create a structured test result.

Recommended fields:

```
test_id
passed
actual
expected
error_type
error_message
```

Example:

```
{
    "test_id": "p001_t003",
    "passed": true,
    "actual": 6,
    "expected": 6,
    "error_type": null,
    "error_message": null
}
```

For failures:

```
{
    "test_id": "p001_t004",
    "passed": false,
    "actual": null,
    "expected": 10,
    "error_type": "AssertionError",
    "error_message": "..."
}
```

---

# 26. JSONL Dataset

Persist the validated problems as:

```
data/problems/problems.jsonl
```

There must be exactly one JSON object per line.

Do not pretty-print multiple JSON objects across multiple lines.

The resulting file must be loadable line by line.

---

# 27. Dataset Loader

Create a loader such as:

```
load_problems(path) -> list[Problem]
```

Requirements:

* parse JSONL
* validate every record
* reject malformed records
* reject duplicate IDs
* provide useful error messages
* preserve problem order

Do not silently skip invalid records.

---

# 28. Dataset Writer

Create a writer such as:

```
save_problems(problems, path)
```

Requirements:

* create parent directories if necessary
* write valid JSONL
* use UTF-8
* produce deterministic output
* preserve problem order

---

# 29. Dataset Integrity Validation

Implement:

```
validate_dataset(problems)
```

It must check:

* exactly 10 problems
* unique IDs
* valid categories
* valid difficulty values
* non-empty prompts
* non-empty signatures
* non-empty reference solutions
* at least 5 tests/problem
* unique test IDs within each problem
* reference solution passes all tests

---

# 30. CLI

Extend the CLI created in Stage 1.

Add:

```
python -m python_dpo problems validate
```

and:

```
python -m python_dpo problems build
```

The exact CLI framework may follow the Stage 1 implementation.

### `problems build`

Must:

1. Load the manually defined problem specifications.
2. Validate them.
3. Validate reference solutions.
4. Run all tests.
5. Write:

   ```
   data/problems/problems.jsonl
   ```

### `problems validate`

Must:

1. Load `data/problems/problems.jsonl`.
2. Validate schema.
3. Validate uniqueness.
4. Execute reference tests.
5. Report failures.

Do not make `problems validate` silently modify the dataset.

---

# 31. Validation Output

A successful validation should produce a summary similar to:

```
Problems:              10
Valid:                 10
Invalid:                0

Categories:
  lists:                1
  dictionaries:         1
  strings:              1
  sets:                 1
  sorting:              1
  recursion:            1
  edge_cases:           1
  exceptions:           1
  generators:           1
  async:                1

Reference tests:
  Total:                75
  Passed:               75
  Failed:                0

Dataset validation: PASS
```

The exact test count may differ depending on the number of tests selected.

---

# 32. Unit Tests

Create tests covering:

### Problem schema

* valid problem
* missing required field
* invalid category
* invalid difficulty
* empty prompt
* empty signature
* empty reference solution

### Test cases

* valid test
* malformed test
* duplicate test ID

### Dataset

* duplicate problem ID
* invalid problem
* correct problem count
* missing category

### Serialization

* write problem
* load problem
* round-trip serialization

---

# 33. Integration Tests

Create an integration test that:

1. Builds the 10 problems.
2. Validates every reference solution.
3. Runs every reference test.
4. Writes the JSONL dataset.
5. Loads the JSONL dataset.
6. Validates it again.

The complete round trip must succeed.

Conceptually:

```
source problems
      ↓
   validate
      ↓
   reference
      ↓
   test suite
      ↓
   JSONL
      ↓
    reload
      ↓
   validate again
      ↓
      PASS
```

---

# 34. Determinism

The dataset creation process must be deterministic.

Given the same source problem definitions:

```
build → problems.jsonl
```

must produce the same logical dataset.

Do not use:

* random problem IDs
* random test generation
* timestamps inside problem records

unless explicitly required.

---

# 35. Data Version

Add a dataset version.

Example:

```
"dataset_version": "0.1.0"
```

This should allow future versions of the problem dataset to coexist.

Do not use the software package version as a substitute for dataset versioning.

---

# 36. Source Metadata

For these manually curated problems:

```
source = "manual"
```

Do not claim that the problems originate from HumanEval, MBPP, or another dataset.

External datasets may be incorporated in a later stage.

---

# 37. No Data Leakage Yet

Do not create train/validation/test splits in Stage 2.

The dataset is still a development dataset.

Splitting will be implemented after candidate generation and preference creation.

---

# 38. No LLM Generation Yet

The reference solutions and problems must be deterministic and manually specified.

Do not ask an LLM to:

* create problems
* create reference solutions
* create tests

during this stage.

The objective is to establish a trustworthy ground-truth layer before introducing generated data.

---

# 39. Documentation

Update `README.md` with a new section:

## Stage 2 — Problem Dataset

Document:

* purpose
* problem schema
* categories
* difficulty levels
* reference solution concept
* test-case concept
* JSONL location
* CLI commands
* validation procedure

Example:

```
python -m python_dpo problems build

python -m python_dpo problems validate
```

Do not document candidate generation as implemented.

---

# 40. Acceptance Criteria

Stage 2 is complete only when:

* [ ] Exactly 10 problems exist.
* [ ] All 10 required categories are represented.
* [ ] Difficulty distribution is 5 easy / 4 medium / 1 hard.
* [ ] Every problem has a unique ID.
* [ ] Every problem has a clear prompt.
* [ ] Every problem has a function signature.
* [ ] Every problem has a reference solution.
* [ ] Every problem has at least 5 executable tests.
* [ ] All reference solutions pass all tests.
* [ ] Dataset can be serialized to JSONL.
* [ ] Dataset can be loaded from JSONL.
* [ ] Dataset validation detects malformed records.
* [ ] Duplicate problem IDs are detected.
* [ ] Duplicate test IDs are detected.
* [ ] CLI supports `problems build`.
* [ ] CLI supports `problems validate`.
* [ ] Unit tests pass.
* [ ] Integration tests pass.
* [ ] No Qwen model is loaded.
* [ ] No generated candidate code is executed.
* [ ] No Docker sandbox is implemented.
* [ ] No DPO functionality is implemented.

---

# 41. Verification Commands

Run:

```
pytest -q
```

Then:

```
python -m python_dpo problems build
```

Then:

```
python -m python_dpo problems validate
```

Then:

```
wc -l data/problems/problems.jsonl
```

Expected:

```
10
```

Finally:

```
python -m python_dpo problems validate
```

The validation result must indicate:

```
Dataset validation: PASS
```

---

# 42. Final Implementation Report

After completing Stage 2, report:

1. Files created.
2. Files modified.
3. Problem categories implemented.
4. Number of problems.
5. Number of tests per problem.
6. Total number of tests.
7. Reference-solution pass rate.
8. CLI commands implemented.
9. Test-suite result.
10. Any deviations from this specification.
11. Any design decisions that should be reviewed before Stage 3.

Do NOT automatically proceed to Stage 3.

Wait for explicit approval before implementing Qwen candidate generation.
