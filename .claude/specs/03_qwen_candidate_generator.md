Below is the **Step 3 specification file**. This stage introduces Qwen inference but deliberately stops **before Docker execution, pytest evaluation, ranking, and DPO dataset generation**.

Save it as:

```text
specs/03_qwen_candidate_generator.md
```

# Python DPO Data Generation Pipeline

## Step 3 — Qwen Candidate Generator

**Specification Version:** 1.1
**Status:** Implementation Specification
**Step:** 3 of 12
**Depends On:** Step 1 — Project Skeleton
**Depends On:** Step 2 — Python Problem Dataset

## Revision History

**1.1** — Resolved three internal contradictions found while planning the implementation
(see `.claude/plans/03_qwen_candidate_generator_plan.md`). No requirement was added or
removed; each amendment picks one of two readings the document already contained.

1. **§19 vs §26 — syntax errors.** §26 listed "syntax error" as a generation *failure*,
   while §19, §42, and §49 required persisting a candidate with `syntax_valid: false`.
   Obeying both would record one generation twice. Resolved in favour of §19/§42/§49:
   a syntax-invalid candidate **is** a candidate. §19 and §26 amended.
2. **§26 — model loading failure.** Listed as a per-generation failure that "must not crash
   the job", but weights that fail to load cannot be retried per candidate. Now specified as
   a run-level abort. §26 amended.
3. **§21 vs §52 — candidate ID uniqueness under `--force`.** §21 required deterministic
   `{problem_id}_c{index}` IDs; §52 required `--force` to create a *new generation run*
   while §25/§41 forbid discarding data. These are only compatible if the ID is unique per
   run rather than per file. §21, §28, and §52 amended.

---

# 1. Objective

Implement the Qwen Coder candidate-generation layer.

The purpose of this stage is to take a validated Python programming problem and generate multiple independent Python implementations using a Qwen Coder model.

The generated implementations will become candidate solutions that are evaluated in later stages.

The pipeline at the end of this stage must be:

```
Problem
   ↓
Prompt Builder
   ↓
Qwen Coder
   ↓
Raw Model Output
   ↓
Python Code Extraction
   ↓
Candidate Record
   ↓
candidates.jsonl
```

This stage must NOT implement candidate execution or evaluation.

---

# 2. Scope

This stage MUST implement:

1. Model-client abstraction.
2. Qwen model client.
3. Model configuration.
4. Candidate-generation configuration.
5. Generation strategies.
6. Prompt construction.
7. Qwen inference.
8. Python-code extraction.
9. Candidate validation at the syntax/extraction level.
10. Candidate persistence.
11. Resumable generation.
12. Generation logging.
13. CLI commands.
14. Unit tests using a mock model.
15. An optional small real-model smoke test.

This stage MUST NOT implement:

* Docker sandbox execution.
* `pytest` execution of generated candidates.
* Candidate correctness evaluation.
* Candidate ranking.
* Preference generation.
* DPO.
* TRL.
* LoRA/QLoRA.
* Benchmarking.
* Ruff.
* mypy.
* LLM-as-a-judge.
* Mutation testing.

---

# 3. Design Principle

The candidate generator must be independent of the later evaluation pipeline.

The generator's responsibility is:

> Generate and persist candidate Python programs.

It must NOT decide whether a candidate is correct.

Correctness will be determined in later stages by executing the candidate against the problem's test cases.

---

# 4. High-Level Architecture

Implement the following architecture:

```
ProblemRepository
       │
       ▼
CandidateGenerator
       │
       ├── PromptBuilder
       │
       ▼
  ModelClient
       │
       ▼
   Qwen Coder
       │
       ▼
  RawGeneration
       │
       ▼
 CodeExtractor
       │
       ▼
   Candidate
       │
       ▼
CandidateRepository
       │
       ▼
candidates.jsonl
```

The Qwen-specific implementation must exist behind a generic model interface.

---

# 5. Model Abstraction

Create a generic model-client interface.

Recommended module:

```
src/python_dpo/models/base.py
```

Define an abstraction similar to:

```
class ModelClient(Protocol):
    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig
    ) -> GenerationResult:
        ...
```

The exact implementation mechanism may use an abstract base class or `Protocol`.

The rest of the pipeline must depend on `ModelClient`, not directly on Hugging Face Transformers.

---

# 6. Qwen Model Client

Create:

```
src/python_dpo/models/qwen.py
```

Implement:

```
QwenModelClient
```

The implementation should use the Hugging Face Transformers stack.

Do not hard-code a specific Qwen model name into Python source code.

The model identifier must come from configuration.

Example conceptual configuration:

```
model:
  provider: transformers
  name: <QWEN_MODEL_ID>
  revision: null
  device: auto
  dtype: auto
```

The implementation must allow the model to be changed without modifying source code.

---

# 7. Model Loading

The model should be loaded lazily.

Do NOT load Qwen when the package is merely imported.

The model should be initialized only when generation is requested.

Bad:

```
import python_dpo
→ loads several GB of model weights
```

Desired:

```
import python_dpo
→ no model loading

python -m python_dpo generate
→ model loading begins
```

---

# 8. Model Loading Configuration

Support configuration for:

```
model.name
model.revision
model.device
model.dtype
model.trust_remote_code
```

Do not silently enable unsafe settings.

If `trust_remote_code` is supported, make it an explicit configuration option.

Default it to the safer value unless the selected model requires otherwise.

---

# 9. Quantization

Do NOT require quantization in Step 3.

The architecture should not prevent adding:

* 4-bit quantization
* 8-bit quantization
* BitsAndBytes
* vLLM

later.

For the first implementation, use standard Transformers inference.

The configuration should leave room for a future:

```
model.quantization
```

field, but do not implement complex quantization logic unless required for the selected model to run.

---

# 10. GPU / CPU Support

The model client should support:

```
device: auto
```

If CUDA is available, `auto` should select an appropriate CUDA device.

If CUDA is unavailable, the application should produce a clear error if the selected model cannot reasonably run on CPU.

Do not add GPU-specific assumptions throughout the code.

The model client must encapsulate hardware-specific behavior.

---

# 11. Generation Configuration

Create a typed configuration object:

```
GenerationConfig
```

It should include at least:

```
temperature
top_p
max_new_tokens
do_sample
repetition_penalty
seed
```

Example:

```
generation:
  candidates_per_problem: 5
  temperature: 0.8
  top_p: 0.95
  max_new_tokens: 512
  do_sample: true
  repetition_penalty: 1.0
  seed: 42
```

The exact defaults may be adjusted after testing.

---

# 12. Candidate Generation Count

The initial target is:

```
5 candidates per problem
```

Do not hard-code `5` in the implementation.

Use:

```
generation.candidates_per_problem
```

from configuration.

The CLI should support overriding this value.

Example:

```
python -m python_dpo generate --num-candidates 5
```

---

# 13. Generation Strategies

Implement exactly five initial strategies:

```
normal
straightforward
edge_case_focused
alternative
optimized
```

Each strategy should produce a different instruction to the model.

Example:

### normal

```
Solve the problem using a clear, correct Python implementation.
```

### straightforward

```
Provide a simple and easy-to-understand Python implementation.
```

### edge_case_focused

```
Pay particular attention to empty inputs, boundary conditions,
duplicates, and other edge cases specified by the problem.
```

### alternative

```
Solve the problem using an alternative reasonable algorithm
or implementation approach.
```

### optimized

```
Produce an efficient implementation appropriate for the
problem constraints.
```

These are generation prompts, not correctness labels.

Do not assume that the optimized candidate will actually be more efficient.

Later evaluation determines that.

---

# 14. Prompt Builder

Create:

```
src/python_dpo/generation/prompt_builder.py
```

The prompt builder must receive a `Problem` and generation strategy.

It must construct a prompt containing:

1. Problem statement.
2. Function signature.
3. Relevant requirements.
4. Generation strategy.
5. Explicit output-format instructions.

Example:

```
You are an expert Python programmer.

Solve the following programming problem.

Problem:
{problem.prompt}

Required function signature:
{problem.signature}

Strategy:
{strategy_instruction}

Requirements:
- Implement the requested function.
- Follow the function signature.
- Handle the specified edge cases.
- Use Python.
- Return only the implementation.
- Do not provide an explanation.
- Do not use eval().
- Do not use exec().
- Do not perform network operations.
- Do not read or write files.
```

The prompt should be deterministic given the same problem and strategy.

---

# 15. Prompt Versioning

Every generated candidate must record the prompt version.

For example:

```
prompt_version: "v1"
```

This allows later experiments to distinguish datasets generated with different prompt templates.

Do not silently change the prompt template without changing its version.

---

# 16. Model Output Format

The model is expected to return Python code.

The model may nevertheless return:

````
```python
def solution(...):
    ...
```
````

or:

````
Here is the solution:

```python
def solution(...):
    ...
```
````

or other formatting.

Therefore, implement a code-extraction layer.

---

# 17. Code Extractor

Create:

```
src/python_dpo/generation/code_extractor.py
```

The extractor should handle at least:

### Plain code

```
def foo(x):
    return x + 1
```

### Markdown code fence

````
```python
def foo(x):
    return x + 1
```
````

### Generic code fence

````
```
def foo(x):
    return x + 1
```
````

### Explanatory prefix

````
Here is the implementation:

```python
def foo(x):
    return x + 1
```
````

The extractor must preserve the actual Python source.

---

# 18. Code Extraction Rules

The extractor must NOT attempt to repair arbitrary Python code.

It should:

1. Extract likely Python source.
2. Normalize surrounding whitespace.
3. Preserve internal formatting.
4. Return extraction metadata.

For example:

```
ExtractionResult(
    code="def foo(x):\n    return x + 1",
    extracted=True,
    source_format="python_fence"
)
```

If extraction fails:

```
ExtractionResult(
    code=None,
    extracted=False,
    source_format="unknown",
    error="No Python code detected"
)
```

Do not silently convert extraction failures into successful candidates.

---

# 19. Syntax Validation

After extracting code, perform Python syntax validation.

Use the Python AST parser:

```
ast.parse(code)
```

Do NOT execute the candidate.

A candidate with invalid syntax must be recorded as:

```
syntax_valid: false
```

A syntactically valid candidate:

```
syntax_valid: true
```

This is only a syntax check.

It does NOT establish correctness.

## 19.1 Syntax failure is not a generation failure

A candidate whose code fails `ast.parse` is still a candidate.

It MUST be written to `candidates.jsonl` with `syntax_valid: false`, and MUST NOT also be
written to `generation_failures.jsonl`. Recording it in both files would count one
generation twice.

The record is the model's actual output (§44). Refusing to persist it because it does not
parse would discard exactly the low-quality outputs that later stages need in order to
build preference pairs.

The syntax error message SHOULD be retained alongside the flag:

```
syntax_valid: false
syntax_error: "unexpected EOF while parsing (line 4)"
```

Contrast with §26: extraction failure means there is **no code to store**, so no candidate
exists. Syntax failure means there **is** code, and it is stored as generated.

---

# 20. Candidate Model

Create a typed `Candidate` model.

Recommended fields:

```
candidate_id
problem_id
model
model_revision
strategy
prompt_version
prompt
raw_output
code
syntax_valid
generation_config
generation_index
created_at
```

Example:

```
{
  "candidate_id": "p001_c001",
  "problem_id": "p001",
  "model": "Qwen/...",
  "strategy": "normal",
  "prompt_version": "v1",
  "syntax_valid": true,
  "code": "def sum_even(...): ...",
  ...
}
```

---

# 21. Candidate IDs

Candidate IDs must be deterministic and unique.

Use a structure such as:

```
{problem_id}_c{index}
```

Example:

```
p001_c001
p001_c002
p001_c003
p001_c004
p001_c005
```

Do not use random UUIDs as the primary candidate ID.

A separate run ID may be used to distinguish experiments.

## 21.1 Scope of uniqueness

`candidate_id` is unique **within a generation run**, not within the file.

`candidates.jsonl` is append-only (§25, §41), and `--force` starts a new run (§28.1), so
the same `candidate_id` will legitimately appear more than once with different `run_id`
values. That is the mechanism §22 describes for comparing multiple generations of the same
problem.

The file-wide unique key is therefore the pair:

```
(run_id, candidate_id)
```

Readers that want one record per candidate must select the newest `run_id` per
`candidate_id`. This keeps §21's deterministic IDs and §52's "a new generation run is
created" true at the same time, without deleting anything.

---

# 22. Generation Run ID

Every generation invocation should have a `run_id`.

Example:

```
run_id: "20260817_103000"
```

or another deterministic/unique representation.

Store the run ID in each candidate record.

This makes it possible to compare multiple generations of the same problem.

---

# 23. Random Seeds

Support a configurable seed.

Example:

```
seed: 42
```

Where supported by the inference backend, initialize the relevant random generators.

The goal is reproducibility.

However, do not claim bit-for-bit reproducibility unless the underlying inference stack actually provides it.

Document this distinction.

---

# 24. Candidate Persistence

Create:

```
src/python_dpo/candidates/repository.py
```

Candidates must be persisted to:

```
data/candidates/candidates.jsonl
```

Write one candidate per line.

Write the candidate immediately after successful generation/extraction.

Do not wait until all candidates have been generated.

This ensures partial runs are recoverable.

---

# 25. Raw Output Persistence

The complete raw model output must be retained.

Do not store only the extracted Python code.

This is important for debugging extraction failures.

Store both:

```
raw_output
```

and:

```
code
```

Example:

````
raw_output:
    "Here is the solution:\n```python\n...\n```"

code:
    "def foo(...):\n    ..."
````

---

# 26. Generation Failure Handling

A **generation failure** is a generation that produced no candidate at all.

Possible per-generation failures:

* tokenizer failure
* inference exception
* timeout
* empty model response
* code extraction failure

These must not crash the entire dataset generation job. Record a structured generation
failure and continue with the next candidate.

Example:

```
GenerationResult(
    success=False,
    problem_id="p001",
    generation_index=3,
    error_type="code_extraction",
    error_message="No Python code detected"
)
```

Do not create a fake candidate for a failed generation.

## 26.1 Syntax errors are NOT generation failures

A syntax error is not in the list above. Code that was extracted but does not parse yields
a persisted candidate with `syntax_valid: false` — see §19.1.

## 26.2 Model loading failure aborts the run

Model loading failure is a **run-level** failure, not a per-generation one.

If the weights, tokenizer, or device cannot be initialized, no candidate in the run can
succeed, and retrying per candidate would emit one identical failure record per generation.

On model loading failure:

1. Record a single failure with `error_type: "model_load"`.
2. Log the error.
3. Abort the run and exit non-zero.

Candidates already persisted before the failure are retained, and the run remains resumable
(§28).

---

# 27. Failed Generation Persistence

Create a separate file:

```
data/candidates/generation_failures.jsonl
```

Each failure should include:

```
run_id
problem_id
generation_index
strategy
error_type
error_message
timestamp
```

`error_type` must come from a closed set, so failures can be counted and compared across
runs rather than grouped by free text:

```
model_load        (run-level, per §26.2)
tokenizer
inference
timeout
empty_output
code_extraction
```

Note that `syntax_error` is deliberately absent — see §19.1 and §26.1.

This ensures failures are observable.

---

# 28. Resumability

The generator must be restartable.

If:

```
p001_c001
p001_c002
p001_c003
```

already exist for the current run, the generator must not regenerate them unless:

```
--force
```

is specified.

Example:

```
python -m python_dpo generate
```

should resume.

Explicit regeneration:

```
python -m python_dpo generate --force
```

## 28.1 What resume and `--force` actually compare

Resume is keyed on `(problem_id, generation_index)` across **all** runs present in
`candidates.jsonl`. If any run already produced that pair, the generation is skipped.

A generation that previously *failed* left no candidate record behind, so it is retried on
the next run. This is intended: failures are observable in
`generation_failures.jsonl` (§27) but are not a reason to give up permanently.

`--force` does not delete or rewrite anything. It:

1. Mints a new `run_id`.
2. Ignores the resume index entirely.
3. Appends the newly generated candidates.

After a forced regeneration of five candidates, `candidates.jsonl` holds ten records — the
original run and the new one — and both remain readable (§21.1).

---

# 29. Problem Selection

The CLI must support:

```
--problem-id
```

Example:

```
python -m python_dpo generate --problem-id p001
```

It should generate candidates only for that problem.

Also support:

```
--limit
```

Example:

```
python -m python_dpo generate --limit 2
```

This should generate candidates for the first two problems.

---

# 30. Candidate Count Override

Support:

```
--num-candidates
```

Example:

```
python -m python_dpo generate --problem-id p001 --num-candidates 5
```

This should override the configuration value for that execution only.

---

# 31. Strategy Override

Support:

```
--strategy
```

Example:

```
python -m python_dpo generate \
    --problem-id p001 \
    --strategy normal
```

If no strategy is specified, use the configured strategy list.

When generating five candidates by default, use:

```
normal
straightforward
edge_case_focused
alternative
optimized
```

one candidate per strategy.

---

# 32. Real Model vs Mock Model

The architecture must support two modes.

### Real mode

Uses the actual Qwen model.

### Mock mode

Uses a deterministic fake model for automated tests.

Example:

```
MockModelClient
```

must return predefined Python implementations.

Unit tests must NEVER require:

* GPU
* Qwen weights
* Hugging Face authentication
* internet access

---

# 33. Mock Model

Create a deterministic mock model.

For example:

```
class MockModelClient:
    ...
```

It should return predictable output based on the supplied problem or strategy.

This allows testing:

* prompt construction
* candidate generation
* extraction
* persistence
* resumability
* failure handling

without loading Qwen.

---

# 34. Real-Model Smoke Test

Provide an optional test/command for real Qwen inference.

It must NOT run as part of normal pytest.

Example:

```
python -m python_dpo generate \
    --problem-id p001 \
    --num-candidates 1 \
    --real-model
```

The exact CLI option may instead be configuration-driven.

The important requirement is:

> Real-model tests must be explicitly requested.

---

# 35. CLI

Extend the CLI with:

```
python -m python_dpo generate
```

Support:

```
--problem-id
--limit
--num-candidates
--strategy
--force
--dry-run
```

Example:

```
python -m python_dpo generate \
    --problem-id p001 \
    --num-candidates 5
```

---

# 36. Dry Run

Implement:

```
--dry-run
```

Dry run must:

* load the selected problems
* construct prompts
* display/log what would be generated
* NOT load Qwen
* NOT perform inference
* NOT write candidate records

This allows prompt inspection before expensive inference.

Example:

```
python -m python_dpo generate \
    --problem-id p001 \
    --dry-run
```

---

# 37. Configuration

Extend `config.yaml`.

Add:

```
model:
  provider: transformers
  name: "<QWEN_MODEL_ID>"
  revision: null
  device: auto
  dtype: auto
  trust_remote_code: false

generation:
  candidates_per_problem: 5
  temperature: 0.8
  top_p: 0.95
  max_new_tokens: 512
  do_sample: true
  repetition_penalty: 1.0
  seed: 42

generation_strategies:
  - normal
  - straightforward
  - edge_case_focused
  - alternative
  - optimized
```

Do not put API keys or authentication tokens in `config.yaml`.

---

# 38. Environment Variables

Allow sensitive model configuration to be provided through environment variables if needed.

Never write:

```
HF_TOKEN=...
```

into:

* source code
* config.yaml
* JSONL datasets
* logs
* README

If Hugging Face authentication is required, document the expected environment variable without storing its value.

---

# 39. Logging

Log:

* model loading start
* model loading completion
* problem ID
* generation index
* generation strategy
* generation start
* generation completion
* output length
* extraction status
* syntax status
* persistence status
* errors

Example:

```
INFO | Loading Qwen model
INFO | Generating p001 candidate 1/5 | strategy=normal
INFO | Generated p001_c001 | syntax_valid=true
INFO | Persisted p001_c001
```

Do not log:

* authentication tokens
* secrets
* full prompts at INFO level unless explicitly requested
* full model outputs at INFO level

Raw outputs are persisted separately.

---

# 40. Prompt Logging

For debugging, support an optional DEBUG mode that logs prompts.

Example:

```
logging:
  level: DEBUG
```

However, do not duplicate large prompts and model outputs unnecessarily in standard logs.

The authoritative copy should be the candidate record.

---

# 41. Duplicate Detection

At this stage, detect exact duplicate generated code.

If:

```
candidate A.code == candidate B.code
```

record:

```
duplicate_of = candidate_A_id
```

Do NOT automatically delete the duplicate.

Keep it for analysis.

Do not yet perform semantic duplicate detection.

---

# 42. Candidate Validation

Candidate validation at this stage is limited to:

1. Non-empty output.
2. Successful code extraction.
3. Valid Python syntax.
4. Candidate implements or contains the requested function name where practical.

Do NOT determine whether the function is semantically correct.

For example:

```
def foo(x):
    return 123
```

may be syntactically valid even if it is logically wrong.

That is expected.

---

# 43. Function Name Extraction

Use the declared function signature from the Problem model.

Where practical, parse the candidate using `ast` and verify that the expected function exists.

For example:

```
expected:
    def first_unique(s):
```

Candidate:

```
def first_unique(s):
    ...
```

→ valid structure

Candidate:

```
def solution(s):
    ...
```

→ structural mismatch

Record:

```
function_name_valid: false
```

Do not execute the candidate to determine this.

---

# 44. No Automatic Code Repair

Do NOT ask another model to repair malformed candidate code.

Do NOT automatically modify generated Python.

The candidate must represent the model's actual generated output.

If extraction or syntax validation fails, record the failure.

This preserves the integrity of the preference-generation experiment.

---

# 45. Data Integrity

Every candidate must be traceable:

```
problem
   ↓
prompt
   ↓
model
   ↓
generation configuration
   ↓
raw output
   ↓
extracted code
   ↓
validation result
```

Do not create candidates whose provenance cannot be reconstructed.

---

# 46. Testing Requirements

Create tests for:

## Model abstraction

* interface behavior
* mock model

## Prompt builder

* correct problem included
* signature included
* strategy included
* output instructions included
* deterministic output

## Code extractor

Test:

* plain Python
* Python markdown fence
* generic markdown fence
* explanatory text
* empty output
* malformed output

## Syntax validator

Test:

* valid Python
* invalid Python

## Function-name validator

Test:

* correct function name
* wrong function name
* missing function

## Candidate model

Test:

* valid candidate
* invalid candidate

## Candidate repository

Test:

* write
* read
* duplicate detection
* persistence
* resume behavior

## Generation pipeline

Using `MockModelClient`, test:

* one problem
* five candidates
* strategy assignment
* failure handling
* resume
* force regeneration

No test should load the real Qwen model.

---

# 47. Integration Test

Create an integration test using the mock model.

Input:

```
1 problem
```

Generate:

```
5 candidates
```

Expected:

```
5 candidate records
```

Verify:

* all records have the correct problem ID
* all candidate IDs are unique
* all five strategies are represented
* raw output is preserved
* extracted code is preserved
* syntax validation occurs
* records are persisted

---

# 48. Real-Model Smoke Test

Provide a manual smoke-test procedure.

It should:

1. Load the configured Qwen model.
2. Select one problem.
3. Generate one candidate.
4. Extract Python.
5. Validate syntax.
6. Persist the candidate.

Do not run this automatically in CI.

---

# 49. Expected Candidate Dataset

After running:

```
python -m python_dpo generate \
    --problem-id p001 \
    --num-candidates 5
```

the output should contain five records.

Example conceptual record:

```
{
  "candidate_id": "p001_c001",
  "problem_id": "p001",
  "run_id": "...",
  "model": "Qwen/...",
  "strategy": "normal",
  "prompt_version": "v1",
  "generation_index": 1,
  "raw_output": "...",
  "code": "def ...",
  "syntax_valid": true,
  "function_name_valid": true,
  "generation_config": {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 512
  }
}
```

The exact model output must not be hard-coded.

---

# 50. Important Separation of Responsibilities

The following responsibilities belong to later stages:

### Candidate Generator

Responsible for:

```
"What code did Qwen generate?"
```

### Evaluator

Responsible for:

```
"Does the code work?"
```

### Ranker

Responsible for:

```
"Which candidate is better?"
```

### Preference Builder

Responsible for:

```
"How do we construct chosen/rejected pairs?"
```

Do not mix these responsibilities.

---

# 51. Acceptance Criteria

Step 3 is complete only when:

* [ ] `ModelClient` abstraction exists.
* [ ] `QwenModelClient` exists.
* [ ] Model configuration is externalized.
* [ ] Model loads lazily.
* [ ] Generation configuration is typed.
* [ ] Five generation strategies exist.
* [ ] Prompt builder exists.
* [ ] Prompt versioning exists.
* [ ] Code extraction exists.
* [ ] Syntax validation exists.
* [ ] Function-name validation exists.
* [ ] Candidate model exists.
* [ ] Candidate repository exists.
* [ ] Raw model output is persisted.
* [ ] Candidate code is persisted.
* [ ] Generation failures are persisted.
* [ ] A syntax-invalid candidate is persisted as a candidate with `syntax_valid: false`,
  and is NOT written to `generation_failures.jsonl` (§19.1, §26.1).
* [ ] An extraction failure produces a failure record and NO candidate (§26).
* [ ] Model loading failure aborts the run with a single `model_load` failure (§26.2).
* [ ] Generation is resumable.
* [ ] `--force` regeneration works, appending a new run without overwriting the previous
  one (§21.1, §28.1).
* [ ] `--problem-id` works.
* [ ] `--limit` works.
* [ ] `--num-candidates` works.
* [ ] `--strategy` works.
* [ ] `--dry-run` works.
* [ ] Mock model exists.
* [ ] Unit tests do not require GPU.
* [ ] Unit tests do not require internet.
* [ ] Real Qwen inference can be explicitly invoked.
* [ ] One-problem real-model smoke test succeeds.
* [ ] Five candidates can be generated for one problem.
* [ ] No generated code is executed.
* [ ] No Docker sandbox has been implemented.
* [ ] No pytest evaluation of candidates has been implemented.
* [ ] No DPO code has been implemented.
* [ ] All automated tests pass.

---

# 52. Verification Procedure

First run:

```
pytest -q
```

Then inspect prompt generation:

```
python -m python_dpo generate \
    --problem-id p001 \
    --dry-run
```

Then run the real model for one candidate:

```
python -m python_dpo generate \
    --problem-id p001 \
    --num-candidates 1
```

Verify:

```
data/candidates/candidates.jsonl
```

Then generate all five:

```
python -m python_dpo generate \
    --problem-id p001 \
    --num-candidates 5
```

Verify:

```
five candidate records
```

Then run the same command again:

```
python -m python_dpo generate \
    --problem-id p001 \
    --num-candidates 5
```

The existing candidates must not be regenerated unless `--force` is specified.

Finally:

```
python -m python_dpo generate \
    --problem-id p001 \
    --num-candidates 5 \
    --force
```

Verify that a new generation run is created:

```
candidates.jsonl now contains ten records
five carry the original run_id
five carry the new run_id
the earlier five are unchanged
```

Nothing is overwritten or removed (§21.1, §28.1).

---

# 53. Expected Output After Step 3

At the end of this stage:

```
data/
├── problems/
│   └── problems.jsonl
│
└── candidates/
    ├── candidates.jsonl
    └── generation_failures.jsonl
```

The system should support:

```
Problem
   ↓
Qwen Coder
   ↓
5 candidate programs
   ↓
Persist candidates
```

Nothing beyond this should be implemented.

---

# 54. Final Implementation Report

After implementation, report:

1. Model abstraction design.
2. Qwen model used.
3. Model loading configuration.
4. Generation configuration.
5. Generation strategies.
6. Prompt format.
7. Code-extraction behavior.
8. Candidate schema.
9. Persistence mechanism.
10. Resume behavior.
11. Failure handling.
12. Test results.
13. Real-model smoke-test results.
14. Files created/modified.
15. Dependencies added.
16. Any deviations from this specification.
17. Any issues requiring review before Step 4.

Do NOT implement Step 4 automatically.

Wait for explicit approval before implementing candidate persistence/evaluation integration.
