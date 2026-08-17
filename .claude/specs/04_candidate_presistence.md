# Python DPO Data Generation Pipeline

## Step 4 — Candidate Persistence, Runs and Reproducibility

**Specification Version:** 1.0
**Status:** Implementation Specification
**Step:** 4 of 12
**Depends On:** Step 1 — Project Skeleton
**Depends On:** Step 2 — Python Problem Dataset
**Depends On:** Step 3 — Qwen Candidate Generator

---

# 1. Objective

Strengthen the candidate-generation persistence layer so that generated Qwen candidates are:

* durable
* traceable
* reproducible
* resumable
* independently identifiable
* recoverable after interruption
* auditable
* protected against accidental overwrites

Step 3 introduced basic candidate generation and persistence.

Step 4 turns that persistence mechanism into a reliable experiment artifact store.

The primary objective is:

> A candidate-generation run must be safely interruptible and restartable without losing already-generated candidates or silently changing existing artifacts.

---

# 2. Scope

This stage MUST implement:

1. Generation run management.
2. Run IDs.
3. Run manifests.
4. Candidate artifact persistence.
5. Atomic writes.
6. Resume capability.
7. Force/new-run capability.
8. Candidate lookup.
9. Candidate indexing.
10. Duplicate detection.
11. Candidate checksums/hashes.
12. Generation statistics.
13. Failure persistence.
14. Run-level metadata.
15. Dataset integrity validation.
16. CLI commands for inspecting runs and candidates.
17. Comprehensive tests.

This stage MUST NOT implement:

* Docker execution.
* Candidate correctness evaluation.
* pytest execution against generated code.
* Candidate ranking.
* Preference generation.
* DPO.
* LoRA/QLoRA.
* Benchmarking.
* Ruff/mypy.
* Security scanning.
* LLM judging.

---

# 3. Design Principle

The candidate dataset is an experimental artifact.

A future researcher must be able to answer:

> Exactly which model, prompt, generation configuration, strategy, and problem produced this candidate?

The system must therefore preserve the complete provenance chain:

```
Problem
   ↓
Problem version
   ↓
Prompt
   ↓
Prompt version
   ↓
Model
   ↓
Model revision
   ↓
Generation configuration
   ↓
Generation strategy
   ↓
Raw model output
   ↓
Extracted code
   ↓
Candidate ID
```

---

# 4. Important Clarification

Step 3 already implemented basic candidate persistence.

Do NOT rewrite the Step 3 candidate generator unnecessarily.

Instead:

* inspect the existing implementation
* preserve compatible interfaces
* improve persistence where required
* refactor only where necessary
* maintain backward compatibility with existing candidate records where practical

The goal is to strengthen the existing implementation, not replace it with an unrelated architecture.

---

# 5. Run Concept

Every candidate-generation execution belongs to a **generation run**.

Example:

```
run_20260817_133700_a81f
```

A run represents one experiment.

A run may generate candidates for:

* one problem
* several problems
* the complete dataset

A run must have a unique ID.

---

# 6. Run Directory Structure

Create:

```
data/candidates/runs/
```

Each run should have its own directory.

Example:

```
data/candidates/runs/
└── run_20260817_133700_a81f/
    ├── manifest.json
    ├── candidates.jsonl
    ├── failures.jsonl
    ├── statistics.json
    └── prompts/
        └── prompts.jsonl
```

Do not duplicate the complete raw dataset unnecessarily.

The exact directory layout may differ slightly, but all run-level artifacts must remain associated with a specific run.

---

# 7. Run Manifest

Create a run manifest.

Example:

```
{
  "run_id": "run_20260817_133700_a81f",
  "created_at": "...",
  "status": "running",
  "model": {
    "name": "Qwen/...",
    "revision": "..."
  },
  "prompt_version": "v1",
  "generation_config": {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "do_sample": true,
    "seed": 42
  },
  "strategies": [
    "normal",
    "straightforward",
    "edge_case_focused",
    "alternative",
    "optimized"
  ],
  "requested_problems": 10,
  "requested_candidates_per_problem": 5
}
```

The manifest must capture the configuration used for the run.

---

# 8. Run Status

A run must have one of:

```
created
running
completed
failed
interrupted
cancelled
```

The system must update run status appropriately.

For example:

```
created
   ↓
running
   ↓
completed
```

If the process is interrupted:

```
running
   ↓
interrupted
```

When resumed:

```
interrupted
   ↓
running
   ↓
completed
```

---

# 9. Run Completion

A run can only be marked:

```
completed
```

when all requested generation work has either:

* successfully produced candidates, or
* been explicitly recorded as a generation failure.

A candidate-generation failure must not silently disappear.

---

# 10. Run Failure

If an infrastructure failure prevents the pipeline from completing, mark the run:

```
failed
```

and record:

* error type
* error message
* timestamp
* affected problem
* affected candidate index

Do not mark an incomplete run as `completed`.

---

# 11. Resume Semantics

The following command:

```
python -m python_dpo generate
```

must resume an incomplete run when explicitly requested.

Provide a CLI mechanism such as:

```
--resume RUN_ID
```

Example:

```
python -m python_dpo generate --resume run_20260817_133700_a81f
```

The implementation may also support automatic resume, but explicit resume is required.

---

# 12. Resume Rules

Suppose a run requires:

```
10 problems
×
5 candidates
=
50 candidates
```

and the process stops after candidate 23.

When resumed:

* candidates 1–23 must NOT be regenerated
* candidates 24–50 must be generated
* existing artifacts must remain unchanged

The system must determine completed work from persisted records rather than relying only on in-memory state.

---

# 13. Force Regeneration

Support:

```
--force
```

Force regeneration must NOT overwrite existing candidate records in place.

Instead, create a new generation run.

Example:

```
run_A
   ↓
--force
   ↓
run_B
```

This preserves experiment history.

---

# 14. Candidate Identity

A candidate must have a stable identity within a run.

Recommended:

```
candidate_id = "{problem_id}_c{generation_index}"
```

Example:

```
p001_c001
p001_c002
p001_c003
```

The candidate ID alone is not globally unique across runs.

Therefore the complete identity is:

```
run_id + candidate_id
```

---

# 15. Candidate Record

Ensure candidate records contain at least:

```
run_id
candidate_id
problem_id
generation_index
strategy
model
model_revision
prompt_version
prompt
raw_output
code
syntax_valid
function_name_valid
generation_config
created_at
```

Do not remove provenance fields introduced in Step 3.

---

# 16. Candidate Hash

Calculate a cryptographic hash of the extracted candidate code.

Use a standard cryptographic hash such as SHA-256.

Example:

```
code_sha256:
  "8f3a..."
```

This allows:

* exact duplicate detection
* artifact integrity checks
* reproducibility analysis

Do not use the hash as the candidate ID.

---

# 17. Prompt Hash

Also calculate a SHA-256 hash of the final prompt.

Store:

```
prompt_sha256
```

This makes it possible to verify that two candidates were generated from identical prompts.

---

# 18. Raw Output Hash

Calculate:

```
raw_output_sha256
```

for the raw model response.

This allows integrity verification without exposing raw outputs in logs.

---

# 19. Duplicate Detection

Detect exact duplicate code within a run.

Example:

```
p001_c001 → code_hash=A
p001_c002 → code_hash=B
p001_c003 → code_hash=A
```

Then:

```
p001_c003.duplicate_of = "p001_c001"
```

Do NOT delete the duplicate.

The duplicate is useful experimental information.

---

# 20. Cross-Run Duplicate Detection

Do not automatically reject duplicates across different runs.

For example:

```
run_A / p001_c001
run_B / p001_c003
```

may contain identical code.

Record the hash so that cross-run analysis can identify duplicates later.

Do not deduplicate historical experiment artifacts.

---

# 21. Atomic Persistence

Candidate records must be persisted atomically.

Do not directly write partial JSON records into the authoritative dataset.

Use a safe pattern such as:

```
write temporary record
      ↓
flush
      ↓
fsync where appropriate
      ↓
atomic rename
```

For JSONL append operations, ensure that a process interruption cannot leave an ambiguous half-record that is treated as valid data.

The exact implementation can use a temporary staging mechanism or another robust strategy.

---

# 22. JSONL Integrity

Every line in:

```
candidates.jsonl
```

must be a complete valid JSON object.

The repository must provide a validation utility that detects:

* malformed JSON
* truncated records
* duplicate candidate IDs
* missing required fields
* invalid hashes
* inconsistent run IDs

---

# 23. Candidate Repository

Extend the repository from Step 3.

It must support:

```
save(candidate)

get(run_id, candidate_id)

exists(run_id, candidate_id)

list(run_id)

count(run_id)

find_by_problem(run_id, problem_id)

find_by_hash(run_id, code_sha256)
```

Do not expose raw filesystem operations throughout the rest of the application.

The repository owns candidate persistence.

---

# 24. Run Repository

Create a run repository.

It should support:

```
create_run(...)

get_run(run_id)

update_status(run_id, status)

list_runs()

resume_run(run_id)

complete_run(run_id)
```

The exact API may differ, but run state must be managed centrally.

---

# 25. Run Statistics

Maintain:

```
statistics.json
```

containing information such as:

```
{
  "problems_requested": 10,
  "problems_completed": 10,
  "candidates_requested": 50,
  "candidates_generated": 48,
  "generation_failures": 2,
  "syntax_valid": 47,
  "syntax_invalid": 1,
  "duplicates": 4
}
```

Statistics should be reconstructable from persisted records.

Do not rely exclusively on counters stored in memory.

---

# 26. Statistics Semantics

Distinguish:

```
requested
```

from:

```
generated
```

from:

```
valid
```

from:

```
failed
```

For example:

```
candidates_requested = 50

candidates_generated = 48

generation_failures = 2

syntax_valid = 46

syntax_invalid = 2
```

These must not be conflated.

---

# 27. Failure Records

Persist generation failures separately.

Each failure must contain:

```
run_id
problem_id
generation_index
strategy
error_type
error_message
created_at
```

Optional:

```
traceback
retry_count
```

Do not include secrets or authentication credentials.

---

# 28. Retry Semantics

Distinguish between:

### Infrastructure failures

Examples:

* temporary model-loading failure
* transient inference error
* out-of-memory error

and:

### Candidate failures

Examples:

* empty model output
* code extraction failure
* syntax-invalid generated code

Candidate failures should be recorded as candidate-generation outcomes.

Infrastructure failures may be retried according to a configurable retry policy.

Do not endlessly retry.

---

# 29. Retry Configuration

Add configuration such as:

```
generation:
  retry:
    max_attempts: 2
```

The exact configuration is flexible.

A retry must not overwrite the original failure record.

Record the retry attempt.

---

# 30. Generation Attempt

A candidate index may have multiple attempts.

For example:

```
problem: p001
candidate index: 3
```

Attempt 1:

```
model timeout
```

Attempt 2:

```
successful generation
```

The final candidate record should identify the successful attempt, while the failure history should remain available.

---

# 31. Prompt Artifact

Persist the exact prompts used during generation.

This may be represented inside candidate records, but also provide a run-level prompt artifact if useful.

The exact final prompt sent to Qwen must be recoverable.

Do not reconstruct historical prompts from the current prompt template.

The stored prompt is authoritative for that candidate.

---

# 32. Model Provenance

Record:

```
model_name
model_revision
```

If available, also record:

```
tokenizer_revision
transformers_version
torch_version
```

These values are important for experiment reproducibility.

Do not require all fields if the inference backend cannot provide them.

---

# 33. Environment Metadata

Record useful environment information in the run manifest:

```
python_version
platform
transformers_version
torch_version
CUDA version if available
```

Do not record:

* usernames unless required
* home directory paths
* API keys
* authentication tokens
* unnecessary personal information

---

# 34. Configuration Snapshot

At run creation, persist the effective configuration used by the run.

For example:

```
manifest.json
```

must contain the resolved:

```
model configuration
generation configuration
strategy configuration
prompt version
```

Do not rely on the current `config.yaml` to reproduce a historical run.

The run manifest is the historical source of truth.

---

# 35. CLI Commands

Extend the CLI with:

```
python -m python_dpo runs list

python -m python_dpo runs show RUN_ID

python -m python_dpo runs validate RUN_ID

python -m python_dpo candidates list RUN_ID

python -m python_dpo candidates show RUN_ID CANDIDATE_ID

python -m python_dpo candidates stats RUN_ID
```

---

# 36. `runs list`

Example output:

```
RUN ID                         STATUS       CANDIDATES
run_20260817_133700_a81f       completed    50
run_20260817_140200_b912       interrupted  23
```

The output should be concise.

---

# 37. `runs show`

Example:

```
Run:
  ID: run_20260817_133700_a81f
  Status: completed
  Model: Qwen/...
  Prompt version: v1
  Problems: 10
  Candidates: 50
```

It should also display:

```
generation configuration
creation time
completion time
failure count
```

---

# 38. `runs validate`

This command must verify:

* manifest exists
* candidate records are valid
* candidate IDs are unique
* hashes are correct
* referenced problem IDs exist
* run IDs match
* statistics are consistent

Example:

```
python -m python_dpo runs validate RUN_ID
```

Success:

```
Run validation passed.
```

Failure:

```
Run validation failed:
  candidate p001_c004:
  code_sha256 does not match stored code.
```

---

# 39. `candidates list`

Example:

```
python -m python_dpo candidates list RUN_ID
```

Output:

```
candidate_id   problem_id   strategy              syntax
p001_c001      p001         normal                 valid
p001_c002      p001         straightforward       valid
p001_c003      p001         edge_case_focused     invalid
...
```

---

# 40. `candidates show`

This command should display candidate metadata.

Example:

```
python -m python_dpo candidates show RUN_ID p001_c001
```

Display:

```
candidate ID
problem ID
strategy
model
prompt version
syntax status
code hash
duplicate information
generation timestamp
```

Do not dump the complete raw model output by default.

Provide an explicit option such as:

```
--show-code
```

if appropriate.

---

# 41. Candidate Statistics

Support:

```
python -m python_dpo candidates stats RUN_ID
```

Report:

```
Problems
Candidates requested
Candidates generated
Generation failures
Syntax-valid candidates
Syntax-invalid candidates
Duplicate candidates
Candidates by strategy
```

Example:

```
normal                  10
straightforward        10
edge_case_focused      10
alternative             9
optimized              10
```

---

# 42. Resume Workflow

The complete resume workflow must be tested.

Test scenario:

1. Create a run requiring 10 candidates.
2. Generate the first 4.
3. Simulate process interruption.
4. Resume the run.
5. Generate remaining 6.
6. Verify exactly 10 final candidates.
7. Verify the first four candidate records are byte-for-byte unchanged.

This is a mandatory integration test.

---

# 43. Interruption Testing

The implementation should handle:

* Ctrl+C
* process termination where practical
* model inference exception
* filesystem exception

At minimum, Ctrl+C should leave the run in a recoverable state.

Do not attempt to guarantee perfect recovery from abrupt machine power loss, but use atomic persistence to minimize corruption risk.

---

# 44. No Data Loss

If a candidate has been successfully persisted before the process is interrupted, restarting the pipeline must recognize it.

Do not regenerate it automatically.

Historical candidate data must not be silently overwritten.

---

# 45. No Candidate Execution

This stage MUST NOT execute candidate Python.

The following are explicitly prohibited in the persistence layer:

```
exec(candidate.code)

eval(candidate.code)

subprocess.run(candidate.code)

shell execution of candidate code
```

Candidate execution belongs to the Docker sandbox stage.

---

# 46. Backward Compatibility

Existing Step 3 candidate files must remain readable where practical.

If the schema needs to evolve:

* introduce a schema version
* provide migration logic if necessary
* do not silently reinterpret old records

Add:

```
schema_version
```

to candidate records.

Example:

```
"schema_version": "1.0"
```

---

# 47. Data Schema Versioning

The following should have explicit versions:

```
candidate_schema_version
run_manifest_version
prompt_version
```

This allows future schema evolution without corrupting historical datasets.

---

# 48. Testing Requirements

Create comprehensive tests for:

## Run repository

* create run
* retrieve run
* update status
* list runs
* completed run
* failed run
* interrupted run

## Candidate repository

* save
* retrieve
* existence check
* list
* count
* problem lookup
* hash lookup

## Hashing

* deterministic code hash
* deterministic prompt hash
* raw-output hash
* hash mismatch detection

## Duplicate detection

* identical code
* different code
* duplicate across strategies
* duplicate across candidates

## Atomic persistence

* successful write
* simulated interrupted write
* malformed/truncated record detection

## Resume

* partial run
* resume
* no duplicate generation
* candidate preservation

## Force

* new run created
* old run preserved

## Statistics

* counters match persisted records

## Validation

* corrupted candidate
* missing candidate
* invalid hash
* inconsistent run ID
* malformed JSONL

---

# 49. Integration Test

Create an integration test using the Step 3 mock model.

Scenario:

```
3 problems
×
5 candidates
=
15 candidates
```

Run generation.

Then simulate interruption after 7 candidates.

Resume.

Verify:

```
15 candidates exist
```

and:

```
first 7 candidates are unchanged.
```

Verify the final run status is:

```
completed
```

---

# 50. Reproducibility Test

Using the mock model:

1. Create Run A.
2. Generate candidates.
3. Create Run B with identical:

   * problems
   * model mock
   * prompt version
   * generation configuration
   * seed
   * strategies
4. Compare candidate outputs.

The mock implementation should produce identical outputs.

Document that real-model reproducibility may be affected by:

* GPU kernels
* framework versions
* sampling implementation
* model revision
* hardware

Do not claim deterministic real-model output without evidence.

---

# 51. Dataset Integrity Test

Create a command/test that detects:

* duplicate candidate IDs
* malformed JSON
* invalid hashes
* missing required fields
* incorrect run IDs
* invalid problem IDs
* inconsistent statistics

The integrity checker must fail loudly.

---

# 52. Performance Requirements

Do not optimize for high throughput yet.

The goal is correctness and reliability.

Do not introduce:

* distributed storage
* databases
* multiprocessing
* distributed workers
* object stores

unless required by the existing implementation.

JSONL is sufficient for the current dataset size.

---

# 53. Future Compatibility

The persistence architecture should eventually support:

```
500 problems
×
5 candidates
= 2,500 candidates
```

without architectural changes.

It should also be possible to move to:

```
SQLite
Parquet
S3
object storage
distributed workers
```

later.

Do not implement those systems now.

---

# 54. Acceptance Criteria

Step 4 is complete only when:

* [ ] Generation runs have unique IDs.
* [ ] Run manifests exist.
* [ ] Run status is persisted.
* [ ] Effective generation configuration is persisted.
* [ ] Model provenance is persisted.
* [ ] Candidate provenance is complete.
* [ ] Candidate schema has a version.
* [ ] Prompt hash is stored.
* [ ] Raw-output hash is stored.
* [ ] Code hash is stored.
* [ ] Exact duplicates are detected.
* [ ] Duplicates are retained.
* [ ] Candidate records are atomically persisted.
* [ ] Malformed/truncated records can be detected.
* [ ] Generation failures are persisted.
* [ ] Runs can be resumed.
* [ ] Existing candidates are not regenerated during resume.
* [ ] `--force` creates a new run.
* [ ] Run statistics are persisted.
* [ ] Run statistics can be reconstructed from artifacts.
* [ ] Run validation works.
* [ ] Candidate lookup works.
* [ ] CLI run inspection works.
* [ ] Integration tests pass.
* [ ] No generated candidate code is executed.
* [ ] No Docker execution is implemented.
* [ ] No candidate correctness evaluation is implemented.
* [ ] No preference generation is implemented.
* [ ] No DPO code is implemented.
* [ ] All automated tests pass.

---

# 55. Verification Procedure

Run the complete test suite:

```
pytest -q
```

Create a small run:

```
python -m python_dpo generate \
    --limit 2 \
    --num-candidates 5
```

List runs:

```
python -m python_dpo runs list
```

Inspect the run:

```
python -m python_dpo runs show RUN_ID
```

Validate it:

```
python -m python_dpo runs validate RUN_ID
```

Inspect candidates:

```
python -m python_dpo candidates list RUN_ID
```

Show statistics:

```
python -m python_dpo candidates stats RUN_ID
```

---

# 56. Mandatory Resume Test

Perform a real manual test.

Start a generation run and interrupt it after several candidates.

Then:

```
python -m python_dpo generate --resume RUN_ID
```

Verify:

```
already-generated candidates are preserved
```

and:

```
missing candidates are generated
```

and:

```
final candidate count matches the requested count
```

and:

```
final run status = completed
```

---

# 57. Expected Output

After Step 4, the artifact structure should look approximately like:

```
data/
└── candidates/
    └── runs/
        └── run_20260817_133700_a81f/
            ├── manifest.json
            ├── candidates.jsonl
            ├── failures.jsonl
            └── statistics.json
```

The exact internal organization may differ slightly, but the run must be self-contained and independently auditable.

---

# 58. Final Implementation Report

After implementation, report:

1. Run-management architecture.
2. Candidate persistence architecture.
3. Run manifest schema.
4. Candidate schema changes.
5. Hashing strategy.
6. Duplicate-detection behavior.
7. Resume behavior.
8. Force-regeneration behavior.
9. Failure-handling behavior.
10. Statistics implementation.
11. CLI commands added.
12. Tests added.
13. Resume integration-test results.
14. Reproducibility-test results.
15. Files created/modified.
16. Dependencies added.
17. Any deviations from this specification.
18. Any issues requiring review before Step 5.

Do NOT implement Step 5 automatically.

Wait for explicit approval before implementing the Docker sandbox.
