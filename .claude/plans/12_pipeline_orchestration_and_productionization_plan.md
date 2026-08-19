# Stage 12 — End-to-End Pipeline Orchestration, Model Packaging and Productionization

## Context

Eleven stages have produced eleven command groups. Running the pipeline today means a human
reads a run id out of `runs list`, pastes it into `evaluate run`, reads the next id out of
`evaluations list`, pastes it into `rank`, and so on — six times. `scripts/smoke_real_model.sh`
does exactly this, in bash, with a `python -c` one-liner to scrape the newest run id off disk.

The artifacts on this machine prove the cost. `dpo_20260818_081231_a91d` descends from
`pref_20260818_074347_5eff` ← `rank_20260817_161726_a84d` ← `eval_20260817_115154_dcd4` ←
`20260817_055411`, but that chain exists **only** as a sequence of manifest hops. Nothing
records it as a fact, nothing verifies it, and nothing would notice if the preference dataset
were regenerated underneath the adapter.

`.claude/specs/12_pipeline_orchestration_and_productionization.md` closes the loop:

> One configuration file, one command, one experiment id, and every artifact traceable to it.

Four boundaries define the stage:

- **The orchestrator adds a layer; it does not rewrite one.** Stages 2–10 already mint run
  ids, write atomic manifests, and settle `created → running → completed/failed/interrupted`.
  Stage 12 sequences them and records lineage. Any change that forces a rewrite of an existing
  repository is out of scope.
- **Cache invalidation is derived, never tabulated (§18, §19).** A stage's cache key includes
  the *output hashes of its inputs*, so changing DPO beta invalidates training → evaluation →
  analysis automatically, and provably cannot invalidate problem or candidate generation.
  There is no hand-maintained invalidation table to drift.
- **The sandbox boundary is untouched (§34, §75).** Every candidate and every generated test
  still runs only through `SandboxExecutor` / `PytestRunner`. Stage 12 adds a *second, separate*
  training/inference container (§32, §33) and must never conflate the two.
- **Nothing is promoted automatically (§48).** Packaging registers a model as `EXPERIMENTAL`.
  `RECOMMENDED` requires an explicit command and a passing Stage 10 success-criteria record.

**Outcome:** `python -m python_dpo experiment run --config configs/experiments/qwen_python_dpo_v1.yaml`
produces `data/experiments/runs/exp_.../` with a resolved immutable config, an environment
capture, nine stage manifests, an artifact manifest of SHA-256 pointers, a packaged loadable
model, an experiment report, and a complete lineage chain — resumable, cacheable and archivable.

---

### What exploration established

Measured against this repository as it stands, not assumed. The first three findings determine
what this stage can honestly build.

| Finding | Consequence |
|---|---|
| **Stage 11 does not exist.** `src/python_dpo/analysis/` is absent, `data/analysis/` is absent, `__version__` is `0.10.0`, and `README.md` calls Stage 11 "Specified and planned". Only its plan is committed (`7f949ef`) | `error_analysis` is a **registered but disabled** stage (decision 1). Its dependency edges are real, its adapter raises `StageNotImplementedError` if enabled, and the shipped config sets `enabled: false` with the state persisted as `SKIPPED` + reason. Nothing is silently absent |
| **Spec §5's nine stages do not map 1:1 onto the repo's commands.** Stage 6 (`evaluate run`) *executes* candidates in Docker **and** runs pytest in one pass; Stage 7 (`rank`) classifies, scores and orders them | `candidate_execution` → Stage 6, `candidate_evaluation` → Stage 7. Nine addressable stages are preserved and the spec's names keep their intended meaning: execution produces evidence, evaluation produces judgement |
| **`problem_generation` is not generation.** Stage 2's `problems build` calls `build_catalog()` — ten hand-authored problems with reference solutions, validated by executing those solutions in-process. No LLM, no network, no `problem_count` knob | The spec's `problem_generation.problem_count: 1000` is **unimplementable as written**. The stage is named `problem_dataset`, and `problem_count` can only *select a subset* of the curated catalog (which is what `--smoke-test` needs anyway). Flagged as a deviation, not quietly dropped |
| **Six repositories already implement the same run lifecycle.** `RunRepository`, `EvaluationRunRepository`, `RankingRunRepository`, `PreferenceRunRepository`, `TrainingRunRepository`, `ModelEvaluationRunRepository` each have `new_run_id()` (`<prefix>_YYYYMMDD_HHMMSS_xxxx`), `create_run()`, `start_run/complete_run/fail_run/interrupt_run/cancel_run`, and an atomic `manifest.json`. `runs/models.py` holds the status set and a `RUN_STATUS_TRANSITIONS` table | Stage 12's `PENDING/RUNNING/COMPLETED/FAILED/SKIPPED/CANCELLED/BLOCKED` is a **stage-level** machine layered on top, not a replacement. Each stage manifest records the underlying stage run id; the two state machines stay independent |
| **`atomic_io.atomic_write_json` already does temp-file + fsync + `os.replace` + directory fsync**, and `iter_jsonl` raises on a torn final line | §68 (atomic artifacts) is satisfied by reuse. No new persistence primitive is needed anywhere in this stage |
| **Two `capture_environment()` functions already exist** — `runs/environment.py` (python, platform, transformers, torch, cuda) and `training/versions.py`. `runs/environment.py` documents that `platform.node()` is *deliberately never called* | §30 **extends** these rather than replacing them, and §76–§78's secret/PII rule is already the established house rule rather than something new to enforce |
| **Stage bodies live inside `cli.py`'s private helpers** — `_execute_run`, `_run_evaluation`, `_rank_problem_group`, `_finalize_preference_run`, and the 200-line `_cmd_preferences_generate` / `_cmd_train_dpo` / `_cmd_evaluate_model_run`. `cli.py` is **3,589 lines** | The orchestrator must not shell out to `python -m python_dpo` and must not fabricate an `argparse.Namespace`. Those helpers **move** into `pipeline/stages/`, and `cli.py` imports them. One code path per stage, driven by both entry points |
| **`CandidateEvaluator.evaluate()` accepts an in-memory `Candidate`**, and `AdapterModelRunner` + `verify_adapter_integrity()` already load base + adapter in 4-bit and generate with a per-call seed | §38's package verification (load → generate → sandbox → assert success) and §39–§41's `model` verbs are **assembly of existing parts**, not new inference or execution code |
| **`_PLACEHOLDER_STAGES = {"run": "Full pipeline run"}`** has sat in `cli.py:163` since Stage 1, wired into `build_parser()` and asserted in `tests/test_project.py` | The placeholder was reserved for exactly this. It is removed here, and its test updated |
| **`_REQUIRED_PATH_KEYS` has nine entries.** Stage 11's plan claims `analysis` as the tenth | Stage 12 claims **`experiments`** and leaves `analysis` to Stage 11. Both plans touch `config.py`, `config.yaml`, `data/`, `.gitignore` and the *three* literal path enumerations in `tests/test_project.py` — a merge point to expect, not a conflict |
| **`src/python_dpo/models/` is already taken** by the `ModelClient` protocol, `QwenModelClient` and `MockModelClient` | Spec §96's `src/python_dpo/models/` for packaging is unavailable. Packaging goes to **`src/python_dpo/packaging/`**; the `models/registry.json` *file* still lands at the project root as §45 asks. Flagged as a deviation |
| **`--smoke-test` already exists** as a flag on `evaluate-model`, and `generate --mock-model` drives the deterministic `MockModelClient` from `src/` | Both smoke tiers reuse existing switches. The offline tier is possible *because* the mock client deliberately lives in `src/` rather than `tests/` |
| **Nothing to lock or containerize exists yet.** No `requirements.lock`, no `docker/Dockerfile` (only `docker/evaluator/Dockerfile`), no `models/registry.json`, no `data/experiments/` | §31, §32, §45 and §12 are all net-new files |
| **The runtime dependency set is PyYAML alone**, with `model` and `training` as optional extras, and `addopts = "-ra -m 'not integration and not gpu'"` keeps `pytest -q` offline with zero skips | The pipeline test tier **must** run under `--mock-model` with stubbed training/packaging adapters, or it breaks the project's zero-skip property |
| **The real data is degenerate and will stay that way.** One training run of a single optimizer step; two model-eval runs with 0 wins / 0 losses / 7 ties | A real `--smoke-test` validates the *pipeline*, which is precisely what §25 says it is for. The plan does not tune anything to make the numbers look better |

---

### Decisions confirmed with the user

1. **`error_analysis` is registered but disabled.** The stage graph carries its real edges
   (`error_analysis` requires `model_evaluation`); the shipped config sets `enabled: false`;
   the adapter raises a clear `StageNotImplementedError` if someone enables it. State is
   persisted as `SKIPPED` with `reason: "Stage 11 not implemented"` — never an omission.
   When Stage 11 lands, one adapter body changes and `enabled` flips to `true`.

2. **Canonical stores plus pointers.** Stages keep writing to `data/<stage>/runs/<run_id>/`
   through their existing repositories, entirely untouched. The experiment directory holds
   `manifest.json`, `resolved_config.yaml`, `environment.json`, `artifacts.json` (path +
   SHA-256), `stages/<stage>/stage_manifest.json`, `logs/`, `reports/`, and the packaged
   `model/`. No GB-scale duplication and no second source of truth.

3. **Full spec, executed in four phases** (see *Execution order*). Explicitly-optional
   capabilities are recorded as deliberate deferrals with reasoning, matching how Stages 9–11
   handled theirs.

4. **Two smoke tiers.** An offline `tests/pipeline/` end-to-end test in the default `pytest -q`
   suite (mock model, stubbed training and packaging), plus a real `--smoke-test` run on this
   machine's RTX 3060 + Docker whose artifacts are committed — the Stage 9/10 verification
   pattern.

---

## New package — `src/python_dpo/pipeline/`

| File | Responsibility |
|---|---|
| `__init__.py` | Public surface: `PipelineOrchestrator`, `ExperimentConfig`, `StageState`, `STAGES`, error types |
| `errors.py` | `PipelineError` base; `ExperimentConfigError`, `DependencyError`, `StageFailedError`, `StageNotImplementedError`, `PreflightError` |
| `config.py` | `ExperimentConfig.load(path)` — YAML → frozen dataclasses, one per stage section. Resolves the four-level hierarchy of §9 (root `config.yaml` → experiment file → stage section → CLI `--set key=value` overrides, CLI winning). `to_dict()` round-trips to `resolved_config.yaml`. Rejects unknown keys, matching `config.py`'s house style |
| `stages.py` | The stage registry: nine `StageSpec(name, requires, adapter, config_section)` entries in dependency order. `topological_order()`, `dependents_of(stage)`, `validate_graph()` |
| `state.py` | `StageState` enum (`PENDING/RUNNING/COMPLETED/FAILED/SKIPPED/CANCELLED/BLOCKED`) + transition table, mirroring `runs/models.py`'s shape |
| `manifest.py` | `ExperimentManifest` and `StageManifest` frozen dataclasses with `to_dict`/`from_dict` validation (§14, §28, §83). Schema tags `experiment_manifest_v1` / `stage_manifest_v1` |
| `hashing.py` | `sha256_file()`, `sha256_tree()` (sorted-path, content-addressed directory digest for `adapter/`), `config_hash()` (canonical JSON of the stage's resolved section) |
| `cache.py` | `cache_key(stage, input_hashes, config_hash, code_version, model_version)`; `is_reusable(stage_manifest, current_key)`; `invalidate(stage, cascade=True)` |
| `lineage.py` | Builds and verifies the §27 chain by hopping the existing manifests; persists `lineage.json` |
| `artifacts.py` | `ArtifactRef(name, path, sha256, bytes)`; `write_artifact_manifest()` → `artifacts.json` (§69, §70) |
| `environment.py` | §30 capture — extends `runs/environment.py` and `training/versions.py` with OS, NVIDIA driver, GPU name/VRAM, TRL, PEFT, bitsandbytes, accelerate, datasets, pytest, Docker version. Every probe in a `try`, so importing it never forces a heavy import (`tests/test_no_heavy_imports.py` must stay green) |
| `gitinfo.py` | §29 — commit SHA, branch, dirty flag via `git` subprocess with a fixed argv; `on_dirty: warn\|fail` from config |
| `preflight.py` | §60, §61 — dataset, model, benchmark, training config, evaluation config, GPU, Docker, disk. Returns a report; `format_preflight_report()` renders the `[PASS]`/`[FAIL]` lines |
| `resources.py` | §51 — duration always; GPU utilization/memory via `nvidia-smi` with a fixed argv, CPU/RAM via `os`/`/proc`, all `None` when unavailable |
| `cost.py` | §52, §53 — GPU-hours from stage wall-clock. Records the LLM-API schema with an explicit `providers: []`, since no external LLM is called anywhere in this pipeline |
| `orchestrator.py` | `PipelineOrchestrator` — resolve config, validate graph and dependencies, mint the experiment run id, write the immutable resolved config, run stages in order with per-stage logging, cache checks, state persistence, failure blocking, signal handling, and the final manifest |
| `repository.py` | `ExperimentRunRepository` — the seventh instance of the established run-repository shape, over `data/experiments/runs/` |
| `report.py` | §50, §102 — `experiment_metrics.json`, `experiment_summary.md`, `model_comparison.md`, `next_experiment.md` |
| `archive.py` | §73, §74 — `experiment archive` (tar.gz + `archive_manifest.json`) and `experiment inspect --archive` (reads the manifest member without extracting) |
| `reproduce.py` | §71, §72 — render the commands to recreate an experiment; `--verify-only` diffs model revision, dataset hash, config hash and environment against the recorded manifest |
| `stages/__init__.py` | The `StageAdapter` protocol: `run(context) -> StageResult` |
| `stages/problem_dataset.py` | Stage 2 — `build_catalog()` + `validate_dataset()` + `save_problems()`; `problem_count` selects a subset |
| `stages/candidate_generation.py` | Stage 3/4 — **receives `_execute_run` moved out of `cli.py`** |
| `stages/candidate_execution.py` | Stage 6 — **receives `_run_evaluation`'s core, moved out of `cli.py`** |
| `stages/candidate_evaluation.py` | Stage 7 — **receives `_rank_problem_group` and ranking-run settlement** |
| `stages/preference_generation.py` | Stage 8 — **receives `_finalize_preference_run` and the body of `_cmd_preferences_generate`** |
| `stages/dpo_training.py` | Stage 9 — drives `DpoTrainingJob` via `ExperimentConfig` from `configs/training/dpo_qlora.yaml` |
| `stages/model_evaluation.py` | Stage 10 — resolves the benchmark (see *Benchmark protection*), drives `EvaluationDriver` |
| `stages/error_analysis.py` | Stage 11 — raises `StageNotImplementedError`; the disabled path records `SKIPPED` |
| `stages/packaging.py` | Stage 12 — delegates to `python_dpo.packaging` |

## New package — `src/python_dpo/packaging/`

Named `packaging/` because `src/python_dpo/models/` is already the model-client package.

| File | Responsibility |
|---|---|
| `errors.py` | `PackagingError`, `VerificationError`, `MergeUnsupportedError`, `RegistryError` |
| `package.py` | §35–§37 — builds `model/{adapter/,tokenizer/,manifest.json}`. The base model is **referenced by name + revision, never copied** (§36). `ModelPackage.load(path)` reads it back |
| `verify.py` | §38 — load base → load adapter (`AdapterModelRunner`) → load tokenizer → generate Python for a fixed verification prompt → execute through `CandidateEvaluator`/`SandboxExecutor` → require success. **Packaging fails if this fails**; no `--skip-verification` escape hatch is provided |
| `merge.py` | §43, §44 — refuses to merge when the package records 4-bit quantization unless an explicit higher-precision reload path is available, and reloads the base in bf16/fp16 for the merge. Writes `merged_model/` and **never deletes `adapter/`**. If the stack cannot do it safely, raises `MergeUnsupportedError` with the reason |
| `registry.py` | §45–§48 — `models/registry.json` (`registry_v1`), atomic writes via `atomic_write_json`. Entry per §45; statuses `EXPERIMENTAL/VALIDATED/RECOMMENDED/RETIRED/REJECTED`. `promote()` refuses `RECOMMENDED` unless a recorded evaluation run passed `evaluate_success_criteria`. Packaging only ever registers `EXPERIMENTAL` |
| `inference.py` | §39, §40 — `generate(package, prompt)` and `generate_batch(package, input_jsonl, output_jsonl)` over `AdapterModelRunner`. Requires only the package plus the `model` extra (§42) |
| `compare.py` | §49 — joins registry entries and their evaluation runs: pass@1/5/10, syntax success, timeout rate, latency, memory. Reuses `model_evaluation.comparison` |

---

## Persistence layout

```
data/experiments/runs/exp_20260819_141500_a92f/
    manifest.json              # experiment_manifest_v1 (§28)
    resolved_config.yaml       # immutable after start (§10)
    environment.json           # §30
    artifacts.json             # name -> {path, sha256, bytes} (§70)
    lineage.json               # §27
    stages/
        problem_dataset/stage_manifest.json
        candidate_generation/stage_manifest.json
        ...                    # nine directories, one per stage
    model/
        adapter/               # copied from the training run (the deliverable)
        tokenizer/
        manifest.json          # model_package_v1 (§37)
    reports/
        experiment_metrics.json
        experiment_summary.md
        model_comparison.md
        next_experiment.md
    logs/
        experiment.log
        <stage>.log            # nine, per §64

models/registry.json           # project root, per §45 and §96
```

Stage outputs stay in their canonical stores; `artifacts.json` and each `stage_manifest.json`
point at them by path + SHA-256. The one deliberate copy is `model/adapter/` — the packaged
artifact is meant to be usable standalone (§42), and the adapter is ~14 MB.

### Cache key (§18) and invalidation (§19)

```
cache_key = sha256(canonical_json({
    "stage":          stage_name,
    "input_hashes":   {artifact_name: sha256, ...},   # the upstream stages' outputs
    "config_hash":    sha256(canonical_json(resolved stage section)),
    "code_version":   python_dpo.__version__,
    "model_version":  f"{base_model_name}@{revision or 'default'}",
}))
```

The cascade of §19 is **derived, not tabulated**: `dpo_training`'s inputs are the preference
dataset's hashes, so changing DPO beta changes only `dpo_training`'s `config_hash` → its
adapter hash changes → `model_evaluation`'s `input_hashes` change → it reruns → `error_analysis`
follows. `problem_dataset` and `candidate_generation` keys are untouched by construction, which
is what §91's test asserts.

**The git SHA is recorded in the manifest but deliberately excluded from the cache key.**
Including it would invalidate every stage on every commit — including documentation-only
commits — which would defeat §19's requirement that a DPO hyperparameter change leave problem
and candidate generation cached. `--force` and `--set` remain the explicit invalidation levers.

### Benchmark protection (§56, §92)

`model_evaluation` **requires an existing benchmark manifest** and fails with a clear error if
one is missing. It builds a benchmark only when the experiment config sets
`model_evaluation.build_benchmark_if_missing: true`, and even then it never overwrites an
existing manifest. Leakage is checked with the existing `check_leakage()` before every
evaluation, so a held-out problem reaching a training split fails the run (§92) rather than
producing a silently invalid number.

---

## Modifications to existing code

**`src/python_dpo/config.py` + `config.yaml` + `data/` + `.gitignore` + `tests/test_project.py`** —
`experiments` becomes the tenth `_REQUIRED_PATH_KEYS` entry and the tenth `Paths` field, with
`data/experiments/.gitkeep` and a `.gitignore` negation for `data/experiments/runs/*/logs/*.log`
(the existing rule for Stage 9/10 logs, extended). `tests/test_project.py` enumerates the paths
literally in three places; all three change. *This is the same set of files Stage 11's plan
edits for `analysis` — expect to merge, not to conflict.*

**`src/python_dpo/cli.py`** — the largest edit. The stage bodies named in the exploration table
**move** into `pipeline/stages/*`, and the existing `_cmd_*` handlers import and call them, so
there is exactly one implementation of each stage driven by both entry points. Two new command
groups are added in the established shape (`_add_experiment_parser`, `_add_model_parser`):

```
experiment  preflight | run | resume | status | graph | retry | archive | inspect | reproduce | list
model       package | generate | generate-batch | evaluate | compare | merge | promote | list
```

`experiment run` carries `--config`, `--dry-run`, `--smoke-test`, `--force <stage>`,
`--set key=value` (repeatable), `--resume`. `_PLACEHOLDER_STAGES` and its placeholder handler
are deleted, and `tests/test_project.py`'s assertion about them updated.

**`src/python_dpo/__init__.py`** — `__version__` → `0.12.0`.

**`pyproject.toml`** — no new runtime dependency (PyYAML still suffices; tar, hashlib, subprocess
are stdlib). `testpaths` unchanged.

**Docs** — `src/python_dpo/pipeline/README.md` and `src/python_dpo/packaging/README.md` (new),
plus Stage 12 sections in `README.md` (roadmap table, layout, quickstart), `src/python_dpo/README.md`,
`data/README.md`, `tests/README.md`, `scripts/README.md`, `docker/` and `.claude/plans/README.md`.

---

## New non-Python files

| File | Content |
|---|---|
| `configs/experiments/qwen_python_dpo_v1.yaml` | §8's full experiment configuration, extended with the hypothesis/success-criteria block of §58 and `error_analysis.enabled: false` with an explanatory comment |
| `configs/experiments/template.yaml` | §59's template |
| `configs/experiments/smoke.yaml` | §87's tiny experiment: 3 problems, 2 candidates/problem, 1 training step, 2 evaluation problems |
| `requirements.lock` | §31 — `pip freeze` of the verified `.venv`, committed, with a header naming the Python version and the extras it covers |
| `docker/Dockerfile` | §32 — the **training/inference** image (Python, PyTorch, Transformers, TRL, PEFT, bitsandbytes, Accelerate, Datasets, pytest, the project package). A header states plainly that this is *not* the candidate sandbox and must never be used to execute generated code (§33) |
| `docker/README.md` | The two-container separation of §33, so the distinction is documented rather than folkloric |
| `benchmarks/README.md` | Extended with §56's guardrail |

---

## Tests

New directories `tests/pipeline/` and `tests/packaging/`, all offline and in the default suite.

**`tests/pipeline/`** — `test_config.py` (hierarchy, CLI override precedence, unknown-key
rejection, immutability of `resolved_config.yaml`), `test_stages.py` (topological order,
`dependents_of`, cycle rejection), `test_state.py` (legal and illegal transitions),
`test_manifest.py` (round-trip, schema version, rejection of malformed manifests),
`test_hashing.py` (file, tree, canonical config hash stability under key reordering),
`test_cache.py` (hit on identical inputs; **§91's exact case**: change DPO beta → training/eval
reruns, problem/candidate stay cached), `test_dependencies.py` (missing artifact → `DependencyError`,
no silent reconstruction, §16), `test_orchestrator.py` (dry run executes nothing, §23; ordering;
`--force` cascade, §22), `test_failure.py` (**§88**: failing stage → `FAILED`, downstream
`BLOCKED`, nothing executed against incomplete artifacts), `test_resume.py` (**§89**: interrupt
after preference generation, resume, earlier stages reused and later ones executed),
`test_signals.py` (§67: SIGINT/SIGTERM → state persisted, logs flushed, completed artifacts
intact), `test_preflight.py`, `test_environment.py` (no hostname, no username, no token — §76–§78),
`test_gitinfo.py` (dirty-tree warn vs fail), `test_archive.py`, `test_reproduce.py`,
`test_report.py`, `test_error_analysis_disabled.py` (`SKIPPED` with reason; raises when enabled).

**`tests/pipeline/test_end_to_end.py`** — §87 and §94 offline: three problems, two candidates
each via `MockModelClient`, with the Docker-backed execution stage and the GPU-backed training,
model-evaluation and packaging-verification stages behind injected stub adapters. Asserts the
full nine-stage sequence, a complete manifest, a resolved lineage chain, and `artifacts.json`
hashes matching the files on disk.

**`tests/packaging/`** — `test_package.py` (manifest fields per §35/§37; base model referenced,
not copied), `test_registry.py` (entry schema, atomic write, status set, **no automatic
promotion** — §48 — and `RECOMMENDED` refused without a passing success-criteria record, §47),
`test_merge.py` (the 4-bit guard of §44 raises `MergeUnsupportedError` with an explanation, and
`adapter/` survives a merge), `test_inference.py` (batch JSONL contract), `test_compare.py`.

**`tests/pipeline/test_benchmark_protection.py`** — §92: adding a held-out benchmark problem to
a training split fails.

**Marked tiers** — the real end-to-end orchestration test is marked `@pytest.mark.gpu` and
`@pytest.mark.integration` and stays deselected by default, preserving the zero-skip property.

---

## Execution order

**Phase 1 — foundation (offline, no stage runs).** `config.py` path key, `pipeline/errors.py`,
`config.py`, `stages.py`, `state.py`, `manifest.py`, `hashing.py`, `cache.py`, `artifacts.py`,
`repository.py`, `environment.py`, `gitinfo.py`. Tests for each. At the end of this phase the
graph, cache and manifests are fully testable with no stage executing.

**Phase 2 — orchestration.** Move the stage bodies out of `cli.py` into `pipeline/stages/*`;
`orchestrator.py`; `preflight.py`; `lineage.py`; the `experiment` CLI group; the experiment
configs. Verify with `--dry-run`, `graph`, `preflight`, `status`, and the offline end-to-end test.

**Phase 3 — packaging and the model CLI.** `packaging/*`, `models/registry.json`, the `model`
CLI group, the `packaging` stage adapter. Verify by packaging the existing
`dpo_20260818_081231_a91d` adapter and running §93's load → generate → execute cycle.

**Phase 4 — productionization.** `report.py`, `resources.py`, `cost.py`, `archive.py`,
`reproduce.py`, `requirements.lock`, `docker/Dockerfile`, all documentation.

**Then** the real `--smoke-test` on the RTX 3060, with its artifacts committed.

---

## Verification

```bash
# The offline suite must stay green and skip-free throughout
pytest -q

# §100.1 / §61
python -m python_dpo experiment preflight --config configs/experiments/qwen_python_dpo_v1.yaml

# §100.2 / §62
python -m python_dpo experiment graph --config configs/experiments/qwen_python_dpo_v1.yaml

# §23 — must execute nothing; assert no new run directory appears
python -m python_dpo experiment run --config configs/experiments/qwen_python_dpo_v1.yaml --dry-run

# §100.3 / §24 — the real end-to-end run (GPU + Docker)
python -m python_dpo experiment run --config configs/experiments/qwen_python_dpo_v1.yaml --smoke-test

# §100.4 / §63
python -m python_dpo experiment status --experiment-run-id EXP_RUN_ID

# §90 — a second identical run must reuse, not recompute
python -m python_dpo experiment run --config configs/experiments/qwen_python_dpo_v1.yaml --smoke-test

# §91 — beta change invalidates training onward, leaves generation cached
python -m python_dpo experiment run --config configs/experiments/qwen_python_dpo_v1.yaml \
    --smoke-test --set dpo_training.beta=0.2

# §100.5 / §41 / §93
python -m python_dpo model evaluate --model-package data/experiments/runs/EXP_RUN_ID/model \
    --benchmark python_eval_v1
python -m python_dpo model generate --model-package data/experiments/runs/EXP_RUN_ID/model \
    --prompt "Write a Python function that reverses a list."

# §100.6 / §73
python -m python_dpo experiment archive --experiment-run-id EXP_RUN_ID

# §100.7 / §71, §72
python -m python_dpo experiment reproduce --experiment-run-id EXP_RUN_ID --verify-only
```

Then confirm by inspection:

- `data/experiments/runs/EXP_RUN_ID/` matches the §101 tree, with nine stage manifests.
- Every `sha256` in `artifacts.json` matches `sha256sum` of the file it names.
- `lineage.json` reproduces the model → training → preference → ranking → candidate → dataset chain.
- `resolved_config.yaml` is byte-identical after editing the source config mid-experiment (§10).
- `models/registry.json` shows the model as `EXPERIMENTAL` and **not** promoted (§48).
- `grep -rIn` over the experiment directory finds no token, key, hostname or username (§76–§78).
- `shell=True` still appears nowhere in `src/`, and `tests/sandbox/test_sandbox_security.py` passes (§34).

---

## Deviations to flag in the final report

1. **`problem_generation` → `problem_dataset`.** The curated ten-problem catalog cannot generate
   1,000 problems; `problem_count` selects a subset. Spec §8's example value is unimplementable
   against Stage 2 as built.
2. **`candidate_execution` = Stage 6, `candidate_evaluation` = Stage 7.** The spec's §163 allows
   differing internal names; this mapping keeps nine independently addressable stages.
3. **Packaging code lives in `src/python_dpo/packaging/`,** not §96's `src/python_dpo/models/`,
   which is occupied by the model-client package. `models/registry.json` is still at the root.
4. **Tests stay in the existing per-package layout** (`tests/pipeline/`, `tests/packaging/`)
   rather than §96's `unit/ integration/ smoke/ pipeline/` split, which would relocate every
   existing test file for no behavioural gain.
5. **`error_analysis` ships disabled** pending Stage 11 (decision 1).
6. **The git SHA is excluded from the cache key** for the reason given above.
7. **Deliberate deferrals**, recorded per CLAUDE.md's Scope Control rule: §74's automatic archive
   *restoration* (the spec says the initial implementation need not support it — `inspect` reads
   the manifest only), and §53's LLM API cost accounting, which records its schema with an empty
   provider list because this pipeline calls no external LLM anywhere.
