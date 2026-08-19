# Stage 12 Implementation Details — Pipeline Orchestration, Packaging and Productionization

How `src/python_dpo/pipeline/` and `src/python_dpo/packaging/` implement the layer specified
in `.claude/specs/12_pipeline_orchestration_and_productionization.md`. For usage, see the
"Stage 12 — Pipeline Orchestration" section of the root `README.md`. This file is about
*how* it is built, what the real run found, and what that finding does and does not mean.

## Goal

Eleven stages had produced eleven command groups. Running the pipeline meant reading a run
id out of `runs list`, pasting it into `evaluate run`, reading the next id out of
`evaluations list`, pasting it into `rank`, and so on — six times, with the lineage existing
only as a sequence of manifest hops that nothing recorded and nothing verified.

Stage 12 closes that loop: **one configuration file, one command, one experiment id, and
every artifact traceable to it.** Plus the productionization half — packaging a trained
adapter into something loadable, verifying it by actually executing what it generates, and
registering it without ever promoting it automatically.

## 1. Experiment run

`exp_20260819_060948_f72e` — the real end-to-end run on this machine's RTX 3060 + Docker,
`completed`, 2026-08-19T06:09:52Z → 06:21:13Z (11 min 21 s).

## 2. Stages executed

Eight, all `COMPLETED`, none reused (`error_analysis` was still disabled at the time; Stage
11 landed afterwards and the shipped configs now enable it, taking the graph to nine live
stages):

| Stage | Duration | Underlying run id |
|---|---|---|
| problem_dataset | 0 s | `exp_20260819_060948_f72e_problem_dataset` |
| candidate_generation | 262 s | `run_20260819_060953_1cba` |
| candidate_execution | 187 s | `eval_20260819_061432_5204` |
| candidate_evaluation | 6 s | `rank_20260819_061722_9de4` |
| preference_generation | 3 s | `pref_20260819_061728_849c` |
| dpo_training | 85 s | `dpo_20260819_061731_8314` |
| model_evaluation | 111 s | `eval_20260819_061857_114a` |
| packaging | 24 s | `exp_20260819_060948_f72e_packaging` |

Real GPU generation, real Docker-sandboxed execution, a real LoRA training step, real
base-vs-DPO evaluation, and a packaging verification that loaded the adapter and executed
what it generated.

## 3. The stage graph

Nine `StageSpec` nodes with `requires` edges, topologically ordered by Kahn's algorithm with
ties broken by declaration order so the sequence is deterministic:

```
problem_dataset -> candidate_generation -> candidate_execution
  -> candidate_evaluation -> preference_generation -> dpo_training
       -> model_evaluation -> error_analysis
       -> packaging
```

`packaging` requires `dpo_training`, **not** `model_evaluation` — the spec is explicit that
packaging needs a trained adapter, not a held-out evaluation of it. That makes packaging and
model_evaluation graph-parallel siblings.

Adapters are referenced as dotted `"module:function"` strings rather than imported
callables, so describing the graph never imports a stage submodule (and transitively torch).

## 4. Stage naming deviation

The spec's nine stage names do not map 1:1 onto the repo's commands. Stage 6 (`evaluate
run`) *executes* candidates in Docker **and** runs pytest in one pass; Stage 7 (`rank`)
classifies, scores and orders them. So `candidate_execution` → Stage 6 and
`candidate_evaluation` → Stage 7, preserving nine addressable stages and the spec's intended
meaning: execution produces evidence, evaluation produces judgement.

`problem_generation` is named `problem_dataset` because it is not generation at all —
Stage 2's catalog is ten hand-authored problems with reference solutions. The spec's
`problem_count: 1000` is unimplementable against it; `problem_count` can only select a
subset.

## 5. Cache and invalidation

```
cache_key = sha256(canonical_json({
    stage, input_hashes, configuration_hash, code_version, model_version
}))
```

The cascade is **derived, never tabulated**. Changing DPO beta changes only `dpo_training`'s
config hash → its adapter's output hash changes → `model_evaluation`'s input hashes change →
it reruns → `error_analysis` follows. `problem_dataset` and `candidate_generation` are
structurally unreachable from that change. There is no hand-maintained invalidation table to
drift out of sync.

**Demonstrated on real data, not just in tests.** Rerunning the reference experiment with
only `dpo_training.allow_small_dataset` changed hit cache on all five upstream stages and
reached `dpo_training` in under a second.

**The git SHA is recorded in the manifest but deliberately excluded from the cache key.**
Including it would invalidate every stage on every commit — documentation-only commits
included — defeating the spec's own requirement that a hyperparameter change leave problem
and candidate generation cached. `--force` and `--set` are the explicit invalidation levers.

## 6. Persistence layout

```
data/experiments/runs/exp_20260819_060948_f72e/
    manifest.json            experiment_manifest_v1
    resolved_config.yaml     immutable after start
    environment.json         OS, CUDA, GPU, package versions, Docker
    artifacts.json           8 entries: path + sha256 + bytes
    lineage.json             the resolved chain
    stages/<stage>/stage_manifest.json    (nine directories)
    model/{adapter,tokenizer,manifest.json}
    reports/{experiment_metrics.json,experiment_summary.md,
             model_comparison.md,next_experiment.md}
```

Stage outputs stay in their canonical `data/<stage>/runs/` stores; the experiment directory
holds manifests, SHA-256 pointers, reports and the packaged model. No gigabyte-scale
duplication and no second source of truth. The one deliberate copy is `model/adapter/`,
because the packaged artifact is meant to be usable standalone.

## 7. Artifact manifest

Eight entries, each a path + SHA-256 + byte count:

| Artifact | Bytes | SHA-256 (12) |
|---|---|---|
| problem_dataset | 19,028 | `c79836f64eb2` |
| candidate_generation | 157,838 | `b5961be32e9c` |
| candidate_execution | 215,959 | `1d57d33e4f3a` |
| candidate_evaluation | 67,057 | `c267e99c7ec9` |
| preference_generation | 91,595 | `c943754870f0` |
| dpo_training | 55,752,427 | `93b4927273c0` |
| model_evaluation | 73,812 | `12f0da9cee0e` |
| packaging | 26,225,237 | `c2a16aca7792` |

## 8. Lineage

```json
{
  "model_adapter": {
    "problem_dataset_run_id": "exp_20260819_060948_f72e_problem_dataset",
    "candidate_run_id":       "run_20260819_060953_1cba",
    "evaluation_run_id":      "eval_20260819_061432_5204",
    "ranking_run_id":         "rank_20260819_061722_9de4",
    "preference_run_id":      "pref_20260819_061728_849c",
    "training_run_id":        "dpo_20260819_061731_8314"
  },
  "model_evaluation_run_id": "eval_20260819_061857_114a",
  "packaging_run_id":        "exp_20260819_060948_f72e_packaging"
}
```

The chain that previously existed only as a sequence of manifest hops is now recorded as a
fact.

## 9. State machines

Two, kept independent. The experiment level uses `created → running →
completed/failed/interrupted/cancelled`. The stage level adds
`PENDING/RUNNING/COMPLETED/FAILED/SKIPPED/CANCELLED/BLOCKED`, layered *on top of* the six
existing per-stage run repositories rather than replacing them — each stage manifest records
the underlying run's own id.

A failing stage is recorded `FAILED` with its error; every downstream stage becomes
`BLOCKED` and is never executed against incomplete artifacts. A disabled stage is persisted
`SKIPPED` **with a reason**, never silently omitted.

## 10. Signal handling

`SIGTERM` is made to raise `KeyboardInterrupt` so both signals take one path: through
whatever handling the running stage adapter already has (persisting its own run's state),
then up to the orchestrator, which marks the in-flight stage `CANCELLED` and the experiment
`interrupted`. Already-completed stages' artifacts are untouched.

## 11. Cost and resources

GPU-hours are derived from each GPU-using stage's own recorded wall clock — no separate
timer to keep in sync:

```
total 0.1339 GPU-hours
  candidate_generation 0.0728 · model_evaluation 0.0308
  dpo_training 0.0236 · packaging 0.0067
```

The LLM-API cost schema is recorded with an explicit `providers: []` rather than omitted:
this pipeline calls no external LLM anywhere, and the empty list is a checked fact rather
than a gap.

## 12. Packaging

`model/` = `adapter/` + `tokenizer/` + `manifest.json` (`model_package_v1`). The **base model
is referenced by name and revision, never copied** — it is tens of gigabytes and already in
the HF cache.

Packaging lives in `src/python_dpo/packaging/` rather than the spec's
`src/python_dpo/models/`, which is already the model-*client* package (`ModelClient`,
`QwenModelClient`, `MockModelClient`). `models/registry.json` still lands at the project
root as specified.

## 13. Package verification

Not optional and with no `--skip-verification` escape hatch. The packaged adapter is loaded,
asked to write a function, and the code it produces is **executed through the Stage 5/6
Docker sandbox**; the package is registered only if the tests pass.

Recorded for the real run: `2/2` tests passed, generating a working `add_two`. A verification
failure raises before anything is registered.

## 14. Model registry

`models/registry.json` (`registry_v1`), atomic writes. One entry:

```
exp_20260819_060948_f72e  EXPERIMENTAL  training=dpo_20260819_061731_8314  verified 2/2
```

**Nothing is promoted automatically.** `register()` only ever writes `EXPERIMENTAL`, and the
status machine makes `RECOMMENDED` reachable only via `VALIDATED` — so a model can never be
recommended sight-unseen. `promote(..., "RECOMMENDED")` additionally requires an
`evaluation_run_id` *and* a passing recorded success-criteria record, and the CLI reads
`DPO_SUCCESS` out of the Stage 10 report rather than taking the caller's word.

## 15. CLI surface

```
experiment  preflight | graph | run | resume | retry | status | list
            archive | inspect | reproduce
model       package | generate | generate-batch | evaluate | compare
            merge | promote | list
```

`experiment run` carries `--config`, `--dry-run`, `--force <stage>`, `--set key=value`
(repeatable), `--smoke-test`. The `_PLACEHOLDER_STAGES = {"run": ...}` entry that had sat in
`cli.py` since Stage 1 was reserved for exactly this and is now removed.

## 16. Stage bodies moved, not duplicated

The stage implementations lived inside `cli.py`'s private helpers (`_execute_run`,
`_run_evaluation`, `_rank_problem_group`, `_finalize_preference_run`, and the 200-line
`_cmd_preferences_generate` / `_cmd_train_dpo` / `_cmd_evaluate_model_run`). They **moved**
into `pipeline/stages/*`, and `cli.py` imports them — one implementation per stage, driven
by both entry points. The orchestrator neither shells out to `python -m python_dpo` nor
fabricates an `argparse.Namespace`.

The same pattern applies to packaging: `package_and_verify()` is the single body called by
both the `packaging` stage adapter and the `model package` command.

## 17. Benchmark protection

`model_evaluation` **requires an existing benchmark manifest** and fails clearly if one is
missing; it builds one only when `build_benchmark_if_missing` is explicitly set, and never
overwrites an existing manifest. Leakage is re-checked before every evaluation using the
training run's own recorded preference run, so there is no way to skip the check by omitting
a setting.

**This fired for real.** A later run reached `model_evaluation` and stopped: `p005` bore
preference pairs and is also a `python_eval_v1` benchmark problem, so training on it would
have contaminated the held-out set. The guard refused rather than reporting a compromised
number.

## 18. Two bugs the real run surfaced

Neither was reachable from the test suite; both needed an actual end-to-end execution.

- **`experiment_summary.md` reported `Status: running` on a completed experiment.** The
  report was built from a manifest snapshot taken *before* `complete_run()` transitioned the
  status, so every report would have shown `running` and `Ended: -`. Fixed by completing the
  run before rendering reports.
- **Packaging copied ~40 MB of dead weight into every package.** `build_package` copied the
  whole adapter directory, including TRL's frozen reference adapter (`ref/`, ~29 MB, never
  loaded by anything) and a duplicate `tokenizer.json` (~11 MB) — the exact files
  `.gitignore` already excludes for training runs. The packaged model dropped from 67 MB to
  26 MB. A regression test now asserts both are excluded.

## 19. Files created/modified

**Created — `src/python_dpo/pipeline/` (4,797 lines):** `__init__.py`, `errors.py`,
`config.py`, `stages/` (registry + `_context.py` + nine adapters), `state.py`, `manifest.py`,
`hashing.py`, `cache.py`, `lineage.py`, `artifacts.py`, `environment.py`, `gitinfo.py`,
`preflight.py`, `resources.py`, `cost.py`, `report.py`, `archive.py`, `reproduce.py`,
`orchestrator.py`, `repository.py`.

**Created — `src/python_dpo/packaging/` (1,107 lines):** `__init__.py`, `errors.py`,
`package.py`, `verify.py`, `merge.py`, `registry.py`, `inference.py`, `compare.py`,
`pipeline_stage.py`.

**Created — `tests/pipeline/` + `tests/packaging/` (3,436 lines, 240 tests).**

**Created — other:** `configs/experiments/{qwen_python_dpo_v1,smoke,template}.yaml`,
`requirements.lock`, `docker/Dockerfile`, `models/registry.json`, `data/experiments/`.

**Modified:** `src/python_dpo/cli.py` (the `experiment` and `model` groups; stage bodies
moved out; `_PLACEHOLDER_STAGES` removed), `src/python_dpo/config.py` + `config.yaml`
(`experiments` as the tenth `paths` entry), `src/python_dpo/__init__.py` (`0.10.0` →
`0.12.0`), `.gitignore`, `tests/test_project.py`, `tests/test_no_heavy_imports.py`,
`tests/{evaluation,sandbox}/test_config.py`.

## 20. Dependencies added

**None.** tar, hashlib, subprocess and json are stdlib; PyYAML was already required.
`pyproject.toml`'s dependency list is unchanged. `requirements.lock` is a `pip freeze` of the
verified `.venv` — the exact stack that trained and packaged on this machine, not a resolved
set of floors.

## 21. Deviations from the specification

- **`candidate_execution` = Stage 6, `candidate_evaluation` = Stage 7** (see §4); the spec
  allows differing internal names.
- **`problem_generation` → `problem_dataset`**, and `problem_count` selects a subset of the
  curated catalog rather than generating problems (see §4).
- **Packaging lives in `src/python_dpo/packaging/`**, not the spec's
  `src/python_dpo/models/`, which is occupied (see §12).
- **`error_analysis` shipped registered-but-disabled** because Stage 11 did not exist when
  Stage 12 was built. Its adapter raised `StageNotImplementedError` if enabled and its state
  persisted as `SKIPPED` with a reason. **Now resolved** — Stage 11 landed afterwards and the
  adapter is real.
- **The git SHA is excluded from the cache key** (see §5).
- **Tests stay in the existing per-package layout** (`tests/pipeline/`, `tests/packaging/`)
  rather than the spec's `unit/ integration/ smoke/ pipeline/` split, which would relocate
  every existing test file for no behavioural gain.
- **Archive *restoration* is not implemented.** `experiment archive` writes a tar.gz with an
  embedded `archive_manifest.json`, and `experiment inspect --archive` reads that one member
  without extracting; restoring an archive back into a live run is out of scope, as the spec
  itself permits for an initial implementation.
- **LLM API cost accounting records its schema with an empty provider list** rather than
  implementing per-provider accounting, since no external LLM is called anywhere.
- **Resource capture is a host snapshot at report time, not a per-stage trace.** Sampling GPU
  utilization continuously through every stage would need a background poller this stage does
  not build. Durations are exact; utilization is best-effort and `None` when unavailable.
- **The training/inference Dockerfile is not Docker-in-Docker.** It carries the project and
  its dependencies but no `docker` CLI, so running the orchestrator *inside* it (which shells
  out to Docker for candidate execution) needs the host socket mounted and the CLI installed
  on top.
- **Documentation was written selectively.** `docker/Dockerfile` carries the two-container
  security explanation inline; the per-package `README.md` files the plan listed were not
  created.

## 22. Known limitations

- **The real end-to-end run is a smoke test, not an experiment.** Three problems, two
  candidates each, one optimizer step, a two-problem evaluation slice. It proves the
  apparatus works; it says nothing about model quality, and the reports it produced should be
  read that way.
- **The reference experiment has never completed end to end.** Its two attempts both stopped
  at guards that were correct to fire — the preference splitter needs ten pair-bearing
  problems and this dataset yields three, and the only pair-bearing problem inside the
  benchmark (`p005`) would have leaked. Both stops are committed as evidence. Completing it
  needs more problems, not more code.
- **Cross-experiment cache reuse is unproven at scale.** It demonstrably works (§5), but only
  across runs minutes apart on one machine. Nothing has tested a cache hit against a run from
  a different checkout or a different day.
- **`experiment reproduce --verify-only` compares what is persisted, not everything.** Config
  hash, model identity, dataset hash and selected environment keys are diffed; it cannot
  detect a changed problem *definition* that preserved the dataset hash, or a dependency
  whose version string did not change.
- **`model merge` is untested against a real merge.** The 4-bit guard and the never-delete-
  the-adapter behaviour are covered, but the actual `merge_and_unload()` path has only ever
  run against a faked backend — no full-precision merge has been performed on this hardware.
- **`experiment archive` has no size guard.** Archiving a run whose `model/` carries an
  adapter will happily produce a multi-megabyte tarball; nothing warns or streams.
- **A seventh copy of the run-directory plumbing**, rather than the shared base deferred
  since Stage 7. Extracting it would now touch eight stages at once; the debt is carried
  deliberately and re-flagged in `repository.py`'s docstring.

Do NOT implement the next pipeline stage automatically. Wait for explicit approval.
