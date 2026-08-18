# Stage 10 Implementation Details — Base vs DPO Model Evaluation

How `src/python_dpo/model_evaluation/` implements the layer specified in
`.claude/specs/10_model_evaluation.md`. For usage, see the "Stage 10 — Base vs DPO Model
Evaluation" section of the root `README.md`. This file is about *how* it is built, what it
found, and what that finding does and does not mean.

## Goal

Every stage so far deferred the same question. Stage 10 finally asks it: did DPO actually
make Qwen better at Python programming? Base Qwen and base+adapter are generated against
the identical held-out benchmark, prompts, generation config and seeds, evaluated through
the unmodified Stage 5/6 sandbox, and compared with pass@k, bootstrap confidence intervals,
and win/tie/loss — never an LLM judge, never a benchmark consulted to tune anything, never
an automatic promotion.

## 1. Benchmark version

`python_eval_v1`.

## 2. Benchmark hash

`600953284286b946508c563b2b47a52a48b1572bfb2d911a4081e5fb8a2d6d31` — SHA-256 over the
canonical JSON of the 7 selected problems, recomputed and verified by `benchmark validate`
before every run.

## 3. Number of benchmark problems

7: `p001, p002, p003, p004, p005, p006, p009` — every problem never assigned to a Stage 8
preference split for `pref_20260818_074347_5eff` (`p007`/`p008` trained, `p010` validated).
`benchmark check-leakage` confirmed zero overlap.

## 4. Base model

`Qwen/Qwen2.5-Coder-3B-Instruct`.

## 5. DPO model

`Qwen/Qwen2.5-Coder-3B-Instruct` + the Stage 9 LoRA adapter, loaded with PEFT.

## 6. Adapter

`data/training/runs/dpo_20260818_081231_a91d/adapter` (training run `dpo_20260818_081231_a91d`).
Integrity-checked before every load: directory present, `adapter_config.json` present and
parseable, a weights file present, and `base_model_name_or_path` matching the configured
base model exactly.

## 7. Model revisions

`null` for both variants (the default branch), taken directly from the Stage 9 training
manifest rather than a separately configured value — so base-model integrity (spec section
14) is satisfied by construction, not by a runtime comparison that could drift.

## 8. Generation configuration

Identical for both variants, from `configs/evaluation/python_eval.yaml`:

| | |
|---|---|
| temperature | 0.2 |
| top_p | 0.95 |
| max_new_tokens | 512 |
| do_sample | true |
| repetition_penalty | 1.0 |
| base_seed | 1000 |
| quantization | 4-bit NF4, double quant, bfloat16 compute — same `QuantizationSettings` object feeds both runners |

## 9. Number of samples

10 per problem per variant — 140 generations total (7 problems × 10 samples × 2 models).

## 10. Random seeds

`seed = base_seed + problem_index*1000 + sample_index`, with `problem_index` over the
benchmark's problems sorted by id. Verified byte-identical between variants:

```bash
diff <(jq -c '{problem_id,sample_index,seed}' generations/base.jsonl | sort) \
     <(jq -c '{problem_id,sample_index,seed}' generations/dpo.jsonl | sort)
# empty
```

## 11–16. pass@k

| k | Base | DPO |
|---|------|-----|
| 1 | 77.14% | 78.57% |
| 5 | 85.66% | 85.66% |
| 10 | 85.71% | 85.71% |

pass@5/pass@10 are identical to 2+ decimal places — expected, since with 10 samples both
metrics approach "did any sample solve it," and base and DPO solve the exact same set of
problems at that level (see §22–24).

## 17. Test pass rates

Base: 90.23%. DPO: 90.23%. Identical to 4 significant figures — the mean per-candidate
`tests_passed/tests_total` across all 140 candidates.

## 18. Syntax success rates

100% for both. Every one of the 140 generated candidates parsed.

## 19. Timeout rates

0% for both. No candidate exceeded the sandbox's timeout budget.

## 20. Runtime error rates

0% for both — `import_error` and `runtime_error` are both zero in the failure analysis for
both variants; every non-passing candidate failed on a plain assertion, not an exception.

## 21. Generation failure rates

0% for both — every one of the 140 generations produced extractable, parseable code. No
`generation_error` record exists in either `generations/{base,dpo}.jsonl`.

## 22. DPO wins

0.

## 23. Ties

7 (all 7 problems).

## 24. Base wins (DPO losses)

0.

Win/tie/loss is computed at the problem level (spec section 54): a problem is "solved" if
*any* of its 10 samples passed every test. Per-problem correct-sample counts (`c` of `n=10`):

| Problem | Base c/n | DPO c/n |
|---|---|---|
| p001 | 10/10 | 10/10 |
| p002 | 0/10 | 0/10 |
| p003 | 10/10 | 10/10 |
| p004 | 5/10 | 5/10 |
| p005 | 9/10 | **10/10** |
| p006 | 10/10 | 10/10 |
| p009 | 10/10 | 10/10 |

Both variants solve the same 6 problems (`p001, p003, p004, p005, p006, p009`) and both
fail the same one (`p002`, identically at 0/10 — every sample on both sides fails the same
subset of tests, a structural miss rather than noise). No problem flips from failing to
passing or vice versa, hence 0 wins / 0 losses. The entire §25 pass@1 movement (+1.4pp) is
exactly **one flipped sample on `p005`** (base 9/10 → DPO 10/10): `(10+0+10+5+10+10+10)/10
− (10+0+10+5+9+10+10)/10, both ÷ 7 = +1.4pp`. One sample, out of 140, is the entire
measured effect.

## 25. Absolute improvement

pass@1: **+1.4 percentage points** (77.14% → 78.57%). pass@5/pass@10: 0.0pp.

## 26. Relative improvement

pass@1: **+1.85%** relative ((0.7857 − 0.7714) / 0.7714).

## 27. Bootstrap confidence intervals

1000 iterations, seed 42, resampled at the **problem level** (7 problems, with
replacement):

| | Base | DPO |
|---|------|-----|
| pass@1 95% CI | [48.6%, 98.6%] | [50.0%, 100.0%] |
| pass@5 95% CI | [57.1%, 100.0%] | [57.1%, 100.0%] |
| pass@10 95% CI | [57.1%, 100.0%] | [57.1%, 100.0%] |

Paired bootstrap, pass@1 difference (DPO − Base): **+1.4pp, 95% CI [+0.0, +4.3] pp**. The
lower bound sits at exactly zero — the data does not rule out "no difference," which given
a one-optimizer-step adapter is exactly the honest answer.

## 28. Statistical test results

McNemar's exact test over discordant problem-level outcomes: base-only = 0, DPO-only = 0,
**p = 1.0000** (zero discordant pairs — there is nothing for the test to detect at this
level, consistent with §24).

## 29. Improvement examples

**None.** `reports/improvements.jsonl` is empty (0 rows) — no problem went from
base-fails/DPO-solves.

## 30. Regression examples

**None.** `reports/regressions.jsonl` is empty (0 rows) — no problem went from
base-solves/DPO-fails. All 7 problems are recorded in `reports/ties.jsonl`, one row each
with both models' first-sample code and test result for qualitative inspection.

## 31. Failure analysis

| | Base | DPO |
|---|------|-----|
| generation_error | 0 | 0 |
| syntax_error | 0 | 0 |
| import_error | 0 | 0 |
| runtime_error | 0 | 0 |
| assertion_failure | 16 | 15 |
| timeout | 0 | 0 |
| infrastructure_error | 0 | 0 |

Every failure on both sides is a plain wrong-answer assertion failure, concentrated in
`p002` (which fails identically at 3/7 tests on every one of the 10×2 samples — a
structural miss in the model's understanding of that problem, not noise) and partial
misses on `p004`.

## 32. Peak GPU memory

Base: 2.00 GiB (2,147,642,368 bytes). DPO: 3.37 GiB (3,621,000,192 bytes). The ~1.3 GiB
difference is the PEFT wrapper plus adapter weights held alongside the frozen 4-bit base —
expected, and not a regression signal in itself.

## 33. Inference latency

| | Base | DPO |
|---|------|-----|
| mean | 1420 ms | 2183 ms |
| p50 | 1233 ms | 1914 ms |
| p95 | 2433 ms | 3826 ms |

## 34. Tokens/second

Base: 47.2 tok/s. DPO: 30.5 tok/s. The adapter's forward pass through the additional LoRA
layers costs real throughput — secondary to correctness, but recorded per spec section 76.

## 35. Dataset leakage result

**None detected.** `benchmark check-leakage --benchmark python_eval_v1 --preference-run-id
pref_20260818_074347_5eff` reported "No problem leakage detected." — verified again here:

```bash
jq -r '.problem_ids[]' benchmarks/python_eval_v1/manifest.json | grep -E 'p007|p008|p010'
# no output
```

## 36. Evaluation reproducibility information

- `bootstrap_seed = 42`, `bootstrap_iterations = 1000`, persisted in
  `metrics/pass_at_k.json` / `metrics/bootstrap.json`.
- `generation.base_seed = 1000`, and the full per-`(problem, sample)` seed schedule is
  reconstructable via `compute_seed` and independently verifiable from
  `generations/{base,dpo}.jsonl` (§10).
- The benchmark's `dataset_hash` is stored in `benchmark_manifest.json` inside the run
  directory, so a later `problems.jsonl` edit cannot silently reinterpret this run's result.
- `evaluate-model validate --evaluation-run-id eval_20260818_155511_1633` passes: benchmark
  integrity, model identity, adapter identity, prompt/seed pairing, and candidate-count
  completeness all check out.
- `git diff --stat data/problems/ data/candidates/ data/evaluations/ data/rankings/ data/preferences/ data/training/`
  is empty — nothing upstream was touched by this stage.

## 37. Final recommendation

**`DPO_SUCCESS: False`.** Clause by clause (spec section 143):

| Clause | Result |
|---|---|
| pass@1 improves by ≥ 2pp | **False** (+1.4pp, below the 2pp gate) |
| pass@5 not regressed by > 2pp | True |
| syntax success not regressed by > 2pp | True |
| timeout rate not increased by > 2pp | True |
| paired CI does not strongly support a regression | True (CI is [+0.0, +4.3]pp, entirely ≥ 0) |
| catastrophic regression detected | False |

No automatic model promotion occurs, in either direction — this is evidence for a later
model-selection stage to consult, not a deployment decision. The honest reading: a
single-optimizer-step adapter shows a movement indistinguishable from sampling noise on a
7-problem benchmark. This says nothing about whether DPO *would* work with real training —
only that Stage 10's apparatus correctly detected "not enough signal to call it" rather
than manufacturing a success.

## 38. Files created/modified

**Created:**

- `src/python_dpo/model_evaluation/` — `__init__.py`, `errors.py`, `config.py`,
  `benchmark.py`, `models.py`, `metrics.py`, `statistics.py`, `comparison.py`,
  `runners.py`, `generation.py`, `evaluation.py`, `cache.py`, `run_repository.py`,
  `report.py`, `README.md`
- `tests/model_evaluation/` — nine offline test modules, `test_docker_integration.py`,
  `test_gpu_integration.py`
- `configs/evaluation/python_eval.yaml` — the evaluation experiment configuration
- `benchmarks/python_eval_v1/manifest.json`, `benchmarks/README.md`
- `data/model_evaluations/runs/eval_20260818_155356_23d4/` — the smoke-test run (3
  problems, 1 sample, both models)
- `data/model_evaluations/runs/eval_20260818_155511_1633/` — the full run (7 problems, 10
  samples, both models) — the evidence behind every number in this report
- `10_MODEL_EVALUATION.md` (this file)

**Modified:**

- `src/python_dpo/config.py` + `config.yaml` + `data/model_evaluations/.gitkeep` — the
  ninth data path (`model_evaluations`), threaded through `_REQUIRED_PATH_KEYS`, `Paths`
  and `ensure_exists()`
- `src/python_dpo/cli.py` — the `benchmark` and `evaluate-model` command groups
- `tests/test_no_heavy_imports.py` — every `model_evaluation` submodule added to the probe
- `tests/test_project.py` — the ninth data path in all three enumerations; `benchmark`/
  `evaluate-model` CLI parsing and error-path tests
- `tests/sandbox/test_config.py`, `tests/evaluation/test_config.py` — their fixture YAML
  gained the required `paths.model_evaluations` key
- `src/python_dpo/__init__.py` — version `0.9.0` → `0.10.0`
- `README.md`, `src/python_dpo/README.md`, `data/README.md`, `tests/README.md`

## 39. Dependencies added

**None.** pass@k, bootstrap confidence intervals and McNemar's test are hand-rolled in
pure stdlib (`math.comb`, `random.Random`), preserving PyYAML as the project's only
runtime dependency. `pyproject.toml` is unchanged. Inference reuses the existing
`training` extra (torch, transformers, peft, bitsandbytes) — installing nothing new was
required to run the real evaluation on this machine.

## 40. Deviations from the specification

- **The benchmark is the 7 never-trained-on problems, not a single formally designated
  test problem.** Stage 8's splitter only assigns *pair-bearing* problems to
  train/validation/test; six problems produced no preference pairs at all and are equally
  untrained and equally valid as held-out data.
- **5 of the 7 benchmark problems sit at the base model's ceiling** (confirmed again by
  this run: `p001, p003, p006, p009` solved 10/10 by both variants, and `p005` 9/10 base
  vs. 10/10 DPO — i.e. still at or within one sample of ceiling). `p002` sits at a
  **floor** (0/10, both variants) and `p004` in between (5/10, both). Reported as a
  first-class headroom analysis rather than buried. Selecting only the 2 problems with
  headroom was explicitly rejected as benchmark contamination (spec sections 134, 135).
- **The benchmark manifest references problem ids and hashes their content** rather than
  snapshotting the problems — `problems.jsonl` stays the single source of truth, with
  drift detected by hash rather than prevented by duplication.
- **A new inference layer (`runners.py`) was required.** `QwenModelClient` supports
  per-call seeding but neither adapters nor quantization, and `ModelConfig` actively
  rejects any non-null `quantization`; `training/verify.py` loads adapters in 4-bit but is
  greedy-only and unseeded. Neither is reusable for seeded k-sample pass@k.
- **Candidates are constructed in memory and never persisted through Stage 4.**
  `CandidateEvaluator` needs a `Candidate` object but never reads the candidate
  repository, so Stage 10 reuses Stage 6's execution and classification without
  inheriting Stage 4's run plumbing.
- **Per-variant evaluation repositories** under `evaluations/_sandbox/{base,dpo}/`,
  because `evaluate_many`/candidate-id-keyed resume would otherwise collapse the 10
  samples per problem into one entry, and would also violate spec section 94's
  no-cross-model-cache-collision rule.
- **pass@k, bootstrap and McNemar are hand-rolled in pure stdlib** rather than adding
  numpy/scipy (§39).
- **`evaluate-model` is a command group with a default action**, so the bare
  `--benchmark ... --training-run-id ...` form and the `validate`/`report`/`stats`/
  `compare`/`list` subcommands coexist on one parser.
- **`evaluate-model list` was added** beyond the specified subcommands, since none of them
  is usable without a way to discover an `evaluation_run_id`.
- **A sixth copy of the run-directory plumbing**, rather than the shared base deferred at
  Stages 7, 8 and 9. Extracting it would now touch six stages at once.
- **`reports/{improvements,regressions,ties}.jsonl` are overwritten, not appended**, on
  every `report`/`stats`/`compare` invocation — they are recomputed from persisted
  generation/evaluation records each time, and an append-only write would duplicate rows
  on a second invocation. Written even when empty, so the file's presence is never
  ambiguous (caught and fixed during this implementation, before being committed).
- **Cross-run generation/evaluation caching (spec sections 92, 93, 138) is available as
  key infrastructure (`cache.py`) but not wired into the CLI's automatic flow.** The
  per-run resume mechanism (skip already-persisted `(problem_id, sample_index)` pairs) is
  what actually runs; reusing a cached Base Qwen baseline across multiple DPO experiments
  would need a follow-up CLI flag pointing a new run at a prior run's cache file. Recorded
  here rather than built speculatively, per CLAUDE.md's Scope Control rule.
- **The overfitting check (spec section 70)** is reported as `not_applicable` — it needs
  training-set performance, which spec section 69 forbids evaluating in this stage.

## 41. Known limitations

- **7 problems is a pipeline-validation benchmark, not a statistically meaningful one**
  (spec section 144). The paired bootstrap CI for the pass@1 difference is 4.3 percentage
  points wide at the upper end alone — any result this benchmark produces should be read
  as "the apparatus works," not "the model is better/worse."
- **The adapter trained for a single optimizer step** (Stage 9's own limitation, inherited
  here). A `DPO_SUCCESS = False` verdict on this adapter is expected and uninformative
  about whether DPO *as a method* would improve this model with real training — that
  question needs a properly trained adapter run through this same, now-proven, apparatus.
- **5 of 7 problems are at the base model's ceiling**, so this benchmark can currently
  only detect regression, not improvement, on the majority of its problems. A benchmark
  with more headroom (spec section 145 recommends 500-1,000+ problems) is needed before a
  `DPO_SUCCESS = True` verdict would mean much even if it occurred.
- **Cross-run caching is unused in practice.** Every evaluation run regenerates the base
  model's outputs from scratch, even though spec section 138 permits reusing a prior
  Base Qwen result across multiple DPO experiments. At ~14 minutes per full run on this
  hardware this is a convenience cost, not a correctness one.
- **The regression-problem-list qualitative workflow (spec section 133) is unexercised.**
  `reports/regressions.jsonl` and `reports/improvements.jsonl` are empty for this run by
  construction (zero wins, zero losses) and have not been validated against a run that
  actually produces rows in them.
- **Latency/throughput/memory numbers are single-run measurements on one RTX 3060**, not
  averaged across repeated runs — useful as a sanity check (DPO costs more due to the
  adapter forward pass) but not a rigorous performance benchmark.

Do NOT implement Step 11 automatically. Wait for explicit approval before implementing the
next pipeline stage.
