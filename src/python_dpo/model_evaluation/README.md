# src/python_dpo/model_evaluation/

Stage 10 — answers, with sandboxed execution evidence rather than training-loss numbers,
whether the Stage 9 DPO adapter actually improved Python programming correctness over the
base Qwen model on held-out problems.

Four rules govern the package:

- **The benchmark is never used to tune anything.** Not beta, not LoRA rank, not epochs,
  not the preference policy. Consulting it to change any of those means a new experiment,
  not a better number (spec sections 134, 135).
- **Correctness comes from execution, not judgement.** Generated code goes through the
  Stage 5 sandbox and the Stage 6 executor, unmodified. No LLM judge anywhere.
- **The comparison isolates the adapter.** Same prompt, same chat-template rendering, same
  seeds, same quantization. The only difference between the two runs is base weights
  versus base + adapter.
- **No automatic promotion.** This stage produces evidence and a recommendation against
  configurable criteria (`DPO_SUCCESS`). It never declares a production model.

**Import discipline carries over from Stage 9.** Every torch/transformers/peft touch in
`runners.py` is deferred into the function that needs it; `tests/test_no_heavy_imports.py`
covers this package too.

## Files

### `errors.py`

`ModelEvaluationError` and its subclasses: `BenchmarkError`/`BenchmarkLeakageError` (a
contaminated or drifted benchmark), `ModelIdentityError`/`AdapterIntegrityError` (the
model actually loaded does not match what was configured), `PairingError`/
`IncompleteBenchmarkError` (base and DPO do not describe the same paired experiment), and
the run-plumbing errors mirroring every prior stage.

### `config.py`

The `configs/evaluation/python_eval.yaml` schema — a **standalone file**, for the same
reason as Stage 9's `dpo_qlora.yaml`: `models.base.ModelConfig` rejects a non-null
`quantization`, so 4-bit inference cannot live in the root `config.yaml`, and a second
experiment should be a second file. `EvaluationExperimentConfig.__post_init__` enforces
spec section 43: requesting `pass@10` with `num_samples < 10` is a configuration error,
not a silently-degraded metric.

### `benchmark.py`

The held-out benchmark. `problems.jsonl` stays the single source of truth —
`BenchmarkManifest` references problem ids and a SHA-256 over their canonical content,
never a snapshot. `build_benchmark` selects; `load_benchmark` recomputes the hash and
raises on drift, which is what makes "the benchmark must not change between model
comparisons" (spec section 91) a real guarantee instead of a stated intention.
`check_leakage` asserts the benchmark is disjoint from a preference run's train and
validation splits, raising `BenchmarkLeakageError` naming every offending id — a
contaminated benchmark stops the run, it does not warn.

### `runners.py`

`BaseModelRunner` and `AdapterModelRunner` — separate classes, not one class with an
`adapter_dir=None` flag, so spec section 118's isolation guarantee is visible in the type
itself: `BaseModelRunner` has no adapter-loading code path at all. Both are built from the
same `QuantizationSettings`, so "identical quantization for both models" (spec section 19)
is structural. `verify_adapter_integrity` (spec section 15) is a standalone function —
adapter directory, `adapter_config.json`, weights file, and the adapter's recorded base
model all checked — so `evaluate-model validate` can run it without touching torch, and
`AdapterModelRunner.ensure_loaded` calls the same function rather than a private copy.
Never falls back to the base model on any integrity failure.

### `generation.py`

`GenerationDriver.run` generates every `(problem, sample_index)` pair for one runner.
`compute_seed(base_seed, problem_index, sample_index)` is what makes the comparison
**paired**: both variants are driven through the identical schedule, so
`base.seed[i] == dpo.seed[i]`. The prompt is Stage 9's canonical, strategy-free prompt
(no per-variant prompt engineering, spec section 29); extraction reuses Stage 3's
`extract_code` unmodified for both variants. A failed extraction becomes
`status="generation_error"` with the raw response retained and `extracted_code=None` — it
never reaches `evaluation.py`, since a generation failure has no code to run.

### `evaluation.py`

Wraps Stage 6 rather than reimplementing it: for each generated candidate, an in-memory
`Candidate` is built (never persisted through Stage 4) and handed to
`CandidateEvaluator.evaluate`, unmodified — so Stage 10 cannot drift from Stage 6 about
what "passed" means. Before building the candidate, the canonical prompt is rebuilt from
the problem and hashed again, and checked against the hash recorded at generation time —
spec sections 27/114's prompt-equality claim is verified from the artifacts, not merely
asserted in code. `_classify_error_type` maps each non-passing result onto spec section
124's taxonomy (`syntax_error`/`import_error`/`runtime_error`/`assertion_failure`/
`timeout`/`infrastructure_error`) deterministically, using the Stage 6 test-level error
types already available — no LLM.

### `metrics.py`

Pure stdlib. `pass_at_k(n, c, k)` is the unbiased estimator,
`1 - C(n-c, k) / C(n, k)`, with the `c >= n - k + 1` and `c == 0` shortcuts spec section 42
describes. Spec section 44 is explicit that `c / n` is **not** pass@k, and section 111
gives the boundary values this module is tested against. `mean_pass_at_k` aggregates
**per-problem then averages** — never pooling every candidate across problems into one
`(n, c)`, which section 110 forbids. The other rate functions (syntax success, execution
success, timeout, generation failure) and the section 53 seven-bucket test-failure
distribution are all here too, all taking plain lists of bools/floats so they are testable
against the spec's hand-computed examples directly.

### `comparison.py`

`compare()` turns two variants' `EvaluationRecord` lists into a win/tie/loss verdict.
`paired_problem_set` (spec section 122) is computed first and is the basis of every
headline number — only problems successfully evaluated for **both** variants participate.
A mismatched set is an error (`IncompleteBenchmarkError`), never a silent narrowing,
unless `allow_incomplete=True` is passed explicitly. Per problem: `base_pass`/`dpo_pass`,
`improvement ∈ {+1, 0, -1}`, and `test_pass_delta` — the last of which is how partial
improvement shows up even when neither model solves the problem outright.
`dpo_win_rate = wins / (wins + losses)`, ties excluded from the denominator (spec section
56).

### `statistics.py`

Pure stdlib (`random.Random`, `math.comb`) — no numpy or scipy, matching
`training/lengths.py`'s percentile precedent. `bootstrap_ci` resamples **problems** with
replacement (spec section 60), never candidates. `paired_bootstrap` resamples the same
problem indices for base and DPO in each iteration and reports the distribution of
*differences*, which is what makes it paired rather than two independent intervals.
`mcnemar` is the exact two-sided binomial test over discordant pairs via `math.comb`, not
the chi-square approximation, which is unreliable at this benchmark's scale.

### `report.py`

`build_metrics_summary` computes every spec section 41-53 metric per variant from
persisted records. `evaluate_success_criteria` implements spec section 143's default
rule clause by clause (`pass_at_1_improves`, `pass_at_5_not_regressed`,
`syntax_success_not_regressed`, `timeout_rate_not_increased`,
`paired_ci_supports_improvement`, `catastrophic_regression_detected`), so a `false`
verdict is legible rather than a bare boolean. Spec section 70's overfitting check needs
training-set performance, which section 69 forbids evaluating in this stage — reported as
`not_applicable` rather than fabricated. `render_executive_summary` and
`render_markdown_report` assemble the spec section 84 report; `classify_problem_examples`
picks one representative sample per problem for `improvements.jsonl`/`regressions.jsonl`/
`ties.jsonl` (spec sections 130-132) — qualitative material, never fed back into the
model or the benchmark (spec section 133).

### `cache.py`

`GenerationCacheKey`/`EvaluationCacheKey` reduce to one SHA-256 `digest` over canonical
JSON. **Model identity is always part of the generation key** (spec section 94), so base
and DPO can never collide into one entry merely because the rendered prompt matched.
`JsonCacheStore` is a minimal digest-keyed file store for spec section 138's baseline-reuse
case; nothing in `generation.py`/`evaluation.py` consults a cache automatically — reuse is
opt-in, never a hidden source of truth.

### `run_repository.py`

Mints `eval_YYYYMMDD_HHMMSS_xxxx` ids and owns the run's manifest, config, benchmark
snapshot, generation/evaluation JSONL files, and status lifecycle. A **sixth** copy of this
plumbing after Stages 4, 6, 7, 8, 9; the extraction has been deferred at every stage since
Stage 7, and doing it now would touch six stages at once, so the debt is carried
deliberately. `sandbox_repository` points a fresh Stage 6 `EvaluationRepository` at
`evaluations/_sandbox/<variant>/` per run per variant — full stdout/stderr preserved for
forensics, kept separate from the curated `EvaluationRecord` schema in
`evaluations/<variant>.jsonl`.

## Persistence layout

```
benchmarks/<name>/manifest.json               # ids + hash; problems stay in problems.jsonl

data/model_evaluations/runs/eval_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json              # models, adapter, seeds, hardware, environment, status
├── config.yaml                # the resolved evaluation config actually used
├── benchmark_manifest.json    # the benchmark as consumed, with its hash
├── generations/{base,dpo}.jsonl
├── evaluations/
│   ├── {base,dpo}.jsonl       # curated EvaluationRecords
│   └── _sandbox/{base,dpo}/   # raw Stage 6 EvaluationRepository output
├── metrics/{summary,pass_at_k,bootstrap}.json
├── reports/
│   ├── base_vs_dpo.{md,json}
│   ├── {improvements,regressions,ties}.jsonl
│   └── failure_analysis.json
└── logs/evaluation.log
```

All text, no weights — the manifest records the adapter *path* and `training_run_id`,
never a copy. Committing the whole run directory is therefore cheap, consistent with every
prior stage.
