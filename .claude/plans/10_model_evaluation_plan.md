# Stage 10 — Base vs DPO Model Evaluation

## Context

Every stage so far has deferred the same question. Stage 7 refused to say `chosen`; Stage 8
refused to claim its pairs were good training data; Stage 9's report says in as many words
that a falling DPO loss is *not* evidence of better Python. Stage 10 is where that question
finally gets asked properly:

> Did DPO actually make Qwen better at Python programming?

`.claude/specs/10_model_evaluation.md` asks for a controlled experiment: the same held-out
problems, the same prompts, the same generation config, the same seeds, the same Docker
sandbox and the same pytest suite, run against **base Qwen** and **base Qwen + the trained
LoRA adapter**, with the difference measured statistically rather than eyeballed.

Four boundaries define the stage:

- **The benchmark is never used to tune anything (§134, §135).** Not beta, not LoRA rank,
  not epochs, not the preference policy. Consulting it to change any of those means a new
  experiment, not a better number.
- **Correctness comes from execution, not judgement (§2, §37).** Generated code goes
  through the Stage 5 sandbox and the Stage 6 executor. No LLM judge anywhere (§126, §133).
- **The comparison isolates the adapter (§28, §29).** Same prompt, same template, same
  seeds. The *only* difference between the two runs is base weights versus base + adapter.
- **No automatic promotion (§89, §142).** The stage produces evidence and a recommendation
  against configurable criteria. It does not declare a production model.

**Outcome:** `evaluate-model --benchmark python_eval_v1 --training-run-id dpo_...` produces
`data/model_evaluations/runs/eval_.../` with paired generations, sandboxed evaluations,
pass@k with bootstrap confidence intervals, win/tie/loss, failure analysis, and a
`base_vs_dpo.md` that states plainly what was and was not demonstrated.

---

### What exploration established

Measured on this machine and against the committed artifacts, not assumed. The first two
findings determine what this stage can and cannot show.

| Finding | Consequence |
|---|---|
| **Only 3 problems were ever trained on.** Stage 9 used `pref_20260818_074347_5eff`: train `p007,p008`, validation `p010`. The other **7 problems never entered any preference split** — they produced no pairs in Stage 7 because all their candidates tied | The leakage-clean held-out set is **p001, p002, p003, p004, p005, p006, p009** — 7 problems, not the single formally-designated test problem (`p004`) |
| **5 of those 7 are already solved perfectly by the base model.** From the committed Stage 6/7 evidence: p001, p003, p005, p006, p009 each have **5/5 candidates passing every test**. p002 has 0/5 (all five fail identically at 3/7); p004 has 3/5 | A hard **ceiling effect**. On 5 of 7 problems DPO cannot improve, only tie or regress. Expected base pass@1 ≈ **0.80**, with real headroom on 2 problems. This is the single most important thing the report must say |
| **The Stage 9 adapter trained for one optimizer step**, with all four DPO reward metrics identically zero | Base and DPO will very likely generate near-identical output. §119 is explicit that identical output is **not** an error. Decision 2 sets expectations accordingly |
| **Generation is fast**: measured ~6.0 s/generation, ~30 tok/s, model load 6.5 s warm | 7 problems × 10 samples × 2 models = 140 generations ≈ **14 min**, plus ~140 sandbox evaluations. Compute is **not** a constraint; n=10 for pass@10 is comfortably affordable |
| **Docker 28.3.3 is running and `python-dpo-evaluator:1.0` is already built** (`sha256:50d3905373da…`) | §37's mandated execution path is available with no image build step |
| **`CandidateEvaluator.evaluate(candidate, problem, …)` needs an in-memory `Candidate`, never a persisted one** — nothing in it reads `CandidateRepository`. Its `EvaluationRepository(directory)` is just a directory, not a run manifest | Stage 10 can construct candidates in memory, point the repository at its **own** run directory, and reuse Stage 6 wholesale — no Stage 4/6 run plumbing needed |
| The classification logic (`infrastructure > collection error > timeout > all-passed`) lives **only** in the private `CandidateEvaluator._classify` | Going through `CandidateEvaluator.evaluate` is the *only* way to reuse it. Reimplementing classification would risk Stage 10 disagreeing with Stage 6 about what "passed" means |
| `evaluate_many` skips work keyed on **`candidate_id` alone** | With k samples per problem, each sample needs a distinct `candidate_id` or they silently collapse into one |
| `Candidate.create` requires `candidate_id` to start with `f"{problem_id}_c"`, a **non-empty** `code`, and an `extraction_format` that is not `"unknown"` | A code-extraction failure **cannot** become a `Candidate` at all (§35). It must be recorded as a separate generation-failure record, exactly as Stage 3 does |
| **`QwenModelClient` supports a per-call seed** (`transformers.set_seed(generation_config.seed)` at the top of every `generate`) but supports **neither PEFT adapters nor quantization** — `ModelConfig.__post_init__` actively raises on any non-null `quantization` | Base inference could reuse it, but adapter inference cannot. A new runner is needed |
| **`training/verify.py` already loads base + adapter in 4-bit** — but hardcodes `do_sample=False` and has no seed handling | Unusable as-is for k-sample pass@k. Its `BitsAndBytesConfig`/`PeftModel.from_pretrained` shape is the right reference to build from |
| **No statistics machinery exists anywhere in the repo.** No `pass@k`, no `math.comb`, no bootstrap, no percentile beyond `training/lengths.py`'s token-length helper. scipy is not installed; numpy is only a transitive torch dependency | pass@k, bootstrap CIs and McNemar are all net-new — and all implementable in **pure stdlib** (`math.comb`, `random.Random`, a hand-rolled percentile), preserving the project's PyYAML-only runtime dependency |
| `_REQUIRED_PATH_KEYS` has eight entries; `data/model_evaluations/` does not exist, and there is no top-level `benchmarks/` | A **ninth** data path, threaded through `config.py`, `config.yaml`, `data/`, and the three literal enumerations in `tests/test_project.py`; plus a new `benchmarks/` tree |

---

### Decisions confirmed with the user

1. **The benchmark is all 7 held-out problems, and the ceiling is reported, not hidden.**
   `p001,p002,p003,p004,p005,p006,p009` — every problem never in a DPO train/validation
   split. The report carries an explicit **headroom analysis** stating that 5 of 7 sit at
   the base model's ceiling, so the benchmark can mostly only detect regression. Selecting
   only the 2 problems with headroom was rejected: choosing a benchmark by inspecting base
   results is the contamination §134/§135 exist to prevent.

2. **The first full run is pipeline validation, reported honestly.** With a one-step
   adapter the expected outcome is "no measurable difference" — `DPO_SUCCESS = false`, a
   paired CI straddling zero, wins 0 / losses 0. That still exercises the entire apparatus
   end to end, which is Stage 10's deliverable, and it establishes a reusable base-model
   baseline (§90, §138) for any better-trained adapter later.

3. **The benchmark manifest references problem ids and hashes their content.**
   `problems.jsonl` stays the single source of truth; `benchmark validate` recomputes the
   SHA-256 over the selected problems' canonical JSON and fails loudly on drift. This gives
   §10/§91's stability guarantee without duplicating ground truth into a second file that
   could silently disagree.

---

## New package — `src/python_dpo/model_evaluation/`

Named to sit alongside Stage 6's `evaluation/` without colliding, matching §80's
`data/model_evaluations/`. House style throughout: frozen dataclasses validating in
`__post_init__`, explicit `to_dict()`/`from_dict()` rejecting unknown and missing fields,
two-tier repositories over `atomic_io`, a per-package `README.md`.

**Import discipline carries over from Stage 9.** Every torch/transformers/peft touch is
deferred into the function that needs it, and `tests/test_no_heavy_imports.py` grows to
cover this package too.

**`errors.py`** — `ModelEvaluationError` base; `BenchmarkError`, `BenchmarkLeakageError`,
`ModelIdentityError`, `AdapterIntegrityError`, `PairingError`, `IncompleteBenchmarkError`,
`EvaluationRunNotFoundError`, `EvaluationRunError`, `EvaluationStoreError`.

**`config.py`** — the §140 schema, loaded from `configs/evaluation/python_eval.yaml`
(a sibling of Stage 9's training config, same standalone-file rationale):

```yaml
benchmark: {name: python_eval_v1}
generation: {temperature: 0.2, top_p: 0.95, max_new_tokens: 512, num_samples: 10,
             do_sample: true, repetition_penalty: 1.0, base_seed: 1000}
quantization: {enabled: true, bits: 4, quant_type: nf4, double_quant: true,
               compute_dtype: bfloat16}          # §19: identical for both models
statistics: {bootstrap_iterations: 1000, bootstrap_seed: 42, confidence_level: 0.95,
             pass_at_k: [1, 5, 10]}
success_criteria: {minimum_pass_at_1_improvement: 0.02,
                   maximum_allowed_regression: 0.02}   # §87, §88, §143
```

`__post_init__` enforces §43: requesting `pass@10` with `num_samples < 10` is a
configuration error, not a silently-degraded metric.

**`benchmark.py`** — §6–§10, §91. `Benchmark` (problem ids + the resolved `Problem`
objects), `BenchmarkManifest` (`benchmark_version`, `problem_ids`, `problem_count`,
`creation_date`, `dataset_hash`, `source_dataset_version`).

- `build_benchmark(problems, problem_ids)` → manifest, hashing the selected problems'
  canonical JSON (decision 3).
- `load_benchmark(benchmarks_root, name, problems)` — recomputes the hash and raises
  `BenchmarkError` on drift, which is what makes §91's stability claim real rather than
  aspirational.
- `check_leakage(benchmark, split_manifest)` (§7) — asserts the benchmark is disjoint from
  **both** the training and validation splits, and raises `BenchmarkLeakageError` naming
  the offending ids. A contaminated benchmark must stop the run, not warn.
- `reference_solution` is deliberately never surfaced to a runner (§6).

**`models.py`** — the persisted schema:

`GenerationRecord` (§81) — `problem_id`, `model_variant`, `sample_index`, `seed`,
`prompt_sha256`, `raw_response`, `extracted_code`, `extraction_format`, `syntax_valid`,
`generation_time_ms`, `generated_tokens`, `status` (`generated`/`generation_error`),
`error`. §36's separation is structural: `raw_response` and `extracted_code` are distinct
fields, and a failed extraction keeps the raw text with `extracted_code = None`.

`EvaluationRecord` (§82) — `problem_id`, `model_variant`, `sample_index`, `tests_total`,
`tests_passed`, `tests_failed`, `tests_error`, `timeout`, `status`, `duration_ms`,
`error_type`. `correct` is a **derived property** — `status == "passed" and tests_total > 0
and tests_passed == tests_total` — so §40's exact-correctness rule lives in one place.

`ModelEvaluationManifest` (§32, §78, §79) — `evaluation_run_id`, `benchmark_version`,
`benchmark_hash`, `base_model_name`/`base_model_revision`, `adapter_path`,
`training_run_id`, `generation_config`, `quantization`, `seeds`, `hardware`,
`environment`, `status`, timestamps, and the same status lifecycle as Stages 4/6/7/8/9.

`MetricsSummary` (§83) and `ComparisonReport` (§84) round out the schema.

**`runners.py`** — §16–§19, §28. The inference layer, all deferred-import.

```
ModelRunner (Protocol)        variant · name · revision · generate(prompt, seed) -> Generation
  BaseModelRunner             4-bit base weights only
  AdapterModelRunner          the same 4-bit base + PeftModel.from_pretrained(adapter)
```

Both are built by one factory from one quantization config, so §19's "same quantization
for both" is structural rather than a thing to remember. The model is loaded **once per
variant** and reused across every problem and sample (§17), then explicitly freed before
the other variant loads — the Stage 9 GPU-test lesson, where a held model starved the next
step.

`AdapterModelRunner.__init__` performs §15's integrity checks — adapter directory,
`adapter_config.json`, weights present, and the adapter's recorded base model matching the
configured one — and raises `AdapterIntegrityError` rather than **ever** falling back to
the base model, which §15 forbids and §118 tests for.

Per-call seeding follows `QwenModelClient`'s proven pattern (`transformers.set_seed(seed)`
immediately before generation). The chat template is applied identically for both variants
(§28), matching Stage 3 generation and Stage 9 training.

**`generation.py`** — §20–§36. `GenerationDriver.run(runner, benchmark, config)`:

- The prompt is Stage 9's **canonical, strategy-free** prompt
  (`build_canonical_prompt`) — the same text the adapter was trained against, and
  model-agnostic per §29.
- Seeds are derived deterministically as `base_seed + problem_index * 1000 + sample_index`
  and recorded per record (§25, §26). Both variants consume the **same** schedule, which is
  what makes the comparison paired rather than merely simultaneous.
- `extract_code` (Stage 3) is used for both variants, unmodified (§33, §34). A failed
  extraction becomes `status = "generation_error"` with the raw response retained (§35),
  and — because `Candidate.create` rejects empty code — never reaches the sandbox.
- `prompt_sha256` is computed per record so §27/§114's prompt-equality check is verifiable
  from the artifacts alone, not merely asserted in code.

**`evaluation.py`** — §37–§40. Wraps Stage 6 rather than reimplementing it:

```
for each GenerationRecord with extracted code:
    Candidate.create(...)                     # in memory, never persisted
    CandidateEvaluator.evaluate(candidate, problem, evaluation_run_id=...)
    -> EvaluationRecord
```

`candidate_id` is `f"{problem_id}_c{sample_index+1:03d}"` per variant, into **separate
per-variant `EvaluationRepository` directories** — which is what stops `evaluate_many`'s
`candidate_id`-keyed skip from collapsing samples, and simultaneously satisfies §94's
no-cross-model-cache-collision rule.

An `infrastructure_error` is recorded and **excluded from correctness statistics** rather
than counted as a failure (§120), and surfaces in the §123 completeness table.

**`metrics.py`** — §41–§53. Pure stdlib.

`pass_at_k(n, c, k)` implements the unbiased estimator exactly:
`1 - C(n-c, k)/C(n, k)`, with `1.0` when `c >= n - k + 1` and `0.0` when `c == 0`. §44 is
explicit that `c/n` is **not** pass@k, and §111 gives the boundary cases the tests assert.
Aggregation is **problem-level then averaged** (§45, §46, §110) — never pooling samples as
independent observations.

Also: `mean_test_pass_rate` (§47), `solve_rate` (§48), `syntax_success_rate` (§49),
`execution_success_rate` (§50), `timeout_rate` (§51), `generation_failure_rate` (§52), and
§53's seven-bucket test-failure distribution.

**`comparison.py`** — §54–§58, §109, §121–§123, §127.

`paired_problem_set` (§122) is computed first and is the basis of every headline number:
only problems successfully evaluated for **both** variants participate, and its size is
reported (§123). §121's mismatch case is an error, not a silent narrowing.

Per problem: `base_pass`/`dpo_pass`, `improvement ∈ {+1, 0, -1}`, and
`test_pass_delta` (§57, §127) — the last of which is how partial improvement shows up even
when both models fail outright, which on this benchmark is the more likely place to see
anything at all. `dpo_win_rate = wins / (wins + losses)`, ties excluded from the
denominator (§56).

**`statistics.py`** — §59–§66, §105–§108. Pure stdlib, seeded.

- `bootstrap_ci(values, statistic, iterations, seed, confidence)` — resamples **problems**
  with replacement (§60, §61), never candidates, because each problem contributes k
  correlated samples.
- `paired_bootstrap(base, dpo, …)` (§62, §107) — resamples the problem set once per
  iteration and computes `dpo_metric - base_metric` on that same resample, yielding the
  distribution of *differences* rather than the difference of two independent intervals.
- `mcnemar(base_solved, dpo_solved)` (§63) — the exact binomial test over discordant pairs
  via `math.comb`, no scipy.
- `bootstrap_seed` and `bootstrap_iterations` are persisted (§105), so a reported interval
  is reproducible rather than merely plausible.

**`report.py`** — §84, §85, §124–§132, §141, §149.

`reports/base_vs_dpo.json` and `.md` with the §84 section list. The executive summary is
generated from the computed numbers (§85) and is written to be capable of saying *"DPO did
not improve pass@1"* as fluently as the opposite — with no causal language either way.

Also `improvements.jsonl` (§131), `regressions.jsonl` (§130, including both models' code
for qualitative review), `ties.jsonl` (§132), and `failure_analysis.json` (§124–§126)
classifying failures deterministically by `error_type` — no LLM, per §126.

The §143 success rule is evaluated and reported as `DPO_SUCCESS: true/false` with each
clause's pass/fail shown, so a "false" is legible rather than a bare verdict. §71's
catastrophic-regression check and §70's overfitting warning are separate flags.

**`run_repository.py`** — §79, §80. `ModelEvaluationRunRepository` minting
`eval_YYYYMMDD_HHMMSS_xxxx`, owning `manifest.json`/`config.yaml`/`benchmark_manifest.json`
and the status lifecycle. A **sixth** copy of the run plumbing; the extraction has been
deferred since Stage 7 and doing it now would touch six stages, so the debt is carried
deliberately again and re-flagged.

**`cache.py`** — §92–§94, §138, §139. A content-addressed key over
`(model identity, adapter identity, prompt_sha256, generation config, seed)` for
generations, and `(candidate code hash, problem id, test suite hash, evaluator version,
sandbox config)` for evaluations. **Model identity is always part of the key** (§94), so
base and DPO can never share an entry merely because the prompt matched. Any difference in
revision, generation config, benchmark or seed invalidates reuse (§139).

---

## Persistence layout (§80, §148)

```
benchmarks/python_eval_v1/manifest.json          # ids + hash; problems stay in problems.jsonl

data/model_evaluations/runs/eval_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json              # models, adapter, seeds, hardware, environment, status
├── config.yaml                # the resolved evaluation config actually used
├── benchmark_manifest.json    # the benchmark as consumed, with its hash
├── generations/{base,dpo}.jsonl
├── evaluations/{base,dpo}.jsonl
├── metrics/{summary,pass_at_k,bootstrap}.json
├── reports/base_vs_dpo.{md,json}
├── reports/{improvements,regressions,ties}.jsonl
├── reports/failure_analysis.json
└── logs/evaluation.log
```

All text, no weights — the manifest records the adapter *path* and `training_run_id`, never
a copy (§80). Committing the whole run directory is therefore cheap and consistent with
every prior stage; no new `.gitignore` rules are needed.

---

## Modifications to existing code

**`src/python_dpo/config.py` + `config.yaml` + `data/`** — the **ninth** data path,
`model_evaluations`, threaded through `_REQUIRED_PATH_KEYS`, `Paths`, `ensure_exists()`,
and `data/model_evaluations/.gitkeep`. As in Stage 9, no settings section in the root
config: evaluation parameters live in `configs/evaluation/python_eval.yaml`.

**`src/python_dpo/cli.py`** — two new command groups:

| Command | Behavior |
|---|---|
| `benchmark build --name python_eval_v1 [--problem-id ...] [--exclude-preference-run-id ID]` | Create the benchmark manifest; default excludes the train/validation problems of a preference run |
| `benchmark validate --benchmark NAME` | §147.1 — recompute the hash; `Benchmark validation passed.` |
| `benchmark check-leakage --benchmark NAME --preference-run-id ID` | §147.2 — `No problem leakage detected.` |
| `evaluate-model --benchmark NAME --training-run-id ID [--model base\|dpo\|both] [--num-samples N] [--limit N] [--smoke-test] [--config PATH]` | §95–§101 |
| `evaluate-model validate --evaluation-run-id ID` | §102 |
| `evaluate-model report --evaluation-run-id ID` | §103 |
| `evaluate-model stats --evaluation-run-id ID` | §104 |
| `evaluate-model compare --runs A,B,C` | §136, §137 |
| `evaluate-model list` | Discover a run id — as with `preferences`/`train`, none of the above is usable without it |

`evaluate-model` is a **group with a default action**, so the bare `evaluate-model
--benchmark ... --training-run-id ...` form in §95 works alongside the `validate`/`report`/
`stats` subcommands. Handlers keep the house contract: `(args, config) -> int`, data to
`sys.stdout`, errors via `logger.error`, exit codes 0/1/2/130.

**`pyproject.toml`** — **no new dependencies.** pass@k, bootstrap and McNemar are pure
stdlib, preserving PyYAML-as-the-only-runtime-dependency. The GPU-gated tests reuse Stage
9's `gpu` marker; the sandbox-dependent tests reuse `integration`.

**`tests/test_no_heavy_imports.py`** — the probe grows to import
`python_dpo.model_evaluation` and every submodule.

**`src/python_dpo/__init__.py`** — `__version__` → `0.10.0`.

**Docs** — `src/python_dpo/model_evaluation/README.md` and `benchmarks/README.md` (new),
plus Stage 10 sections in `README.md`, `src/python_dpo/README.md`, `data/README.md`,
`tests/README.md`, `configs/README.md`.

---

## Tests

Three tiers, because the stage spans pure arithmetic, Docker, and a GPU.

**Offline (`tests/model_evaluation/`, default suite):**

- **`test_metrics.py`** — §111's estimator cases exactly: `n=10,c=10 → pass@{1,5,10}=1`;
  `n=10,c=0 → 0` for all k; plus `n=10,c=1 → pass@1=0.1, pass@10=1.0`, the `c >= n-k+1`
  boundary, `k>n` rejected, and — the §44 guard — an explicit assertion that pass@5 for
  `n=10,c=5` is **not** `c/n`. Also every rate in §49–§53 and the §53 buckets.
- **`test_statistics.py`** — bootstrap reproducibility under a fixed seed; a CI that
  brackets a known mean; §60's problem-level resampling verified by construction (a
  degenerate input where candidate-level resampling would give a visibly different answer);
  paired bootstrap on synthetic data with a known difference; McNemar against hand-computed
  binomial values.
- **`test_comparison.py`** — §112's mock benchmark (10 problems, base 4 solved, DPO 6)
  asserting `base=0.4, dpo=0.6, improvement=+0.2`; §113's regression case (8 vs 6 →
  `-0.2`) asserting the report *identifies* it as a regression; win/tie/loss and §56's
  tie-excluded win rate; §57's test-pass delta; §121's mismatched-set refusal and §122's
  paired subset.
- **`test_benchmark.py`** — manifest build and hash stability; §10 drift detection (mutate
  a problem, expect a validation failure); §7 leakage detection against a split manifest,
  in both the train and validation directions; that `reference_solution` never reaches a
  runner.
- **`test_generation.py`** — seed schedule determinism and §116's `base.seed[i] ==
  dpo.seed[i]`; §27/§114's prompt equality; §115's identical generation config; extraction
  failure producing `generation_error` with the raw response retained and **no** candidate
  constructed; all driven through `MockModelClient`, no GPU.
- **`test_cache.py`** — §94's key separation (same prompt, different variant → different
  key); §139's invalidation on revision/config/benchmark/seed change.
- **`test_models.py`** / **`test_config.py`** / **`test_run_repository.py`** — schema
  round-trips, the §43 `pass@10`-needs-`n≥10` rule, run-id minting and the status lifecycle.
- **`tests/test_project.py`** — the ninth data path in all three enumerations; the new CLI
  groups' parsing and error paths.

**Docker-gated (`pytestmark = pytest.mark.integration`):** one real candidate through the
Stage 6 executor via the Stage 10 wrapper, proving §117 — that Stage 10 uses the same
sandbox config, evaluator version, pytest version and test suite as Stage 6, rather than a
parallel path that could drift.

**GPU-gated (`pytestmark = pytest.mark.gpu`):** §118's **adapter isolation test** — that
`BaseModelRunner` does not load the adapter and `AdapterModelRunner` does not silently fall
back to base — asserted by inspecting the loaded module tree for LoRA layers, not by
comparing outputs (which §119 says may legitimately be identical). Plus §15's integrity
failures against a corrupted adapter directory, and a 1-sample smoke generation per variant.

---

## Execution order

1. Write this plan to `.claude/plans/10_model_evaluation_plan.md` and add its entry to
   `.claude/plans/README.md`.
2. `errors.py`, `config.py`, `models.py` + tests — pure schema.
3. `metrics.py` + `statistics.py` + tests. **First real code**, because §111/§112/§113 give
   exact expected values, so correctness is verifiable before any model is involved — and
   because a wrong pass@k estimator would invalidate every downstream number.
4. `benchmark.py` + tests; wire `benchmark build/validate/check-leakage`; build
   `python_eval_v1` over the 7 held-out problems and verify no leakage against
   `pref_20260818_074347_5eff`.
5. `comparison.py` + tests (§112, §113's mock cases).
6. `runners.py` + the GPU adapter-isolation test — the riskiest new code, and the only part
   with no in-repo precedent for seeded, quantized, adapter-aware sampling.
7. `generation.py` + tests via `MockModelClient`.
8. `evaluation.py` + the Docker-gated Stage 6 reuse test.
9. `cache.py`, `run_repository.py`, `report.py` + tests.
10. Config path wiring (ninth path, three `test_project.py` enumerations), CLI wiring,
    `test_no_heavy_imports.py` extension.
11. Run §147's eight verification steps; commit the evaluation run; docs; the §151 report.

---

## Verification

```bash
source .venv/bin/activate
pytest -q                     # offline, zero skips
pytest -q -m integration      # Docker: Stage 6 reuse
pytest -q -m gpu              # CUDA: adapter isolation, smoke generation

# §147.1 / §147.2
python -m python_dpo benchmark build --name python_eval_v1 \
    --exclude-preference-run-id pref_20260818_074347_5eff
python -m python_dpo benchmark validate --benchmark python_eval_v1
python -m python_dpo benchmark check-leakage --benchmark python_eval_v1 \
    --preference-run-id pref_20260818_074347_5eff

# §147.3 / §147.4
python -m python_dpo train hardware-check
python -m python_dpo evaluate-model --benchmark python_eval_v1 \
    --training-run-id dpo_20260818_081231_a91d --smoke-test

# §147.5 — the full run
python -m python_dpo evaluate-model --benchmark python_eval_v1 \
    --training-run-id dpo_20260818_081231_a91d --num-samples 10

# §147.6 / §147.7 / §147.8
python -m python_dpo evaluate-model validate --evaluation-run-id EVAL_ID
python -m python_dpo evaluate-model report   --evaluation-run-id EVAL_ID
python -m python_dpo evaluate-model stats    --evaluation-run-id EVAL_ID
```

**Expected, computed from the committed Stage 6/7 evidence:**

```
benchmark      python_eval_v1 · 7 problems · p001 p002 p003 p004 p005 p006 p009
leakage        none (train p007,p008 · validation p010 all excluded)
work           7 problems x 10 samples x 2 models = 140 generations (~14 min)
                                                  + 140 sandbox evaluations

base, from Stage 6/7 evidence at temperature 0.8:
  ceiling  p001 p003 p005 p006 p009   5/5 candidates correct
  headroom p004  3/5 correct · p002  0/5 correct (all five fail identically at 3/7)
  expected base pass@1 ~= 0.80

dpo            near-identical to base — the adapter is one optimizer step, all four
               reward metrics identically zero
expected       wins 0 · ties ~7 · losses 0 · paired CI straddling zero
               DPO_SUCCESS = false
```

Stage 10's own run uses temperature 0.2 and the canonical prompt, so the numbers will not
reproduce Stage 6's exactly — but the ceiling is a property of the problems, not the
sampling temperature, and will hold.

```bash
# §27, §114, §115, §116: the comparison really is paired
jq -s '[.[0].prompt_sha256] == [.[1].prompt_sha256]' \
   <(head -1 RUN/generations/base.jsonl) <(head -1 RUN/generations/dpo.jsonl)   # true
diff <(jq -c '{problem_id,sample_index,seed}' RUN/generations/base.jsonl) \
     <(jq -c '{problem_id,sample_index,seed}' RUN/generations/dpo.jsonl)        # empty

# §5, §7: no trained problem is in the benchmark
jq -r '.problem_ids[]' benchmarks/python_eval_v1/manifest.json | grep -E 'p007|p008|p010'  # none

# nothing upstream was mutated
git diff --stat data/problems/ data/candidates/ data/evaluations/ \
                data/rankings/ data/preferences/ data/training/                 # empty
```

Scope containment:

```bash
grep -rniE "\b(llm|judge|gpt|claude|openai)\b" src/python_dpo/model_evaluation/  # none (§126, §133)
grep -rniE "\b(train|optimizer|backward|lora_config|get_peft_model)\b" \
     src/python_dpo/model_evaluation/                                            # none (§4)
grep -rn "scipy\|numpy" src/python_dpo/model_evaluation/                         # none
grep -rn "c / n\|c/n" src/python_dpo/model_evaluation/metrics.py                 # none (§44)
```

Then produce the §151 report in `10_MODEL_EVALUATION.md` and **stop — do not start Step 11
without explicit approval** (§151).

**The honest headline for that report (§144, §85, §89):** seven held-out problems, five of
them already at the base model's ceiling, evaluated against an adapter that trained for one
optimizer step. Stage 10 will have demonstrated that the evaluation apparatus works —
paired generation, sandboxed execution, a correct pass@k estimator, bootstrap intervals,
regression detection — and it will almost certainly report **no measurable difference**.
That is the correct finding for this input, and §144 says so directly: a benchmark this
size is suitable for pipeline validation, not for reliable model-performance conclusions.
The pipeline is what is being validated; the model verdict is not yet available.

---

## Deviations to record in the report

- **The benchmark is the 7 never-trained-on problems, not the single formally designated
  test problem** (decision 1). Stage 8's splitter only assigns *pair-bearing* problems to
  splits, so six problems that produced no preference pairs are equally untrained and
  equally valid as held-out data.
- **5 of the 7 benchmark problems sit at the base model's ceiling.** Reported as a
  first-class headroom analysis rather than buried, because it bounds what any result can
  mean. Selecting only the 2 problems with headroom was explicitly rejected as benchmark
  contamination (§134, §135).
- **The benchmark manifest references problem ids and hashes their content** rather than
  snapshotting the problems (decision 3) — `problems.jsonl` stays the single source of
  truth, with drift detected by hash rather than prevented by duplication.
- **A new inference layer was required.** `QwenModelClient` supports per-call seeding but
  neither adapters nor quantization, and `ModelConfig` actively rejects any non-null
  `quantization`; `training/verify.py` loads adapters in 4-bit but is greedy-only and
  unseeded. Neither is reusable for seeded k-sample pass@k.
- **Candidates are constructed in memory and never persisted through Stage 4.**
  `CandidateEvaluator` needs a `Candidate` object but never reads the candidate repository,
  so Stage 10 reuses Stage 6's execution *and its classification* without inheriting Stage
  4's run plumbing.
- **Per-variant evaluation repositories**, because `evaluate_many` skips on `candidate_id`
  alone — which would otherwise collapse k samples, and would also violate §94.
- **pass@k, bootstrap and McNemar are hand-rolled in pure stdlib** rather than adding
  numpy/scipy, preserving PyYAML as the only runtime dependency and matching the
  `training/lengths.py` percentile precedent.
- **`evaluate-model` is a command group with a default action**, so §95's bare form and
  §102–§104's subcommands coexist.
- **`evaluate-model list` was added** beyond §95–§104, since none of the specified commands
  is usable without a way to discover an `evaluation_run_id`.
- **A sixth copy of the run-directory plumbing**, rather than the shared base deferred at
  Stages 7, 8 and 9. Extracting it would now touch six stages at once.
- **The expected verdict is `DPO_SUCCESS = false`**, and the success thresholds are **not**
  adjusted to produce a pass. Tuning the gates to clear a one-step adapter would be exactly
  the dishonesty §85 and §134 exist to prevent.
