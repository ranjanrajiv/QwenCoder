# Stage 11 — Error Analysis, Preference Refinement and Iterative Improvement

## Context

Stage 10 answered *"did DPO make Qwen better at Python?"* with a number: base pass@1
77.1%, DPO pass@1 78.6%, `DPO_SUCCESS = false`. Stage 11 asks the question that number
cannot answer:

> What should we change in the next DPO iteration to improve held-out Python performance?

`.claude/specs/11_error_analysis_and_iteration.md` is emphatic that this must not become a
data-generation reflex (§2, §31): the pipeline is evidence → classification → gap analysis
→ hypothesis → *proposed* experiment. Four boundaries define the stage:

- **Classification is deterministic, never an LLM (§12, §295).** pytest status, exception
  type, exit code and timeout status are the evidence. An LLM judge is explicitly forbidden
  as the primary classifier, and the optional secondary layer is deferred (decision 1).
- **Correlation is never causation (§38).** The reports say *"potential data gap"*, never
  *"DPO failed because of insufficient data."* This is a wording rule enforced in `report.py`
  and asserted in tests.
- **The benchmark must not be contaminated (§64, §65, §66, §104, §117).** Held-out problems
  that DPO failed are the single most tempting thing to add to training, and doing so
  invalidates every future number. Hard examples are provenance-tagged *pointers*, never
  training rows.
- **No automatic retraining (§5, §113).** The stage emits `next_experiment.yaml` and stops.

**Outcome:** `analyze --evaluation-run-id eval_...` produces
`data/analysis/runs/analysis_.../` with error profiles, improvement/regression
classification, test-level failure frequencies, diversity and coverage analysis, an
evidence-backed recommendation set, a refined preference dataset, and an `analysis.md` that
states plainly what the evidence does and does not support.

Stage 11 is **pure computation over persisted artifacts** — no model, no GPU, no Docker.
The whole stage is testable in the default offline suite, which is a first for the back
half of this pipeline.

---

### What exploration established

Measured against the committed artifacts on this machine, not assumed. The first four
findings determine what this stage can honestly conclude.

| Finding | Consequence |
|---|---|
| **The real analysis is degenerate before it starts.** `eval_20260818_155511_1633` has **0 DPO wins, 0 losses, 7 ties**; `improvements.jsonl` and `regressions.jsonl` are both **0 bytes**; `failure_analysis.json` has exactly one non-zero bucket (`assertion_failure`: base 16, dpo 15) | §90's iteration decision is **`insufficient_evidence`** (§95), and it must *gate* the others — `refine_data` also has support, but reporting it as the headline over a 7-problem benchmark with a CI 4.3pp wide would be exactly the overreach §38 and §56 forbid |
| **Per-test forensics exist, but only in `evaluations/_sandbox/<variant>/`.** `EvaluationRecord` carries counts and a coarse `error_type`, no message and no traceback. `test_results.jsonl` carries `test_case_id`, `status`, `error_type` (**the raw exception class** — real values are `AssertionError` ×97, `Failed` ×1), `error_message`, `traceback` | §11's *hierarchical* taxonomy is only reachable through that file. `ModelEvaluationRunRepository.sandbox_repository(run_id, variant)` is **already public** and returns a Stage 6 `EvaluationRepository` with `load_test_results()` / `test_results_for(candidate_id)` — so §45–§49 need **no change to Stage 10**. Join key is `candidate_id`; the variant is knowable only from which directory the file came from |
| **Coverage analysis is the one place with a real, strong signal — and it is total.** Only `p007` + `p008` were ever trained on (`p004` and `p010` went to test/validation). Their categories are `edge_cases` and `exceptions`; the benchmark's are `lists, dictionaries, strings, sets, sorting, recursion, generators`. **Overlap: zero.** Every one of the 7 benchmark categories scores `coverage_ratio = 0.00` | §37's "underrepresented" verdict fires on 7/7 categories. This is the stage's one genuinely informative finding, and it is a *structural* fact about the split, not sampling noise |
| **`coverage_ratio` has two degenerate cases that JSON cannot represent.** `edge_cases`/`exceptions` are 50% of training and 0% of the benchmark → division by zero (`inf`); `hard` is 0% of both → `0/0` (`nan`). Difficulty: easy 2.33 (over), medium **0.00** (under), hard undefined | The ratio must be a `float \| None` beside an explicit verdict enum — `underrepresented`, `balanced`, `overrepresented`, **`not_in_benchmark`**, **`absent_from_both`** — never `inf`/`nan` written to a JSON file |
| **Every category has exactly one problem** (10 problems, 10 categories, 5 easy / 4 medium / 1 hard) | Category "distribution" analysis is arithmetic over n=1 cells. The maths is correct and the conclusion is noise; the report must carry that caveat next to the table, per §95 |
| **Diversity is computable and already interesting.** Measured over `generations/*.jsonl` by SHA-256 of `extracted_code`: base **22/70 = 0.314**, DPO **20/70 = 0.286** — an 8.9% relative reduction | Below §88's 20% mode-collapse gate, so **no warning fires**. Per-problem the collapse is severe (p001 and p002 are 1/10 unique for *both* variants) but that is temperature 0.2, not DPO |
| **Test-level analysis produces 7 non-trivial rows**, computed from the sandbox files: `p002_t002/t005/t006/t007` fail 100% for both (§47 hard tests), `p004_t002` 50%/50%, `p004_t008` 30%→40% (§48 DPO-specific), `p005_t005` 10%→0% (§49 base-specific) | §45–§49 are the *other* part of the stage with real signal on this data. Worth building carefully |
| **Stage 8 already computed much of §32–§41.** `QualityReport` has `strong_pairs`, `medium_pairs`, `score_margin_distribution`, `strategy_distribution` (`{"chosen": {...}, "rejected": {...}}`) and `problems_without_pairs` **with reasons**; `DatasetManifest.statistics["balance"]` has `mean_score_margin`, `strong_pair_percentage`, `medium_pair_percentage` | §32, §33, §39 and §41 are largely a *join and a rendering*, not new computation. `problems_without_pairs` reasons (real data: 6/10 problems, mostly `all_candidates_correct`) are the direct explanation for the coverage hole |
| **`preferences.jsonl` and the split files carry only `{prompt, chosen, rejected}`.** All analysable metadata is in `metadata.jsonl` | Coverage work must go through `PreferenceRepository.load_pairs()`. Also: 22 pairs in metadata vs 7 training records — `duplicate_training_record` must be handled explicitly per metric, and the **training** population is the 12 pairs on `p007`/`p008`, not all 22 |
| **Category and difficulty are on `Problem` only** — no preference, ranking or evaluation record carries them | Every gap metric joins on `problem_id` back to `load_problems(dataset_path(config.paths.problems))` |
| **The "strict vs margin" comparison of §40 is real and available.** Three preference runs share `ranking_run_id = rank_20260817_161726_a84d`, with `selection_policy` `strict` / `margin` / `all_better` | §40 groups `PreferenceRunRepository.list_runs()` by `ranking_run_id` and keys by `selection_policy`. But only `all_better` was ever trained, so the pass@k columns are empty for two of three rows |
| **§85–§87 training-curve analysis has one data point.** `metrics.jsonl` is 3 lines, all `step: 1`: `loss = 0.6931` (exactly ln 2 — zero reward margin), `rewards/margins = 0.0`, `eval_loss = 0.6769` | Overtraining/undertraining detection needs a trend. The correct output is `insufficient_data`, reported as such rather than a fabricated verdict |
| **No `preference_run_id` on the Stage 10 manifest.** The §7 lineage chain runs `eval_* → training_run_id → training manifest → preference_run_id → ranking_run_id → candidate_run_id` | Lineage is resolved by *hopping manifests*, and Stage 11 persists the resolved chain. Watch the collision: the training manifest's `evaluation_run_id` is a **Stage 6** run id, not a Stage 10 one |
| `_REQUIRED_PATH_KEYS` has nine entries; `data/analysis/` does not exist | A **tenth** data path, threaded through `config.py`, `config.yaml`, `data/`, `.gitignore`'s log negation, and the three literal enumerations in `tests/test_project.py` |

---

### Decisions confirmed with the user

1. **Every explicitly-optional capability is deferred.** No LLM-based semantic analysis
   (§13, §44), no `HardProblemGenerator` (§67, §68), no flaky-test detection (§72–§74).
   Flaky detection in particular would require re-executing candidates N times through the
   Stage 5/6 Docker sandbox — it would turn a pure-computation stage into a Docker-dependent
   one for a property this dataset cannot exhibit. All three are recorded as deliberate
   non-implementations with reasoning, per CLAUDE.md's Scope Control rule.

2. **Refinement produces both a dataset and the example files.** `refined_preferences.jsonl`
   is emitted as a filtered, re-versioned copy of the Stage 8 pairs (`dpo_preference_v2`)
   carrying `parent_preference_run_id`, **never overwriting Stage 8** (§77, §78, §79) —
   alongside `refined_dataset/{hard,regression,successful_dpo}_examples.jsonl` (§59–§62) and
   `next_experiment.yaml` (§80).

3. **The real run is analysed and committed reporting `insufficient_evidence`.** Exactly as
   Stage 9 reported a one-step adapter and Stage 10 reported `DPO_SUCCESS = false`. The
   improvement, regression, mode-collapse and non-degenerate coverage paths are exercised by
   unit tests over synthetic fixtures, including §118's 20-problem scenario. No fabricated
   analysis run is committed next to the real one.

---

## New package — `src/python_dpo/analysis/`

House style throughout: frozen dataclasses validating in `__post_init__`, explicit
`to_dict()`/`from_dict()` rejecting unknown and missing fields, a two-tier repository over
`atomic_io`, a per-package `README.md`, docstrings citing spec sections verbatim.

No heavy imports anywhere — this package genuinely has nothing to defer, but every
submodule still joins the `_PROBE` list in `tests/test_no_heavy_imports.py`.

**`errors.py`** — `AnalysisError` base; `AnalysisConfigError`, `AnalysisInputError`,
`AnalysisRunNotFoundError`, `AnalysisRunError`, `AnalysisStoreError`, `LineageError`,
`RefinementLeakageError`.

**`config.py`** — loaded from `configs/analysis/python_analysis.yaml`, a sibling of Stage
9's and Stage 10's standalone configs (same rationale: no root-config section). Every
threshold the spec calls configurable lives here:

```yaml
thresholds:
  regression_threshold: 0.2          # §52
  coverage_underrepresented: 0.5     # §37
  coverage_overrepresented: 2.0      # §37
  mode_collapse_reduction: 0.2       # §88
  hard_test_failure_rate: 0.5        # §47
  variant_specific_test_delta: 0.2   # §48, §49
minimum_evidence:
  benchmark_problems: 30             # §95 — below this, insufficient_evidence
  max_ci_width: 0.15                 # §95 — wider than this, insufficient_evidence
recommendations:
  max_recommendations: 10            # §57
  weights: {expected_impact: 0.5, evidence_strength: 0.3, implementation_cost: 0.2}
refinement:
  minimum_score_margin: 0.2          # §71
  drop_duplicate_code: true          # §71
  drop_infrastructure_errors: true   # §71
```

`__post_init__` enforces `coverage_underrepresented < coverage_overrepresented` and that
every threshold is in its valid range.

**`taxonomy.py`** — §10, §11, §12. The deterministic classifier, and the only place error
strings are minted.

```python
ERROR_CATEGORIES = ("generation_failure", "code_extraction_failure", "syntax_error",
                    "import_error", "runtime_error", "assertion_failure", "timeout",
                    "memory_error", "infrastructure_error")

classify(generation_record, evaluation_record, test_results) -> ErrorClassification
```

The coarse category comes from Stage 10's already-deterministic `EvaluationRecord.error_type`
(§12 prefers the pytest result, and re-deriving it would risk Stage 11 disagreeing with
Stage 10 about what failed) — with two categories Stage 10 has no name for, resolved
locally: `generation_failure` from `GenerationRecord.status == "generation_error"`, and
`code_extraction_failure` distinguished from it by `extracted_code is None` with a non-empty
`raw_response`. `memory_error` is promoted out of `runtime_error` when the subcategory is
`MemoryError`.

The **subcategory** (§11) is the raw exception class from the per-test records —
`TypeError`, `IndexError`, `KeyError`, `AttributeError`, `ValueError`, `MemoryError`, …
with anything unrecognised mapped to `other` rather than invented. The real data's
`Failed` (pytest's own `pytest.fail`, not a builtin) is the case that proves this branch is
needed. Subcategory selection when a candidate fails several tests with different
exceptions is **most frequent, ties broken alphabetically** — deterministic, and recorded
alongside the full per-subcategory counts so nothing is discarded (Data Integrity).

**`models.py`** — the persisted schema. `ErrorClassification`, `ErrorProfile` (§14/§15's
seven counters plus the hierarchical breakdown), `ErrorRateComparison` (§16:
`base_rate`, `dpo_rate`, `delta`, `relative_delta`), `ProblemOutcome`, `TestFailureStat`,
`DiversityReport`, `CategoryGap`, `DifficultyGap`, `PreferenceCoverage`, `StrategyGap`,
`Recommendation`, `AnalysisSummary` (§96's exact field list), `AnalysisManifest`,
`ExperimentLineage`.

`Recommendation` (§54, §55, §57, §58) is the load-bearing one:

```python
category: str            # §58's ten-value closed set
hypothesis: str          # §102, §103 — mandatory, not "change X"
evidence: dict[str, Any] # §55 — mandatory, non-empty (validated)
confidence: str          # low | medium | high
expected_impact: float; evidence_strength: float; implementation_cost: float
recommendation_score: float   # §57, from the configured weights
```

`__post_init__` **rejects an empty `evidence` dict and an empty `hypothesis`** — §56 and
§103 are enforced by the type, not by discipline.

**`ingest.py`** — §6, §7. `load_analysis_inputs(...)` gathers the Stage 10 run (manifest,
config, benchmark manifest, per-variant generation and evaluation records, and the
`_sandbox` per-test results via `sandbox_repository()`), then walks the manifest chain to
resolve `ExperimentLineage` — `evaluation_run_id → training_run_id → preference_run_id →
ranking_run_id → candidate_run_id` — raising `LineageError` when a hop is missing rather
than silently analysing a partial chain. §7 calls the lineage mandatory, so it is a
precondition, not an enrichment.

The optional `--preference-run-id` / `--training-run-id` flags override the resolved chain;
a mismatch between an explicit flag and the resolved value is an error, not a preference.

**`classification.py`** — §14, §15, §16. Builds `base_error_profile.json` and
`dpo_error_profile.json` from the same code path so the two are structurally comparable, and
the per-category rate comparison. Infrastructure errors are excluded from correctness rates
and reported separately, matching Stage 10's §120 treatment.

**`outcomes.py`** — §17–§25, §50–§52. Per problem, over the paired problem set:

| Base | DPO | Classification |
|---|---|---|
| 0 tests passed | all passed | `complete_improvement` (§21) |
| all passed | 0 passed | `complete_regression` (§22) |
| lower rate | higher rate, not all | `partial_improvement` (§19) |
| all passed | higher than 0, fewer | `partial_regression` (§20) |
| equal | equal | `unchanged` (§23) |

Severity (§50, §51) is graded off the test-pass delta against `regression_threshold`, with
`high` reserved for the complete cases. `best_score` is the **maximum** test pass rate
across samples and `solved` is *any* candidate passing everything (§25) — deliberately not
the mean, and asserted in tests.

**`failures.py`** — §45–§49. Joins per-test results to variants and computes
`failure_count_by_test` / `failure_rate_by_test`, then flags hard tests (both variants above
`hard_test_failure_rate`), DPO-specific and base-specific difficult tests. This is where the
real run has actual content.

**`diversity.py`** — §26–§31. SHA-256 over `extracted_code` per `(variant, problem_id)`;
`diversity = unique / total`; the §27 sampling table; and the §31 mode-collapse warning when
DPO's diversity falls more than `mode_collapse_reduction` **relative** to base. §114's tests
pin the definition: 10 identical candidates → 0.1, 10 unique → 1.0.

**`coverage.py`** — §32–§42. The stage's most informative module.

- §32/§33 preference distribution per category, reusing Stage 8's `QualityReport` and
  `PreferenceStatistics.per_problem` rather than recounting.
- §35/§36/§37 category and difficulty gaps. The **training population is the train split's
  pairs**, not every pair in `metadata.jsonl` — a pair on a test-split problem was never
  trained on and counting it would overstate coverage.
- §37's `coverage_ratio` returns `float | None` beside the five-value verdict enum, so the
  `inf` and `nan` cases found in the real data serialise honestly.
- §38's error-to-data correlation, emitted with the mandated `potential data gap` wording
  and no causal claim.
- §39 preference strength, §40 the policy comparison across preference runs sharing a
  `ranking_run_id`, §41/§42 strategy distribution from `chosen_strategy`/`rejected_strategy`
  (the real enum is five values including `edge_case_focused` and `straightforward`, not the
  spec's four-value shorthand).

**`training_curve.py`** — §85–§87. Reads `metrics/metrics.jsonl` and the `FinalReport`.
Returns `insufficient_data` when fewer than two logged steps exist, which is the honest
answer here. §87's preference-overfitting check needs training-set performance that Stage 10
does not measure, so it reports `not_applicable` exactly as Stage 10's §70 clause does.

**`recommend.py`** — §53–§58, §89–§95. The `RecommendationEngine` maps observations onto
§58's ten categories, each carrying its evidence dict and hypothesis. Rules are explicit and
testable: high error rate + low coverage → `add_data` (§116); diversity collapse →
`investigate_mode_collapse`; losses exceeding wins → `investigate_regression` with
`regression_warning = true` (§89); difficulty skew → `increase_problem_difficulty`.
`adjust_dpo_hyperparameters` is **gated behind §56** — it may only be emitted when the data
checks pass and the training curve suggests an optimisation problem.

`decide_iteration(...)` returns §90's five-value decision with **`insufficient_evidence`
evaluated first** (§95): below `minimum_evidence.benchmark_problems`, or a paired CI wider
than `max_ci_width`, and no other decision may be reported as the headline. On the real run
both gates fire.

**`refinement.py`** — §59–§66, §70–§79, §104, §117. The stage's dangerous half.

- `hard_examples.jsonl` (§59), `regression_examples.jsonl` (§61),
  `successful_dpo_examples.jsonl` (§62) — each row carrying §63's mandatory provenance
  (`source_evaluation_run_id`, `problem_id`, `model_variant`, `benchmark_version`) and
  **referencing the problem by id rather than duplicating its definition** (§64).
- `refined_preferences.jsonl` (§77) — the Stage 8 pairs filtered by §71's rules (margin
  below threshold, identical code, infrastructure-tainted evaluations), written to the
  *analysis* run directory as `dpo_preference_v2` with `parent_preference_run_id` (§78,
  §79). Stage 8's files are opened read-only; §77's "do not overwrite" is guaranteed by
  never holding a writable handle to them.
- `assert_no_benchmark_leakage(...)` — the §65/§66/§104 guard. **Every refined preference
  row is checked against the benchmark's problem ids before the file is written**, and a hit
  raises `RefinementLeakageError`. §117 makes this a required test.
- §76's `retain` / `remove` / `regenerate` verdict is recorded per pair, so a removal is
  auditable rather than a silent drop (Data Integrity).

**`experiments.py`** — §81–§84. The §82 matrix across several Stage 10 runs, `best_experiment`
by held-out pass@1 (§83), and §84's explicit tradeoff callout when pass@1 improves while
pass@10 falls. Runs with different benchmarks or sample counts are flagged incomparable
rather than tabulated side by side — the two committed runs (7 problems × 10 samples vs 3 ×
1) are exactly that case.

**`report.py`** — §96–§103. `summary.json` with §96's field list; `reports/analysis.md` with
§97's fifteen sections; `improvements.md`, `regressions.md`, `data_gaps.md`;
`recommendations/next_experiment.{yaml,md}` with §101's six required fields.

Two wording rules are implemented, not merely intended: §38's *"potential data gap"* phrasing
(no causal verbs), and §99's rule that **"Likely failure" may only be printed when a
subcategory actually supports it** — otherwise the line is omitted.

**`run_repository.py`** — `AnalysisRunRepository` minting `analysis_YYYYMMDD_HHMMSS_xxxx`
(§8), owning `manifest.json` / `config.yaml` / `summary.json` and the standard status
lifecycle, with a `run_log_file` context manager teeing into `logs/analysis.log`. This is the
**seventh** copy of the run plumbing. Extraction has been deferred since Stage 7 and now
touches seven stages; the debt is carried deliberately and re-flagged in the module
docstring, consistent with every prior stage.

---

## Persistence layout (§9, §121)

```
configs/analysis/python_analysis.yaml       # thresholds; no root-config section

data/analysis/runs/analysis_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json                 # lineage chain, source run ids, status, timestamps
├── config.yaml                   # the resolved analysis config actually used
├── summary.json                  # §96
├── classifications/{base_errors,dpo_errors}.jsonl
├── classifications/{base_error_profile,dpo_error_profile,error_rate_comparison}.json
├── improvements/improvements.jsonl
├── regressions/regressions.jsonl
├── analysis/{test_failures,diversity,training_curve}.json
├── data_gaps/{category_gaps,difficulty_gaps,preference_coverage,strategy_gaps}.json
├── refined_dataset/{hard_examples,regression_examples,successful_dpo_examples}.jsonl
├── refined_dataset/{refined_preferences.jsonl,refined_preference_plan.json}
├── recommendations/{next_experiment.yaml,next_experiment.md,recommendations.json}
├── reports/{analysis,improvements,regressions,data_gaps}.md
└── logs/analysis.log
```

All text, all derived. Everything here is recomputable from the Stage 10 run plus the
problem dataset, so committing a run directory is cheap and consistent with prior stages.

---

## Modifications to existing code

**`src/python_dpo/config.py` + `config.yaml` + `data/` + `.gitignore`** — the **tenth** data
path, `analysis`, threaded through `_REQUIRED_PATH_KEYS`, the `Paths` dataclass, `Paths.
ensure_exists()`, `Config.load()`'s `Paths(...)` construction, `data/analysis/.gitkeep`, and
`!data/analysis/runs/*/logs/*.log` to override the global `*.log` rule. No `analysis:` section
in the root config — Stage 11 follows Stages 9 and 10 in keeping tunables in a standalone file.

**`src/python_dpo/cli.py`** — one new command group, shaped like `evaluate-model` (a group
with a default action) so §106's bare form coexists with §107–§111's subcommands:

| Command | Spec | Behavior |
|---|---|---|
| `analyze --evaluation-run-id ID [--preference-run-id ID] [--training-run-id ID] [--smoke-test] [--config PATH]` | §106, §112 | Full analysis; creates the run directory and every artifact |
| `analyze errors --evaluation-run-id ID` | §107 | Error profiles and the rate comparison |
| `analyze data-gaps --evaluation-run-id ID --preference-run-id ID` | §108 | Category, difficulty, coverage and strategy gaps |
| `analyze recommend --evaluation-run-id ID` | §109 | Recommendations and the iteration decision |
| `analyze refine --evaluation-run-id ID` | §110 | `next_experiment.yaml` + refined dataset. **Never trains** |
| `analyze compare --evaluation-runs A,B,C` | §111 | The §82 experiment matrix |
| `analyze list` / `analyze show --analysis-run-id ID` | — | Discovery; nothing above is usable without a run id |

Handlers keep the house contract: `(args, config) -> int`, results to `sys.stdout.write`,
errors via `logger.error("%s", exc)`, exit codes 0/1/2/130.

**`src/python_dpo/__init__.py`** — `__version__` → `0.11.0`.

**`tests/test_no_heavy_imports.py`** — every `analysis` submodule added to `_PROBE`.

**`tests/evaluation/test_config.py`, `tests/sandbox/test_config.py`** — their inline
fixture YAML gains `paths.analysis`, as happened for `model_evaluations` in Stage 10.

**`pyproject.toml`** — **no new dependencies.** Everything is stdlib plus the PyYAML already
required for config and `next_experiment.yaml`.

**Docs** — `src/python_dpo/analysis/README.md` (new), plus Stage 11 sections in `README.md`,
`src/python_dpo/README.md`, `data/README.md`, `tests/README.md`, and the
`11_ERROR_ANALYSIS_AND_ITERATION.md` implementation report at the repo root.

---

## Tests

One tier — **everything runs in the default offline suite**, with no `integration` or `gpu`
marker anywhere, because the stage touches no Docker and no GPU. `tests/analysis/`:

- **`test_taxonomy.py`** — §113's three named cases (syntax error → `syntax_error`;
  `TypeError` → `runtime_error`/`TypeError`; timeout → `timeout`), plus
  `MemoryError` → `memory_error`, an unrecognised exception → `other` (the real data's
  `Failed`), `generation_failure` vs `code_extraction_failure` separation, and the
  most-frequent-wins subcategory rule with an alphabetical tie-break.
- **`test_outcomes.py`** — §113's four outcome cases exactly: 0/10 → 10/10 complete
  improvement; 10/10 → 0/10 complete regression; 3/10 → 7/10 partial improvement; 5/10 →
  5/10 unchanged. Plus §20's partial regression, and §25's `best = max` asserted against a
  fixture where the mean would give a different answer.
- **`test_diversity.py`** — §114's two cases (10 identical → 0.1, 10 unique → 1.0), plus the
  §88 threshold firing and *not* firing across the boundary.
- **`test_coverage.py`** — §115's worked example (training 5%, evaluation 25% →
  `coverage_ratio = 0.2`, `underrepresented`), plus the two degenerate cases from the real
  data: benchmark share 0 → `not_in_benchmark` with ratio `None`, and 0/0 →
  `absent_from_both` — asserting the emitted JSON contains no `Infinity` or `NaN`.
- **`test_failures.py`** — §46's frequencies and §47/§48/§49's three flags, on a fixture
  reproducing the real run's seven interesting tests.
- **`test_recommend.py`** — §116 (high error rate + low coverage → `add_data`); that a
  recommendation with an empty `evidence` dict or empty `hypothesis` is **rejected at
  construction** (§55, §103); §56's rule that `adjust_dpo_hyperparameters` is not emitted
  from data-shaped evidence; §57's ordering; and §90's decision precedence — that
  `insufficient_evidence` wins over `refine_data` when the evidence gates fail.
- **`test_refinement.py`** — §117's leakage test as the centrepiece: a benchmark problem
  cannot reach `refined_preferences.jsonl`, asserted by attempting it and expecting
  `RefinementLeakageError`. Plus §63's provenance on every row, §77's non-overwrite (Stage 8
  files byte-identical after a refine), and §78/§79's versioning and parent id.
- **`test_report.py`** — §96's summary field list; that §97's fifteen sections are all
  present; §38's wording rule (no causal verb appears in a data-gap paragraph); and §99's
  rule that "Likely failure" is absent when no subcategory supports it.
- **`test_ingest.py`** — lineage resolution across the manifest chain, and `LineageError` on
  a broken hop; the Stage 6-vs-Stage 10 `evaluation_run_id` collision handled correctly.
- **`test_models.py`** / **`test_config.py`** / **`test_run_repository.py`** — round-trips,
  threshold validation, run-id minting and the status lifecycle.
- **`test_integration.py`** — §118's full scenario, built as fixtures: 20 problems, base 8
  solved, DPO 11 solved, failures concentrated in recursion / DP / edge cases, against a
  preference distribution of 2% / 5% / 8%. Asserts the analysis identifies those as data
  gaps and that the pipeline produces every §121 artifact. This is the only place the
  non-degenerate paths are exercised end to end.
- **`tests/test_project.py`** — the tenth data path in all three enumerations; `analyze`
  parsing, the bare-group help path, `analyze` absent from `_PLACEHOLDER_STAGES`, and the
  unknown-run-id error path.

---

## Execution order

1. Write this plan to `.claude/plans/11_error_analysis_and_iteration_plan.md` and add its
   entry to `.claude/plans/README.md`.
2. `errors.py`, `config.py`, `models.py` + tests — pure schema, including the
   `Recommendation` validation that makes §55/§103 structural.
3. `taxonomy.py` + tests. **First real logic**, because §113 gives exact expected values and
   every downstream profile depends on the categories being right.
4. `ingest.py` + tests — lineage first, since §7 makes it a precondition for everything else.
5. `classification.py`, `outcomes.py`, `failures.py`, `diversity.py` + tests — the four
   modules that need only the Stage 10 run.
6. `coverage.py` + tests, including the two degenerate-ratio cases. The most valuable module
   on this dataset, and the one with the subtlest denominator choice.
7. `training_curve.py` + tests.
8. `recommend.py` + tests, including §90's decision precedence.
9. `refinement.py` + tests — **the leakage test before the writer**, so the guard is proven
   before any file that could carry a benchmark problem is ever produced.
10. `experiments.py`, `report.py`, `run_repository.py` + tests.
11. Config path wiring (tenth path, three `test_project.py` enumerations, two fixture YAMLs),
    CLI wiring, `test_no_heavy_imports.py` extension.
12. Run §120's five verification commands; commit the analysis run; docs; the §125 report.

---

## Verification

```bash
source .venv/bin/activate
pytest -q                     # offline, zero skips, no new markers

# §120's procedure, in order
python -m python_dpo analyze --evaluation-run-id eval_20260818_155511_1633
python -m python_dpo analyze errors --evaluation-run-id eval_20260818_155511_1633
python -m python_dpo analyze data-gaps \
    --evaluation-run-id eval_20260818_155511_1633 \
    --preference-run-id pref_20260818_074347_5eff
python -m python_dpo analyze recommend --evaluation-run-id eval_20260818_155511_1633
python -m python_dpo analyze refine   --evaluation-run-id eval_20260818_155511_1633
```

**Expected, computed from the committed artifacts before writing any code:**

```
lineage    eval_20260818_155511_1633
             -> dpo_20260818_081231_a91d
             -> pref_20260818_074347_5eff
             -> rank_20260817_161726_a84d
             -> 20260817_055411

outcomes   0 improvements · 0 regressions · 7 unchanged
errors     base {assertion_failure: 16}   dpo {assertion_failure: 15}
           every other category 0 · subcategory AssertionError (97) + Failed (1)
tests      hard(both)     p002_t002 t005 t006 t007 (100%/100%) · p004_t002 (50%/50%)
           dpo-specific   p004_t008 (30% -> 40%)
           base-specific  p005_t005 (10% -> 0%)
diversity  base 22/70 = 0.314 · dpo 20/70 = 0.286 · -8.9% relative
           mode_collapse_warning = false (below the 20% gate)
coverage   all 7 benchmark categories coverage_ratio 0.00 -> underrepresented
           edge_cases, exceptions -> not_in_benchmark (ratio None)
           difficulty easy 2.33 over · medium 0.00 under · hard absent_from_both
training   1 logged step, loss 0.6931 (ln 2), rewards/margins 0.0 -> insufficient_data
decision   insufficient_evidence   (7 problems < 30; paired CI [0.0, 4.3]pp)
leakage    refined_preferences.jsonl contains 0 benchmark problems
```

Then the assertions that matter most, checked against the emitted files:

```bash
RUN=data/analysis/runs/<analysis_run_id>

# §65, §66, §104, §117 — no benchmark problem reached the refined dataset
python - <<'PY'
import json, pathlib
run = sorted(pathlib.Path("data/analysis/runs").iterdir())[-1]
bench = set(json.load(open("benchmarks/python_eval_v1/manifest.json"))["problem_ids"])
rows = [json.loads(l) for l in open(run / "refined_dataset" / "refined_preferences.jsonl")]
assert not (bench & {r["problem_id"] for r in rows}), "BENCHMARK LEAKAGE"
print(f"{len(rows)} refined pairs, 0 benchmark problems")
PY

# §37 — no Infinity/NaN escaped into JSON
grep -l 'Infinity\|NaN' $RUN/data_gaps/*.json && echo FAIL || echo "clean"

# §77 — Stage 8's dataset was not touched
git diff --exit-code data/preferences/ && echo "stage 8 untouched"

# §5, §113 — nothing was trained
test -z "$(git status --porcelain data/training/)" && echo "no training occurred"
```

The honest reading, stated up front in `analysis.md` rather than buried: on a 7-problem
benchmark where 5 problems sit at the base model's ceiling, against an adapter that trained
for one optimizer step, **no outcome-level conclusion is available**. The one finding the
evidence does support is structural rather than statistical — the DPO training split and the
evaluation benchmark share **zero categories**, so nothing that was trained was ever
measured. That is a pipeline-design observation, and it is what the recommendation set
should lead with.
