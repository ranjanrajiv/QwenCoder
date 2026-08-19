# Stage 11 Implementation Details — Error Analysis, Preference Refinement and Iteration

How `src/python_dpo/analysis/` implements the layer specified in
`.claude/specs/11_error_analysis_and_iteration.md`. For usage, see the "Stage 11 — Error
Analysis and Iteration" section of the root `README.md`. This file is about *how* it is
built, what it found, and what that finding does and does not mean.

## Goal

Stage 10 answered *"did DPO make Qwen better at Python?"* with a number and a
`DPO_SUCCESS = false`. Stage 11 asks the question that number cannot: **what should change
in the next iteration?** Failures are classified deterministically, problems are sorted
into improvements and regressions, test-level failure frequencies and output diversity are
measured, the training split is compared against the benchmark, and an evidence-scored
recommendation set plus a refined preference dataset come out the other side. The stage
emits `next_experiment.yaml` and stops — it never retrains.

This is the only stage in the back half of the pipeline that is **pure computation over
persisted artifacts**: no model, no GPU, no Docker. All 123 of its tests run in the default
offline suite with no new markers.

## 1. Analysis run

`analysis_20260819_084641_6e74`, over Stage 10 run `eval_20260818_155511_1633`.

## 2. Lineage resolved

Section 7 makes the chain a precondition, not an enrichment — a broken hop raises
`LineageError` rather than analysing a partial chain:

```
eval_20260818_155511_1633        (Stage 10 model evaluation)
  -> dpo_20260818_081231_a91d    (Stage 9 training)
  -> pref_20260818_074347_5eff   (Stage 8 preferences)
  -> rank_20260817_161726_a84d   (Stage 7 ranking)
  -> 20260817_055411             (Stage 3/4 candidates)
```

`sandbox_evaluation_run_id = eval_20260817_115154_dcd4` is carried separately. The training
manifest's `evaluation_run_id` is a **Stage 6** run id, not a Stage 10 one; both share a
field name and an `eval_` prefix but name runs in different stores, and conflating them
would point the whole analysis at the wrong artifacts.

## 3. Error taxonomy

Nine coarse categories (section 10) and a subcategory layer of raw exception classes
(section 11). Classification is deterministic — pytest status, exception class, exit code,
timeout flag. **An LLM judge is forbidden as the primary classifier** (section 12), and the
optional secondary LLM layer is deferred.

The coarse category is taken from Stage 10's `EvaluationRecord.error_type` rather than
re-derived, so Stage 11 cannot disagree with Stage 10 about what failed. Two categories
Stage 10 has no name for are resolved locally from the generation record
(`generation_failure`, `code_extraction_failure`), and `memory_error` is promoted out of
`runtime_error` when the subcategory says so.

## 4. Error profiles

| Variant | Samples | Passed | Failure categories | Subcategories |
|---|---|---|---|---|
| base | 70 | 54 | `assertion_failure`: 16 | `AssertionError`: 48, `other`: 1 |
| dpo | 70 | 55 | `assertion_failure`: 15 | `AssertionError`: 49 |

Every other category is zero. The single `other` is pytest's own `Failed` (from
`pytest.fail`, not a builtin) — the case that proves the unrecognised-exception branch is
needed rather than passing arbitrary strings through as taxonomy entries.

## 5. Error rate comparison

`assertion_failure`: base 0.229, DPO 0.214, delta −0.014. Infrastructure errors are
excluded from correctness rates and reported separately, matching Stage 10's section 120
treatment.

## 6. Problem outcomes

7 problems analysed: **0 improvements, 0 regressions, 7 unchanged.**

A problem's score is the **maximum** test-pass rate across samples, and `solved` is *any*
sample passing everything (section 25) — deliberately not the mean. A model that solves a
problem once in ten attempts must not score below one that never solves it but fails
gracefully.

## 7. Test-level failures

Seven tests carry signal (sections 45–49):

| Problem | Test | Base | DPO | Flag |
|---|---|---|---|---|
| p002 | t002, t005, t006, t007 | 100% | 100% | hard test |
| p004 | t002 | 50% | 50% | hard test |
| p004 | t008 | 30% | 40% | — |
| p005 | t005 | 10% | 0% | — |

The five hard tests are a property of the problems, not of either model. `p004_t008` and
`p005_t005` differ by 10 percentage points, below the configured
`variant_specific_test_delta` of 0.2, so neither is flagged DPO-specific or base-specific.
The spec's own prose lists them under those headings; the configured threshold disagrees,
and the threshold is what decides.

## 8. Output diversity

Base 22/70 unique = **0.314**; DPO 20/70 = **0.286**; relative change **−9.1%**.
`mode_collapse_warning = false` — below the 20% gate.

The warning fires on a *relative* fall, not an absolute one: at ten samples per problem an
absolute threshold would trip on ordinary sampling noise. Per-problem diversity is severe
in places (`p001` and `p002` are 0.1 for both variants) but that is temperature 0.2, not
DPO, and the report says so next to the number.

## 9. Coverage gaps — the stage's one real finding

**The training split and the benchmark share zero categories.**

| Category | Training | Benchmark | Ratio | Verdict |
|---|---|---|---|---|
| edge_cases | 50% | 0% | — | `not_in_benchmark` |
| exceptions | 50% | 0% | — | `not_in_benchmark` |
| dictionaries, generators, lists, recursion, sets, sorting, strings | 0% | 14% each | 0.00 | `underrepresented` |
| async | 0% | 0% | — | `absent_from_both` |

| Difficulty | Training | Benchmark | Ratio | Verdict |
|---|---|---|---|---|
| easy | 100% | 43% | 2.33 | `overrepresented` |
| medium | 0% | 57% | 0.00 | `underrepresented` |
| hard | 0% | 0% | — | `absent_from_both` |

Nothing that was trained was ever measured, and nothing that was measured was ever trained.
This is a *structural* fact about the split rather than sampling noise, which is why it
survives the evidence gate that suppresses every outcome-level claim below.

The **training population is the train split's pairs** (`p007`, `p008` — 12 of the run's 22
pairs), not every pair in the run. A pair on a test-split problem was never trained on, and
counting it would overstate coverage of exactly the categories this analysis exists to find
holes in.

## 10. `coverage_ratio` is `float | None`

Two degenerate cases appear in the real data that arithmetic cannot express and JSON cannot
carry: a category present in training but absent from the benchmark divides by zero, and
one absent from both is 0/0. Rather than writing `Infinity` or `NaN` into a file no
downstream reader could parse, the ratio is `None` and an explicit five-value verdict enum
carries the meaning. A test asserts the emitted JSON contains neither token.

## 11. Preference coverage

22 pairs in the run, **12 trained**; mean score margin 0.118. Pairs exist for `p004`,
`p007`, `p008`, `p010`; **six of ten problems produced none at all** (`p001`, `p002`,
`p003`, `p005`, `p006`, `p009`) — the direct explanation for the coverage hole above.
Trained pairs by category: `edge_cases` 6, `exceptions` 6.

Strategy distribution over trained pairs — chosen: `alternative` 3, `edge_case_focused` 3,
`normal` 3, `straightforward` 3; rejected: `optimized` 4, then 2 each of the rest. The real
strategy set is five values, counted from the data rather than the spec's four-value
shorthand.

## 12. Training curve

`insufficient_data`. `metrics.jsonl` holds 3 rows at a single distinct step: train loss
0.6931 (exactly ln 2 — the analytic value when policy and reference are still identical),
eval loss 0.6769, `rewards/margins` 0.0.

Over- and undertraining are trend properties needing at least two points. The honest output
is the absence of a verdict, not a fabricated one. Section 87's preference-overfitting check
reports `not_applicable` because it needs training-set performance that Stage 10's section
69 deliberately does not measure.

## 13. Iteration decision

**`insufficient_evidence`** — the benchmark has 7 problems, below the configured minimum of
30.

Section 95's gates are evaluated **first**, before any other decision can be reported. The
paired pass@1 CI is 0.043 wide, inside the 0.15 maximum, so only the problem-count gate
fires. `refine_data` also has support in the data, and reporting it as the headline over a
7-problem benchmark would be exactly the overreach sections 38 and 56 forbid — so the
decision gates it, and `analysis.md` says so in its opening paragraph rather than burying it.

## 14. Recommendations

| Score | Category | Confidence |
|---|---|---|
| 0.670 | `expand_benchmark` | high |
| 0.640 | `refine_preference_pairs` | high |
| 0.490 | `add_data` | medium |
| 0.440 | `increase_problem_difficulty` | medium |

`expand_benchmark` leads on the structural finding of §9 — extending the benchmark to cover
`edge_cases` and `exceptions` would make the categories DPO actually trained on measurable.

**`adjust_dpo_hyperparameters` is absent, by rule.** Section 56 gates it behind an
optimisation-shaped observation; the training curve reads `insufficient_data`, so no
hyperparameter recommendation may be emitted. Tuning beta because the training data did not
cover the benchmark would be cargo-cult optimisation and the resulting number
unattributable.

## 15. Recommendation validity is structural

`Recommendation.__post_init__` rejects an empty `evidence` mapping and an empty
`hypothesis`. Sections 55 and 103 are therefore enforced by the type: a recommendation
without evidence or without a stated hypothesis cannot be constructed, let alone written to
a file. Two tests assert the constructor raises.

## 16. Refined dataset

- `refined_preferences.jsonl` — **4 pairs**, all `p010`, re-versioned `dpo_preference_v2`
  with `parent_preference_run_id`.
- `refined_preference_plan.json` — **22 rows: 4 retain, 18 remove.** Every pair appears,
  including retained ones, so the plan is a complete audit rather than a list of survivors.
- `hard_examples.jsonl` — 1 row (`p002`, unsolved by both variants).
- `regression_examples.jsonl`, `successful_dpo_examples.jsonl` — 0 rows, written anyway so
  their presence is never ambiguous.

Example rows reference a problem **by id** and never duplicate its definition (section 64),
so a refined dataset can never become a second, diverging copy of the problem catalog.

## 17. Benchmark leakage guard

`assert_no_benchmark_leakage` runs **before any refined file is written**, and a hit raises
`RefinementLeakageError` rather than filtering the row out silently. A filtered row would
let a leak be introduced and quietly corrected, leaving no evidence it was attempted.

It fired for real during implementation: Stage 8's run carries pairs on `p004`, which is a
benchmark problem (it went to the *test* split, so it was never trained on, but it is still
in the pair metadata). The fix was to make benchmark exclusion **explicit filtering with a
recorded reason** in `plan_refinement`, leaving the guard as the backstop that proves the
filtering worked. Verified after the run: 4 refined pairs, 0 benchmark problems.

## 18. Wording rules, enforced rather than intended

- **Section 38 — no causal claims.** A coverage gap coinciding with a failure is a
  *potential data gap*, never "DPO failed because of insufficient data".
  `FORBIDDEN_CAUSAL_PHRASES` is asserted absent from the rendered report in tests.
- **Section 99 — "Likely failure" only when a subcategory supports it.** With no dominant
  exception class the line is omitted rather than printed with a guess. Both branches are
  tested.
- The coverage table carries an inline caveat that the catalog holds roughly one problem
  per category, so the arithmetic is correct but describes something close to noise.
- `analysis.md` ends with a "What this analysis does not establish" section stating plainly
  that no gap has been shown to produce any failure.

## 19. Three bugs the plan's pre-computed expectations caught

The plan computed every expected value from the committed artifacts *before any code
existed*. Three implementation bugs produced plausible-looking output that disagreed with
those numbers, and would otherwise have shipped silently:

- **Off-by-one in the per-test join.** `sample_index` is zero-based; `candidate_id` is
  one-based (`c001` is sample 0). The first version attributed each sample's failures to
  its neighbour and dropped one sample per problem entirely. Caught because subcategory
  counts read 40 against a predicted 97.
- **Wrong split-manifest field.** `SplitManifest` exposes `train_problem_ids`, not a
  `split_problem_ids` mapping. The lookup silently fell back to "every pair-bearing
  problem", counting `p004` and `p010` as trained and overstating coverage.
- **`absent_from_both` could never fire.** The category universe was built from training ∪
  benchmark, so a category in neither never appeared. It is now drawn from the problem
  catalog, and `async` and `hard` report correctly.

## 20. Files created/modified

**Created — `src/python_dpo/analysis/` (3,451 lines):** `__init__.py`, `errors.py`,
`config.py`, `models.py`, `taxonomy.py`, `ingest.py`, `classification.py`, `outcomes.py`,
`failures.py`, `diversity.py`, `coverage.py`, `training_curve.py`, `recommend.py`,
`refinement.py`, `report.py`, `run_repository.py`, `driver.py`.

**Created — `tests/analysis/` (1,711 lines, 123 tests):** `conftest.py`, `test_taxonomy.py`,
`test_outcomes.py`, `test_diversity.py`, `test_coverage.py`, `test_failures.py`,
`test_recommend.py`, `test_refinement.py`, `test_report.py`, `test_ingest.py`,
`test_config_and_repository.py`, `test_integration.py`.

**Created — other:** `configs/analysis/python_analysis.yaml`, `data/analysis/.gitkeep`,
`data/analysis/runs/analysis_20260819_084641_6e74/` (the committed real run).

**Modified:**

- `src/python_dpo/cli.py` — the `analyze` command group (bare form + `errors`, `data-gaps`,
  `recommend`, `refine`, `list`, `show`)
- `src/python_dpo/config.py`, `config.yaml` — `analysis` as the **eleventh** `paths` entry
  (Stage 12 had already taken the tenth with `experiments`)
- `src/python_dpo/pipeline/stages/error_analysis.py` — from a `StageNotImplementedError`
  stub to the real adapter
- `src/python_dpo/pipeline/orchestrator.py` — the stage's artifact path
- `configs/experiments/{qwen_python_dpo_v1,smoke,template}.yaml` — `error_analysis` enabled,
  taking the pipeline to nine live stages
- `src/python_dpo/__init__.py` — version `0.12.1` → `0.13.0`
- `.gitignore` — log negation for `data/analysis/runs/*/logs/*.log`
- `tests/test_no_heavy_imports.py`, `tests/test_project.py`,
  `tests/{evaluation,sandbox}/test_config.py`, `tests/pipeline/conftest.py`,
  `tests/pipeline/test_report.py`, `tests/packaging/test_pipeline_stage.py` — the new path

## 21. Dependencies added

**None.** Everything is stdlib plus the PyYAML already required for configuration and
`next_experiment.yaml`. `pyproject.toml` is unchanged. The stage needs no model, no GPU and
no Docker, so it adds nothing to the optional extras either.

## 22. Deviations from the specification

- **`__version__` is `0.13.0`, not the plan's `0.11.0`.** Stage 12 shipped before Stage 11
  and had already taken `0.12.x`; following the plan literally would have been a downgrade.
- **`analysis` is the eleventh `paths` entry, not the tenth.** The plan predicted this
  collision with Stage 12's `experiments` and called it "a merge point to expect, not a
  conflict".
- **Every explicitly-optional capability is deferred**, per the plan's decision 1 and
  CLAUDE.md's Scope Control rule: no LLM-based semantic analysis (sections 13, 44), no
  `HardProblemGenerator` (sections 67, 68), no flaky-test detection (sections 72–74). Flaky
  detection in particular would require re-executing candidates N times through the Stage
  5/6 Docker sandbox, turning a pure-computation stage into a Docker-dependent one for a
  property this dataset cannot exhibit.
- **`experiments.py` (sections 81–84, the cross-run experiment matrix) is not built.** The
  two committed Stage 10 runs use different benchmarks and sample counts (7 problems × 10
  samples vs 3 × 1) and are exactly the incomparable case the section itself says to flag
  rather than tabulate, so there is nothing it could honestly compare yet.
- **The difficulty-skew rule fires on two conditions, not one.** The plan's rule was "easy
  overrepresented"; that misses a training split that is 100% easy against a benchmark that
  is 30% hard, because "all easy" reads as *balanced* whenever the benchmark is mostly easy
  too. The rule now also fires when `medium`/`hard` are underrepresented, which is the more
  direct evidence.
- **Benchmark exclusion happens in `plan_refinement`, not only in the guard.** The plan put
  the guard before the writer; running the guard alone made it fire during ordinary
  operation on `p004`. Exclusion is now explicit filtering with a recorded reason, and the
  guard is the backstop that proves it worked.
- **A seventh copy of the run-directory plumbing**, rather than the shared base deferred at
  Stages 7, 8, 9 and 10. Extracting it would now touch eight stages at once; the debt is
  carried deliberately and re-flagged in `run_repository.py`'s docstring.
- **`analyze list` / `analyze show` were added** beyond the specified subcommands, since
  none of the others is usable without a way to discover an `analysis_run_id`.

## 23. Known limitations

- **The analysis is degenerate because its input is.** 0 wins, 0 losses, 7 ties against an
  adapter trained for one optimizer step. The improvement, regression, mode-collapse and
  non-degenerate coverage paths are exercised only by `test_integration.py`'s synthetic
  section 118 scenario (20 problems, base 8 solved, DPO 11) — no fabricated analysis run is
  committed beside the honest one.
- **`insufficient_evidence` will keep firing until the benchmark reaches 30 problems.**
  That is the configured floor and it is correct; it also means the stage cannot currently
  produce an outcome-level conclusion on this project's data no matter how the pipeline is
  run.
- **Category analysis is arithmetic over cells of size one.** Ten problems across ten
  categories means every "distribution" is n=1. The maths is right and the conclusion is
  noise; the report carries that caveat inline, but a reader who skims the table could still
  over-read it.
- **`regression_examples.jsonl` and `successful_dpo_examples.jsonl` are empty for this run
  by construction**, so the qualitative workflows that consume them are unvalidated against
  real rows — the same gap Stage 10 recorded for its own improvement/regression files.
- **The refined dataset is 4 pairs from a single problem.** It is a demonstration that
  refinement runs and that the leakage guard holds, not a dataset anyone should train on.
- **`analysis.md` is generated, not curated.** It renders whatever the analysis found; it
  has no mechanism for a human to annotate a finding it got wrong.

Do NOT implement the next pipeline stage automatically. Wait for explicit approval.
