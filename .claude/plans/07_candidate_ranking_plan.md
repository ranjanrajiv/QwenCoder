# Stage 7 — Candidate Evaluation, Scoring and Ranking

## Context

Stage 6 answered *"what happened when this candidate was tested?"* and stopped there,
deliberately: `data/evaluations/runs/eval_20260817_115154_dcd4/` now holds 50 candidates'
worth of objective execution evidence — counts, statuses, durations, error types — and
nothing that says *better* or *worse*.

`.claude/specs/07_candidate_ranking.md` asks for the layer that turns that evidence into a
judgement: classify each candidate `correct`/`incorrect`/`indeterminate`, score it, and
rank candidates **within each problem** so Step 8 can build DPO preference pairs from an
objective ordering.

Two boundaries define the stage:

- **Objectivity (§2, §56).** The ranking signal is test-case performance, full stop. No LLM
  judge, no random ordering, no timestamps influencing order, no heuristic overriding
  execution results. Given identical evidence and configuration, the output must be
  byte-identical (§46).
- **Scope (§3, §81).** Step 7 produces assessments, rankings and pairwise comparisons. It
  produces **no** `chosen`/`rejected` labels, no DPO JSONL, no preference pairs. Neutral
  terminology only (§67). Step 8 applies preference policy — and §65 requires Step 7 expose
  enough for *several* policies rather than hard-coding one.

**Outcome:** `rank run --evaluation-run-id eval_20260817_115154_dcd4` produces
`data/rankings/runs/rank_.../` with per-candidate assessments, per-problem rankings,
pairwise comparisons, and statistics.

### What exploration established

The committed evaluation run was measured directly. These numbers change the emphasis of
the stage:

| Finding | Consequence |
|---|---|
| **6 of 10 problems yield zero ordering** — p001/p003/p005/p006/p009 have all 5 candidates at pass_rate 1.0; p002 has all 5 failing at the *same* 3/7 | Tie handling (§29, §35, §62) is the **majority** behaviour, not an edge case |
| 100 candidate pairs, **only 22 non-tied (22%)** | §35's "never invent a preference" is what mostly governs this dataset |
| **49 of 50 candidates sit in a tie group of size > 1** (only `p010_c003` is a singleton) | `tie_group` is load-bearing, not decoration |
| No problem has more than **2 distinct pass rates** | The ranking ladder is shallow; ranks are 1/4, 1/3, 1/5 shapes |
| **Zero** timeouts, syntax errors, infrastructure errors in the real data | The `indeterminate` path (§12, §14, §64) has *no* real coverage — it must be driven by synthetic fixtures |
| 3 records have `tests_error > 0` (p008_c001/c002/c005) with `runtime_error=true` | The only place "crashes" differs from "wrong answer"; the classifier must treat both as `incorrect` (§11) |
| **31 of 50 candidates have non-null `duplicate_of`; only 19 unique `code_sha256`** | The 22 non-tied pairs collapse to ~7 genuinely distinct ones. §69 says duplicate code is "possible… do not automatically treat as an error" — but Step 8 needs the means to dedup |
| The evaluation↔candidate join is **exact and total** (50/50 on candidate_id and problem_id) | A candidate join is safe to rely on |
| `syntax_valid`/`function_name_valid`/`extraction_format` are **constant** across all 50; `strategy` is a pure function of the candidate suffix | These are metadata only — which is exactly what §20/§23/§57 mandate. Recording them must not tempt anyone into scoring with them |
| `EvaluationResult.is_candidate_outcome` already exists (status != infrastructure_error) | Reuse it rather than re-deriving the infrastructure test |
| `compute_pass_rate(passed, total)` already exists in `evaluation/models.py` | Reuse; it returns 0.0 for total==0 |
| **`config.py` has no `rankings` path** — `_REQUIRED_PATH_KEYS` and `Paths` list six dirs, and two tests in `tests/test_project.py` enumerate them literally | A seventh path must be threaded through config, `config.yaml`, `data/`, both tests, and `data/README.md` |

### Decisions confirmed with the user

1. **`CandidateAssessment` joins back to the generation run's candidates** and records
   `code_sha256`, `duplicate_of`, `code_lines`, `code_chars`, `strategy`, `syntax_valid` as
   *unused* secondary metadata (§20/§22/§23/§57). The decisive reason is the duplicate
   finding above: without `code_sha256`/`duplicate_of` on the assessment, Step 8 would
   generate ~22 pairs that collapse to ~7 distinct ones.
2. **Pairwise comparisons are persisted** to `comparisons.jsonl` (§42 optional, §58 "unless
   useful"). 100 rows for this run; it hands Step 8 `score_margin` ready-made for its margin
   policies (§65, §68) and makes §74's expectations auditable on disk.
3. **A candidate with no evaluation result becomes an `indeterminate` assessment carrying an
   explicit reason** (§70), never silently skipped (§71, CLAUDE.md Data Integrity).
   `rankings validate` surfaces it as an issue.
4. **Run-directory plumbing mirrors the existing pattern** — a third `run_repository.py`
   modelled on `evaluation/run_repository.py` rather than extracting a shared base and
   refactoring Stages 4 and 6 onto it. Zero risk to working, tested code; the duplication is
   recorded honestly in the report as a deliberate deferral.

---

## New package — `src/python_dpo/ranking/`

House style throughout: frozen dataclasses validating in `__post_init__`, explicit
`to_dict()`/`from_dict()` rejecting unknown and missing fields against a module-level
frozenset, per-folder `README.md`.

**`errors.py`** — `RankingError` base; `RankingConfigError`, `EvaluationRunNotFoundError`,
`RankingRunNotFoundError`, `RankingStoreError`. Stage 6 errors are translated, not leaked.

**`models.py`** — the schema, plus the three closed sets.

```
CORRECTNESS = {correct, incorrect, indeterminate}          # §9
RELATIONS   = {A_BETTER, B_BETTER, TIE, INDETERMINATE}     # §33
RANKING_RUN_STATUSES = {created, running, completed, failed, interrupted, cancelled}
```

`CandidateAssessment` (§24) — provenance (`ranking_run_id`, `evaluation_run_id`,
`candidate_run_id`, `candidate_id`, `problem_id`), the classification (`correctness`,
`all_tests_passed`, `pass_rate`, `score`), the raw counts copied from the evaluation
(`tests_total/passed/failed/error/skipped`), the flags (`timeout`,
`infrastructure_error`), `execution_duration_ms`, `indeterminate_reason: str | None`, the
decision-1 secondary metadata (`code_sha256`, `duplicate_of`, `code_lines`, `code_chars`,
`strategy`, `syntax_valid`), and `created_at`.

`__post_init__` enforces the invariants the validator will independently re-check (§51,
§52): `score == pass_rate`; `pass_rate == tests_passed / tests_total` (or 0.0 when
`tests_total == 0`); `all_tests_passed` true **iff** `tests_total > 0 and tests_passed ==
tests_total`; `correctness == "indeterminate"` **iff** `indeterminate_reason is not None`.

`RankingResult` (§25) — `ranking_run_id`, `evaluation_run_id`, `problem_id`,
`candidate_id`, `rank: int | None`, `score`, `correctness`, `pass_rate`,
`all_tests_passed`, `tie_group: str | None`, `tie_group_size: int`, `tied: bool`,
`eligible_for_preference: bool`, `created_at`.

`ComparisonResult` (§67) — `ranking_run_id`, `problem_id`, `candidate_a`, `candidate_b`,
`relation`, `score_a`, `score_b`, `score_margin`, `correctness_a`, `correctness_b`,
`comparison_eligible: bool`. **Never** `chosen`/`rejected` (§67).

`RankingManifest` (§27) — `ranking_run_id`, `evaluation_run_id`, `candidate_run_id`,
`status`, `created_at`/`started_at`/`completed_at`, `ranking_version`, `scoring_version`,
`comparator_version`, `scoring_configuration: dict`, `requested_problem_ids`, `error`.
`with_status()` enforces the same transition graph shape as Stage 4/6.

`RankingStatistics` (§40, §41) — the ten §40 counters (`problems`, `candidates`,
`correct`, `incorrect`, `indeterminate`, `fully_correct`, `partially_correct`,
`zero_test_pass`, `tied_candidates`, `preference_eligible_candidates`) plus
`per_problem: dict[str, dict]` carrying §41's distribution, `computed_at`, and a trailing
`statistics_version = "1.0"`. `from_records(manifest, assessments, rankings, *,
computed_at=None)` — keyword-only `computed_at` so the validator can recompute against the
on-disk timestamp and compare the whole dataclass with `==`, exactly as
`runs/validation.py` already does.

**`classifier.py`** — `CorrectnessClassifier`, `CLASSIFIER_VERSION = "v1"`.

`classify(result: EvaluationResult | None) -> (correctness, reason)`, in precedence order:

| Condition | Result | Spec |
|---|---|---|
| No `EvaluationResult` at all (an `EvaluationFailure`, or absent) | `indeterminate`, reason from the failure's `error_type` | §70, decision 3 |
| `infrastructure_error` or `status == "infrastructure_error"` | `indeterminate`, reason `"infrastructure_error"` | §12, §14 |
| `tests_total == 0` | `indeterminate`, reason `"no_tests_executed"` | §10, §12, §72 |
| `tests_passed == tests_total` and failed/error/skipped all 0 and not `timeout` | `correct` | §10 |
| otherwise (includes `timeout`, `tests_error > 0`, `tests_skipped > 0`) | `incorrect` | §11, §13 |

The ordering is what makes §13 true: a candidate-caused timeout reaches the final row and
is `incorrect`, never `indeterminate`. §10's requirement that `tests_skipped == 0` for
`correct` means a skipped test makes a candidate `incorrect` — worth stating explicitly
since it is easy to read past.

**`scorer.py`** — `CandidateScorer`, `SCORING_VERSION = "v1"`.

`score = pass_rate = tests_passed / tests_total` (§16, §18), reusing
`evaluation.models.compute_pass_rate`. `all_tests_passed` stays a separate binary signal
(§17). Nothing else contributes: not duration (§21), not code length (§22), not syntax
(§23), not strategy (§20). Builds the `CandidateAssessment`, joining candidate metadata.

**The join path (decision 1)** is fully determined by existing artifacts, no new plumbing:
the evaluation manifest carries `candidate_run_id` (and so does every `EvaluationResult`),
so the scorer resolves `config.paths.candidates / "runs" / <candidate_run_id>` and reads it
through the existing `CandidateRepository`. The join is keyed on `candidate_id`. A
candidate present in the evaluation run but absent from the generation run leaves the
secondary-metadata fields `None` and raises a validation issue — it never blocks scoring,
since the metadata is explicitly unused (§20, §57). Measured today: the join is exact and
total, 50/50.

Serialized floats are rounded to **6 decimal places** for §15's "consistent precision";
comparison and tie detection never use rounded floats — see the ranker.

**`comparator.py`** — `CandidateComparator`, `COMPARATOR_VERSION = "v1"`.

`compare(a: CandidateAssessment, b: CandidateAssessment) -> ComparisonResult` (§33, §66):

- either side `indeterminate` → `INDETERMINATE` (§33, §34)
- both `correct` → `TIE` (§33, §38) — the case that dominates this dataset
- equal pass rate → `TIE` (§33)
- otherwise higher `tests_passed` wins → `A_BETTER` / `B_BETTER` (§36, §37)

`score_margin = abs(score_a - score_b)` (§68), rounded to 6 dp on serialization.
`build_matrix(assessments)` returns all `N(N-1)/2` in-problem comparisons (§58).

**Naming guard.** Two different eligibility notions must not be conflated:
`RankingResult.eligible_for_preference` is *per candidate* (correct/incorrect → true,
indeterminate → false, §32), while `ComparisonResult.comparison_eligible` is *per pair*
(true only for `A_BETTER`/`B_BETTER` — a tie or an indeterminate pair yields no
preference, §35). Deliberately different field names so neither can be mistaken for the
other.

**`ranker.py`** — `CandidateRanker`, `RANKING_VERSION = "v1"`.

Groups strictly by `problem_id` (§7, §8) — candidates from different problems are never
compared. Within a problem:

1. Partition into ranked (`correct` + `incorrect`) and `indeterminate` (§39).
2. Sort ranked candidates by `(correctness_order, -tests_passed, candidate_id)` where
   `correct` precedes `incorrect` (§28). `candidate_id` is a **presentation-only** final
   ordering (§31) and never creates a preference.
3. Assign **competition ranking** — 1, 1, 3, 4, 5 — which is what §79's worked table shows.
   Ties share a rank; the next rank skips by the tie-group size (§30).
4. Every ranked candidate gets a `tie_group`; `tied = tie_group_size > 1` (§31 requires the
   flag be preserved explicitly).
5. Indeterminate candidates get a `RankingResult` with `rank = None`, `tie_group = None`,
   `eligible_for_preference = False`, listed last (§39) — recorded rather than dropped, so
   nothing vanishes silently (§71).

**Float equality is avoided entirely.** All *ranked* candidates within one problem share
the same test suite and therefore the same `tests_total`, so equality of pass rate is
decided by comparing the **integer** `tests_passed`. No float tolerance, no rounding, no
ordering that depends on IEEE representation — this is what makes §46's determinism
structural rather than incidental. (Verified against the data: `tests_total` is constant
within every one of the 10 problems — 7,7,7,8,8,7,8,9,7,6. The only way it could vary is an
indeterminate candidate with `tests_total == 0`, and those are partitioned out in step 1
before any comparison happens. The ranker asserts this invariant rather than assuming it.)

`tie_group` ids are scoped per problem as `<problem_id>_tg001` rather than §30's bare
`tg001`, because one `rankings.jsonl` holds all 10 problems and bare ids would collide.
Recorded as a deviation.

**`repository.py`** — `RankingRepository(directory)`, run-scoped, built on `atomic_io`
(`append_jsonl`/`iter_jsonl`) exactly as Stage 4/6's repositories are. The §53 API:
`save_assessment`, `save_ranking`, `save_comparison`, `get_assessment`, `get_ranking`,
`list_problem_rankings`, `list_all_rankings`, `count`, plus `ranked_problem_ids()` — the
resume index (§54). Never rewrites a record (§53).

**`run_repository.py`** — `RankingRunRepository(rankings_root)`, mirroring
`evaluation/run_repository.py` per decision 4: mints `rank_YYYYMMDD_HHMMSS_xxxx` ids
(§26), owns `manifest.json`/`statistics.json`, `get_run`/`list_runs`/`results`, and the
status lifecycle.

**`statistics.py`** — `RankingStatistics.from_records` lives with the model; this module
holds `format_ranking_statistics(stats) -> str` and the §49 per-problem table renderer.
Extracting a formatter is a mild deviation (Stage 4/6 format stats inline in the CLI); the
precedent is `problems/validation.py:format_report`, and it is justified here because
`rankings show` and `rankings list` render the same table.

**`validation.py`** — mirrors `runs/validation.py` exactly: `RankingValidationIssue(check,
message)`, `RankingValidationReport(ranking_run_id, issues, .valid)`,
`validate_ranking_run(run_dir, evaluation_run_dir, known_problem_ids=None) -> report`,
`format_ranking_report(report) -> str`. All issues fatal, no severity levels, every check
runs to completion accumulating into one list. Like `runs/validation.py` it reads the raw
JSONL directly rather than going through `RankingRepository`, so one call collects *every*
problem instead of raising on the first bad record.

**Two levels of checking, and §51/§52 demand the second.** Internal self-consistency
(`score == pass_rate`, `all_tests_passed` matching the counts) comes free from
`CandidateAssessment.__post_init__` — a tampered record simply fails to construct, exactly
as `Candidate`'s hash check already works in `runs/validation.py`. But §51 says "do not
trust the stored score blindly" and §52 says to verify "from the evaluation results", which
is a **cross-artifact** re-derivation: the validator loads the original evaluation run and
re-runs the classifier and scorer against it. That is the check that catches an assessment
which is internally coherent but no longer reflects the evidence it claims to summarise.
Hence the `evaluation_run_dir` argument.

Checks (§50–§52):

- every assessment's candidate belongs to the named evaluation run and a known problem
- exactly one assessment per candidate; no duplicates (§43); no candidate in the evaluation
  run missing an assessment (§70)
- `pass_rate` recomputed from `tests_passed / tests_total` **and** from the source
  `EvaluationResult`, and compared (§51)
- `score == pass_rate`
- `all_tests_passed` and `correctness` re-derived by re-running the classifier over the
  source evaluation results (§52)
- indeterminate ⇒ `eligible_for_preference == false` (§50)
- ranks are contiguous competition ranks per problem; tie groups internally consistent;
  candidates in one tie group have identical `tests_passed`
- **no tied pair appears with a winner** in `comparisons.jsonl` (§31, §35) — the check that
  directly protects DPO label quality
- `statistics.json` recomputed via `from_records(..., computed_at=<on-disk>)` and compared
  with `==`

---

## Persistence layout (§42, §80)

```
data/rankings/runs/rank_20260817_180500_a91c/
├── manifest.json        # versions, scoring configuration, evaluation_run_id, status
├── assessments.jsonl    # one CandidateAssessment per candidate
├── rankings.jsonl       # one RankingResult per candidate
├── comparisons.jsonl    # one ComparisonResult per in-problem pair (decision 2)
└── statistics.json      # reconstructable from assessments + rankings
```

Every record carries `ranking_run_id` and `evaluation_run_id`; assessments additionally
carry `candidate_run_id`, so the chain back to model and prompt stays traceable (§80).
Historical ranking runs are immutable (§45) — a re-ranking is a new run, never an
overwrite.

---

## Modifications to existing code

**`src/python_dpo/config.py` + `config.yaml` + `data/`** — the seventh data path.
`_REQUIRED_PATH_KEYS` gains `rankings`; `Paths` gains a `rankings: Path` field and includes
it in `ensure_exists()`; `config.yaml` gains `rankings: data/rankings`; `data/rankings/`
is created with a `.gitkeep`. A `ranking:` config section is **not** added — v1 has no
tunable scoring parameters, and `scoring_configuration` in the manifest records the
(currently empty) configuration for future use (§27).

**`src/python_dpo/cli.py`** — two new command groups (§47–§50):

| Command | Behavior |
|---|---|
| `rank run --evaluation-run-id ID [--problem-id P] [--limit N] [--resume RANK_ID] [--force]` | Rank an evaluation run (§47, §48, §54, §55) |
| `rankings list RANKING_RUN_ID` | Per-problem summary table |
| `rankings show RANKING_RUN_ID PROBLEM_ID` | The §49/§79 rank table for one problem |
| `rankings validate RANKING_RUN_ID` | The §50 validator; exit 1 on failure |
| `rankings stats RANKING_RUN_ID` | The §40 counters |

Resume/force follow the **spec literally** (§54, §55): an explicit `--resume RANKING_RUN_ID`
flag, matching Stage 4's `generate` rather than Stage 6's resume-by-default. `--force`
mints a new ranking run and never modifies an existing one (§55). Resume granularity is the
problem group (§54), skipping problems already present in `rankings.jsonl`.

**Honest note to carry into the report:** ranking 50 candidates is pure in-memory
computation taking milliseconds, so resume has no practical value on this dataset. It is
implemented because §54 and the §77 acceptance criteria require it, and because it is the
right shape for a future evaluation run large enough to matter — not because it earns its
keep today.

**`evaluations list` gains an optional `eval_id`.** §78's verification procedure runs
`python -m python_dpo evaluations list` with **no** argument to discover an evaluation run,
but Stage 6 made `eval_id` required and there is no command that lists evaluation *runs*.
With no argument it now lists evaluation runs (newest first, mirroring `runs list`); with
one it keeps today's behaviour. Small, and it makes the spec's own procedure executable.

**`_PLACEHOLDER_STAGES`** — unchanged. `rank`/`rankings` are new groups; `preferences` and
`run` stay placeholders.

**`src/python_dpo/__init__.py`** — `__version__` → `0.7.0`.

**Docs** — `src/python_dpo/ranking/README.md` (new), plus Stage 7 sections in `README.md`,
`src/python_dpo/README.md`, `data/README.md`, `tests/README.md`.

---

## Tests

**No Docker required — the whole suite is pure computation.** `tests/ranking/`:

- **`test_classifier.py`** (§72) — the full decision table: all pass; one fail; all fail;
  timeout → `incorrect` **not** indeterminate (§13); infrastructure error → `indeterminate`
  (§14); `tests_skipped > 0` → `incorrect` (§10); `tests_total == 0` → `indeterminate`;
  missing evaluation → `indeterminate` with a reason (decision 3); `tests_error > 0` →
  `incorrect` (the real p008 shape).
- **`test_scorer.py`** (§72) — 10/10, 5/10, 0/10, fractional rates; `score == pass_rate`;
  `all_tests_passed` distinct from `pass_rate` at 0.95 (§17); secondary metadata recorded
  but provably not affecting score (assert two assessments differing only in
  duration/code length/strategy get identical scores — §20, §21, §22, §23).
- **`test_comparator.py`** (§72, §74) — 10v8, 8v5, 10v10 → TIE, indeterminate → no winner,
  timeout; `score_margin` arithmetic (§68); the §74 matrix over A=10/10, B=8/10, C=10/10,
  D=5/10, E=0/10 asserting every listed relation including `A = C`.
- **`test_ranker.py`** (§72, §73) — the §73 integration shape (A,C rank 1 tied; B rank 3; D
  rank 4; E rank 5) with A and C in the same tie group and neither declared better;
  unique scores; all-correct (§62 — 5 correct, 0 preference relationships); all-incorrect
  with distinct rates preserving order (§63); mixed; all-indeterminate (§64 — zero
  preference-eligible); transitivity A>B, B>C ⇒ A>C (§59); competition-rank numbering;
  candidate_id ordering is presentation-only and does not set `tied=false` (§31).
- **`test_repository.py`** (§72) — persistence, retrieval, the §53 API, duplicate
  prevention, malformed-line rejection with a line number, the resume index.
- **`test_statistics.py`** — every §40 counter against hand-counted fixtures; the §41
  per-problem distribution.
- **`test_validation.py`** — one test per §50/§51/§52 check, each built by mutating a
  valid ranking run: corrupted `pass_rate`, corrupted `correctness`, an indeterminate
  candidate marked preference-eligible, a tied pair given a winner, a duplicate assessment,
  a missing assessment, drifted `statistics.json`.
- **`test_integration.py`** — §73's end-to-end flow; §75's **reproducibility test** (two
  ranking runs over identical input produce identical classifications, scores, ranks, tie
  groups and comparisons, differing only in run id and timestamps); §76's **versioning
  test** (changing `scoring_version` creates a new run and leaves the existing artifacts
  byte-identical).

**Fixtures must be synthetic for the indeterminate and timeout paths** — the real
evaluation run has zero of each, so those branches would otherwise ship untested.

**Extended** — `tests/test_project.py` gains the seventh data directory in both
enumerating tests, plus CLI parsing tests for `rank`/`rankings`.

---

## Execution order

1. Write this plan to `.claude/plans/07_candidate_ranking_plan.md` and add its entry to
   `.claude/plans/README.md`.
2. `errors.py`, `models.py` + tests — pure schema, no dependencies.
3. `classifier.py`, `scorer.py` + tests — the decision table and the score.
4. `comparator.py` + tests, including the §74 matrix.
5. `ranker.py` + tests, including §73, competition ranking and transitivity.
6. `repository.py`, `run_repository.py` + tests.
7. `statistics.py`, `validation.py` + tests.
8. Config path wiring (`config.py`, `config.yaml`, `data/rankings/.gitkeep`, both
   `test_project.py` enumerations) and CLI wiring.
9. `test_integration.py` — reproducibility and versioning last, since they exercise
   everything.
10. Rank the real evaluation run; commit `data/rankings/runs/<rank_id>/`; docs; the §83
    report.

---

## Verification

```bash
source .venv/bin/activate
pytest -q                                  # offline, zero skips, no Docker

python -m python_dpo evaluations list      # now works with no argument
python -m python_dpo rank run --evaluation-run-id eval_20260817_115154_dcd4

python -m python_dpo rankings list  RANK_ID
python -m python_dpo rankings show  RANK_ID p004     # the interesting one: 3 correct, 2 partial
python -m python_dpo rankings show  RANK_ID p001     # the common one: all 5 tied
python -m python_dpo rankings stats RANK_ID
python -m python_dpo rankings validate RANK_ID       # must exit 0

# determinism (§46, §75): a second run differs only in run id and timestamps
python -m python_dpo rank run --evaluation-run-id eval_20260817_115154_dcd4 --force
diff <(jq -S 'del(.ranking_run_id,.created_at)' RUN_A/rankings.jsonl) \
     <(jq -S 'del(.ranking_run_id,.created_at)' RUN_B/rankings.jsonl)   # empty

# nothing upstream was mutated
git diff --stat data/problems/ data/candidates/ data/evaluations/       # empty
```

**Expected output on the real run — computed from the committed data, so these are exact
acceptance numbers, not estimates:**

```
problems 10 · candidates 50 · correct 30 · incorrect 20 · indeterminate 0
fully_correct 30 · partially_correct 20 · zero_test_pass 0
tie_groups 14 · tied_candidates 49 · preference_eligible_candidates 50
pairs 100 · non-tied 22 · tied 78

p001 1:5@7/7                      p006 1:5@7/7
p002 1:5@3/7                      p007 1:2@7/8  3:3@6/8
p003 1:5@7/7                      p008 1:2@9/9  3:3@8/9
p004 1:3@8/8  4:2@6/8             p009 1:5@7/7
p005 1:5@8/8                      p010 1:4@4/6  5:1@1/6
```

Six problems collapse to a single tie group and contribute no ordering at all. That is the
correct result under §62 and §35 — not a bug to be engineered away.

Scope containment:

```bash
grep -rniE "\b(chosen|rejected|dpo|preference_pair)\b" src/python_dpo/ranking/  # only prose disclaimers
grep -rniE "\b(qwen|openai|anthropic|llm|judge)\b"      src/python_dpo/ranking/  # none (§56)
grep -rn  "random\|shuffle\|time.time\|datetime.now"    src/python_dpo/ranking/  # only created_at stamping (§46)
```

Then produce the §83 report in `07_CANDIDATE_RANKING.md` and **stop — do not start Step 8
(DPO preference-pair generation) without explicit approval** (§83).

---

## Deviations to record in the report

- **`tie_group` ids are problem-scoped** (`p001_tg001`) rather than §30's bare `tg001`,
  since one `rankings.jsonl` holds all problems.
- **Indeterminate candidates get a persisted `RankingResult` with `rank = null`** rather
  than being dropped — §39 permits either "excluded from ranking" or "a separate
  indeterminate group"; recording them satisfies §71's no-silent-data-loss rule.
- **Competition ranking (1, 1, 3, 4, 5)** chosen from §30's "or another documented
  convention", because §79's worked example uses exactly that.
- **Tie detection compares integer `tests_passed`, never floats**, since `tests_total` is
  constant within a problem. Makes §46 determinism structural.
- **Serialized floats rounded to 6 dp** for §15's "consistent precision"; comparisons never
  use the rounded value.
- **`comparison_eligible` on `ComparisonResult` is deliberately *not* named
  `eligible_for_preference`**, to keep the pair-level and candidate-level notions (§32 vs
  §67) from being conflated.
- **`CandidateAssessment` carries candidate-derived metadata**, so ranking reads the
  generation run as well as the evaluation run — a wider input surface than §7's literal
  "the ranking stage receives EvaluationResult" (user decision 1).
- **`comparisons.jsonl` is persisted** (§42/§58 make it optional) (user decision 2).
- **Resume is implemented but has no practical value at this data scale** — required by
  §54 and the §77 acceptance criteria; stated plainly rather than oversold.
- **A third copy of the run-directory plumbing** rather than a shared base extracted across
  Stages 4, 6 and 7 (user decision 4) — the rule-of-three moment, deliberately deferred to
  keep Stage 7's blast radius contained.
- **`evaluations list` made argument-optional**, so §78's own verification procedure runs.
- **No `ranking:` section in `config.yaml`** — v1 has no tunable scoring parameters;
  `scoring_configuration` in the manifest holds the (empty) configuration for future use.
- **A `format_ranking_statistics` helper exists**, where Stages 4 and 6 format statistics
  inline in the CLI, because `rankings list`/`show` share the renderer.
