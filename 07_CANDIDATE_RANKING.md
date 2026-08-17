# Stage 7 Implementation Details — Candidate Evaluation, Scoring and Ranking

How `src/python_dpo/ranking/` implements the layer specified in
`.claude/specs/07_candidate_ranking.md`. For usage, see the "Stage 7 — Candidate
Ranking" section of the root `README.md`. This file is about *how* it is built and what
was learned building it.

## Goal

Stage 6 produced objective execution evidence — pass/fail counts, statuses, durations —
and stopped there, deliberately: nothing in the repository said whether a candidate was
*good*. Stage 7 closes that gap: classify each candidate `correct`/`incorrect`/
`indeterminate`, score it, and rank candidates deterministically within each problem,
exposing pairwise comparisons Step 8 will need to build DPO preference pairs — without
ever building a pair itself.

Two constraints define the stage:

- **Objectivity.** The ranking signal is test-case performance, full stop. No LLM judge,
  no random ordering, no timestamp-influenced ordering, no heuristic overriding execution
  results. Given identical evaluation evidence and configuration, the output is
  byte-for-byte deterministic.
- **Scope.** This stage produces assessments, rankings, and pairwise comparisons. It
  produces no `chosen`/`rejected` labels, no DPO JSONL, no preference pairs, and calls no
  LLM. Neutral terminology throughout (`A_BETTER`/`B_BETTER`/`TIE`/`INDETERMINATE`).

## 1. Correctness-classification rules

`CorrectnessClassifier.classify(result)`, in precedence order:

| Condition | Result |
|---|---|
| No `EvaluationResult` at all (an `EvaluationFailure`, or absent) | `indeterminate` |
| `infrastructure_error` / `status == "infrastructure_error"` | `indeterminate` |
| `tests_total == 0` | `indeterminate` |
| `tests_passed == tests_total`, no failures/errors/skips, no timeout | `correct` |
| Otherwise (a failure, a candidate exception, a skipped test, or a candidate-caused timeout) | `incorrect` |

The ordering is the whole point: a candidate-caused timeout reaches the final row and is
`incorrect`, never `indeterminate` — only a genuine absence of usable test evidence
(nothing ran, Docker died, or the problem had zero declared tests) is indeterminate. A
skipped test also falls through to `incorrect`, since `correct` requires
`tests_skipped == 0`. `classify_missing(error_type)` handles the "no result at all" case,
using an `EvaluationFailure`'s own `error_type` as the specific indeterminate reason
rather than a generic one.

`CandidateAssessment.__post_init__` validates the structural consequence of this table
rather than assuming it: `correctness == "indeterminate"` if and only if
`tests_total == 0`. Every path that produces indeterminate collapses to zero usable test
evidence, so this is checked at construction, not merely by convention.

## 2. Scoring formula

`score = pass_rate = tests_passed / tests_total`. Nothing else contributes. Duration,
code length, generation strategy, and syntax validity are recorded as secondary metadata
(joined from the generation run via `candidate_run_id`) purely for traceability —
`test_scorer.py` proves this directly by holding test results constant and varying only
each secondary field, asserting the score never moves. `all_tests_passed` is a separate
binary signal from `pass_rate` (a candidate at `pass_rate = 0.95` is never conflated with
one that passed everything).

## 3. Ranking algorithm

`CandidateRanker` groups strictly by `problem_id` — this module never compares
candidates from different problems, enforced by `rank_problem` raising if it is handed a
mismatched `problem_id`. Within a problem:

1. Partition into ranked (`correct` + `incorrect`) and `indeterminate`.
2. Sort ranked candidates by `(correct before incorrect, tests_passed descending,
   candidate_id)` — `candidate_id` is a presentation-only final tiebreaker (§4 below).
3. Assign **competition ranking**: 1, 1, 3, 4, 5. Ties share a rank; the next rank skips
   by the tie group's size.
4. Indeterminate candidates are appended last with `rank = None`, recorded rather than
   dropped.

**Tie detection compares the integer `tests_passed`, never a float.** Every ranked
candidate within one problem shares the same `tests_total` — the ranker asserts this
(raising `ValueError` on a mismatch) rather than assuming it, since a problem's declared
test suite must be constant across every candidate evaluated against it. This is what
makes the determinism requirement structural rather than incidental: there is no float
tolerance anywhere in the comparison path, so there is nothing for floating-point
representation to perturb.

## 4. Tie-handling policy

Two fully-correct candidates, or two candidates with an identical pass rate, are tied —
never arbitrarily ordered into a preference. `tie_group` records which candidates share a
bucket; `tied = tie_group_size > 1`. `candidate_id` breaks ties only for *display*
ordering within a tie group and is applied strictly after the tie group is formed, so it
can never turn a tie into a decision — `test_ranker.py` pins this directly: two
candidates named `"zzz"` and `"aaa"` at an identical score are both `tied = True` and
share one `tie_group`, regardless of which sorts first for presentation.

This is not an edge case in practice. Measured against the real Stage 6 evaluation run:
**49 of 50 candidates sit in a tie group larger than one**, and **6 of the 10 problems
collapse to a single tie group**, contributing zero ordering at all. Tie handling is the
majority behavior this stage exists to get right.

## 5. Pairwise comparison policy

`CandidateComparator.compare(a, b)`:

- either side `indeterminate` → `INDETERMINATE`
- equal score (within a `1e-9` epsilon, the same tolerance `EvaluationResult` already
  uses for its own `pass_rate` check) → `TIE`
- otherwise, the higher score → `A_BETTER` / `B_BETTER`

`score_margin = abs(score_a - score_b)`. `comparison_eligible` is true only for a
decisive relation — a tie or an indeterminate pair produces no preference.
`build_matrix(assessments)` returns every `N(N-1)/2` in-problem comparison in
deterministic `candidate_id` order, never a set/dict iteration order.

`comparison_eligible` (per-*pair*) is deliberately named differently from
`RankingResult.eligible_for_preference` (per-*candidate*, correct/incorrect → true,
indeterminate → false) — the two notions are easy to conflate by shape, so they are kept
apart by name as well as by field.

Verified against the spec's own worked example (`A=10/10, B=8/10, C=10/10, D=5/10,
E=0/10`): every one of the ten listed relations, including `A = C`, reproduces exactly.

## 6. Indeterminate handling

An indeterminate candidate never blocks a run and is never silently dropped: it gets a
persisted `RankingResult` with `rank = null`, `tie_group = null`,
`eligible_for_preference = false`, listed after every ranked candidate. A problem where
every candidate is indeterminate produces zero preference-eligible results — recorded as
a legitimate outcome, not an error.

The real evaluation run has **zero** indeterminate candidates (zero timeouts, zero
syntax errors, zero infrastructure errors), so this path has **no real-world coverage**
in this run's committed artifact. It is exercised entirely by synthetic fixtures in
`test_classifier.py`, `test_ranker.py`, and `test_comparator.py`.

## 7. Ranking schema

`CandidateAssessment` — provenance ids, `correctness`, `all_tests_passed`, `pass_rate`,
`score`, the four raw test counts plus `tests_total`, `timeout`, `infrastructure_error`,
`execution_duration_ms`, `indeterminate_reason`, and the secondary metadata
(`code_sha256`, `duplicate_of`, `code_lines`, `code_chars`, `strategy`, `syntax_valid`).

`RankingResult` — `rank`, `score`, `correctness`, `pass_rate`, `all_tests_passed`,
`tie_group`, `tie_group_size`, `tied`, `eligible_for_preference`.

`ComparisonResult` — `candidate_a`/`candidate_b`, `relation`, `score_a`/`score_b`,
`score_margin`, `correctness_a`/`correctness_b`, `comparison_eligible`.

Every dataclass validates its own invariants in `__post_init__` — a `passed`-implies-
`all_tests_passed`-implies-`score`-implies-`pass_rate` chain that cannot be constructed
inconsistently. This is the first line of defense the validator (§13 below) builds on.

## 8. Ranking-run architecture

```
EvaluationResult (+ candidate metadata)
        │
        ▼
CorrectnessClassifier -> CandidateScorer -> CandidateAssessment
        │
        ▼
CandidateRanker -> RankingResult
        │
        ▼
CandidateComparator -> ComparisonResult
        │
        ▼
RankingRepository -> assessments.jsonl / rankings.jsonl / comparisons.jsonl
```

| Module | Responsibility |
|---|---|
| `errors.py` | The exception hierarchy — machinery failures only, never a candidate verdict |
| `models.py` | `CandidateAssessment`, `RankingResult`, `ComparisonResult`, `RankingManifest`, `RankingStatistics` |
| `classifier.py` | `CorrectnessClassifier` — the §1 decision table |
| `scorer.py` | `CandidateScorer` — builds an assessment, joining candidate metadata |
| `comparator.py` | `CandidateComparator` — pairwise comparison and the full matrix |
| `ranker.py` | `CandidateRanker` — per-problem competition ranking |
| `repository.py` | `RankingRepository` — one ranking run's persisted records |
| `run_repository.py` | `RankingRunRepository` — multi-run manager, mints `rank_...` ids |
| `statistics.py` | Text formatters (`RankingStatistics` itself lives in `models.py`) |
| `validation.py` | `validate_ranking_run` — cross-artifact integrity checking |

## 9. Versioning strategy

`RankingManifest` records three independent version strings: `ranking_version` (the
ranker's ordering/tie-handling algorithm), `scoring_version` (classification + scoring),
`comparator_version` (pairwise comparison). All three are `"v1"` today. Changing any one
and re-running mints a new `ranking_run_id`; the original run's files are never rewritten
— proven directly by `test_integration.py`'s versioning test, which changes
`scoring_version` mid-suite and asserts the original run's `rankings.jsonl` is
byte-for-byte unchanged afterward.

## 10. CLI commands added

| Command | Behavior |
|---|---|
| `rank run --evaluation-run-id ID [--problem-id] [--limit] [--resume] [--force]` | Rank an evaluation run |
| `rankings list RANKING_RUN_ID` | Per-problem summary (candidates, correct/incorrect/indeterminate, tie groups) |
| `rankings show RANKING_RUN_ID PROBLEM_ID` | The rank table for one problem |
| `rankings stats RANKING_RUN_ID` | The statistics counters |
| `rankings validate RANKING_RUN_ID` | The integrity validator; exit 1 on failure |

`evaluations list` was also made argument-optional: with no `eval_id` it now lists
evaluation runs themselves (mirroring `runs list`), which is what the spec's own §78
verification procedure assumes exists.

`rank run`'s resume semantics follow the spec **literally**, not `evaluate`'s
resume-by-default: a bare invocation always mints a new ranking run (matching
`generate`), and `--resume RANKING_RUN_ID` is the only way to continue one. Any selection
flag combined with `--resume` is rejected — the manifest is authoritative, the same rule
`generate --resume` already enforces. `--force` abandons a `--resume` request and starts
fresh; without `--resume` it has no effect, since a bare invocation already always starts
fresh (identical to `generate --force`'s documented behavior).

## 11. Ranking statistics

`RankingStatistics.from_records(manifest, assessments, rankings)` — always
reconstructable from the two JSONL files, never trusted from an in-memory counter. Counts
`problems`, `candidates`, `correct`/`incorrect`/`indeterminate`, `fully_correct` (always
equal to `correct` — validated in `__post_init__`), `partially_correct`/`zero_test_pass`
(the two ways an `incorrect` candidate can fail: some tests passing vs. none), `tied_
candidates`, `preference_eligible_candidates`, plus a `per_problem` distribution.
`format_ranking_statistics`/`format_ranking_table` render both as text for the CLI.

## 12. Unit-test results

```
tests/ranking/test_classifier.py      13 passed
tests/ranking/test_scorer.py          13 passed
tests/ranking/test_comparator.py      11 passed
tests/ranking/test_ranker.py          15 passed
tests/ranking/test_repository.py      15 passed
tests/ranking/test_run_repository.py  14 passed
tests/ranking/test_statistics.py       6 passed
tests/ranking/test_validation.py      10 passed
```

97 unit tests, 0 Docker, 0 model calls — the whole suite is pure computation.

## 13. Integration-test results

```
tests/ranking/test_integration.py      3 passed
```

The end-to-end flow (spec §73's `A=10/10, B=8/10, C=10/10, D=5/10, E=0/10` fixture,
verified through `validate_ranking_run` as well as direct rank assertions), the
reproducibility test, and the versioning test. All pure computation — no `-m integration`
marker needed, unlike Stages 5 and 6, since nothing here touches Docker.

## 14. Determinism-test results

`test_reproducibility_two_runs_over_identical_input_agree` runs the full
score → rank → compare pipeline twice against byte-identical evaluation evidence into two
separate ranking runs, then strips only `ranking_run_id` and `created_at` from every
record before comparing: assessments, rankings, and comparisons are asserted equal across
the two runs. Passed on every run of the suite, including as part of `pytest -q`'s 743
green, zero-skip result.

## 15. Example ranking output

The real Stage 6 evaluation run, `eval_20260817_115154_dcd4` (50 candidates, 10 problems
× 5), ranked as `rank_20260817_161726_a84d`:

```
$ python -m python_dpo rankings show rank_20260817_161726_a84d p004
RANK  CANDIDATE     TESTS     SCORE   STATUS
1     p004_c001     8/8       1.00    correct
1     p004_c002     8/8       1.00    correct
1     p004_c003     8/8       1.00    correct
4     p004_c004     6/8       0.75    incorrect
4     p004_c005     6/8       0.75    incorrect

$ python -m python_dpo rankings stats rank_20260817_161726_a84d
Problems: 10          Candidates: 50
Correct: 30           Incorrect: 20          Indeterminate: 0
Fully correct: 30     Partially correct: 20  Zero test pass: 0
Tied candidates: 49   Preference eligible candidates: 50
```

Per-problem: p001/p003/p005/p006/p009 are five-way ties at full correctness; p002 is a
five-way tie at partial correctness (all candidates failed identically); p004/p007/p008/
p010 are the four problems with an actual ordering. `rankings validate
rank_20260817_161726_a84d` passes clean — every cross-artifact recomputation against the
source evaluation run agrees with what is stored.

## 16. Files created/modified

**Created:**

- `src/python_dpo/ranking/` — `__init__.py`, `errors.py`, `models.py`, `classifier.py`,
  `scorer.py`, `comparator.py`, `ranker.py`, `repository.py`, `run_repository.py`,
  `statistics.py`, `validation.py`, `README.md`
- `tests/ranking/` — `__init__.py`, `test_classifier.py`, `test_scorer.py`,
  `test_comparator.py`, `test_ranker.py`, `test_repository.py`,
  `test_run_repository.py`, `test_statistics.py`, `test_validation.py`,
  `test_integration.py`
- `data/rankings/runs/rank_20260817_161726_a84d/` — the real ranking run
- `data/rankings/.gitkeep`
- `07_CANDIDATE_RANKING.md` (this file)

**Modified:**

- `src/python_dpo/config.py` — the seventh data path (`rankings`), threaded through
  `_REQUIRED_PATH_KEYS` and the `Paths` dataclass
- `config.yaml` — the `rankings: data/rankings` path entry (no `ranking:` settings
  section — see §18)
- `src/python_dpo/cli.py` — the `rank`/`rankings` command groups; `evaluations list`
  made argument-optional
- `tests/test_project.py` — the seventh data directory in both enumerating tests
  (`test_config_loads_real_config_yaml`, `test_paths_ensure_exists_creates_all_
  directories`, `test_real_data_directories_exist`); CLI parsing/error-path tests for
  `rank`/`rankings`; the `evaluations list` no-argument test
- `tests/sandbox/test_config.py`, `tests/evaluation/test_config.py` — their minimal
  fixture YAML gained the required `paths.rankings` key
- `src/python_dpo/__init__.py` — version `0.6.0` → `0.7.0`
- `README.md`, `src/python_dpo/README.md`, `data/README.md`, `tests/README.md` — Stage 7
  documentation

## 17. Dependencies added

**None.** Ranking is pure Python standard library computation over already-persisted
JSONL — no new runtime or dev dependency anywhere in the stack.

## 18. Deviations from the specification

- **`tie_group` ids are problem-scoped** (`p001_tg001`) rather than the spec's bare
  `tg001` example, since one `rankings.jsonl` holds every problem in a run and bare ids
  would collide across problems.
- **Indeterminate candidates get a persisted `RankingResult` with `rank = null`** rather
  than being excluded outright — the spec permits either "excluded from ranking" or "a
  separate indeterminate group"; recording them satisfies the no-silent-data-loss rule.
- **Competition ranking (1, 1, 3, 4, 5)**, chosen from the spec's "or another documented
  ranking convention" because the spec's own worked example (§79) uses exactly that
  shape.
- **`CandidateAssessment` reads the generation run as well as the evaluation run** (user
  decision, confirmed during planning): the assessment carries `code_sha256`/
  `duplicate_of`/code length/strategy joined from the candidate's own record. The
  decisive reason, measured against the real data: 31 of 50 candidates in the real run
  have non-null `duplicate_of` (only 19 unique `code_sha256`), so without this join,
  Step 8 would have no way to tell that the 22 non-tied pairs this run produces collapse
  to roughly 7 genuinely distinct preference relationships. This is a wider input surface
  than the spec's literal "the ranking stage receives EvaluationResult," but the metadata
  is recorded, never scored — score is still `pass_rate` alone.
- **`comparisons.jsonl` is persisted** (the spec calls this optional) — confirmed with
  the user during planning, since it hands Step 8 `score_margin` already computed and
  makes the spec's own §74 pairwise expectations auditable on disk rather than only
  reproducible in memory.
- **`rank run`'s resume follows the spec literally (explicit `--resume`), not
  `evaluate run`'s resume-by-default** — a deliberate asymmetry with Stage 6, recorded in
  the CLI's own docstring: ranking is pure in-memory computation with no GPU or Docker
  cost to amortize, so there is nothing resume-by-default would actually save here.
- **A third, independent copy of the run-directory-management plumbing**
  (`run_repository.py`) rather than a shared base extracted across Stages 4, 6, and 7 —
  confirmed with the user during planning as a deliberate deferral to keep this stage's
  blast radius contained, rather than refactoring two stages of working, tested code.
- **No `ranking:` section in `config.yaml`** — v1 has no tunable scoring parameters;
  `scoring_configuration` on `RankingManifest` holds the (currently empty) configuration
  for future use, matching the spec's own manifest field list.
- **`evaluations list` made argument-optional** — a small, backward-compatible fix so the
  spec's own §78 verification procedure (`python -m python_dpo evaluations list` with no
  argument, to discover an evaluation run id) actually runs; Stage 6 had made `eval_id`
  required.
- **A `format_ranking_statistics`/`format_ranking_table` module exists** (`statistics.py`)
  where Stages 4 and 6 format statistics inline in the CLI, because `rankings list` and
  `rankings show` both render variants of the same table shape.

## 19. Known limitations

- **The indeterminate path has zero coverage from real data.** The committed evaluation
  run has no timeouts, syntax errors, or infrastructure errors, so every indeterminate
  branch — in the classifier, the ranker, and the comparator — is exercised only by
  synthetic fixtures. The logic is verified sound, but not yet proven against a real
  Docker failure feeding into ranking.
- **Score margin equality uses a fixed `1e-9` epsilon**, matching `EvaluationResult`'s own
  precedent, rather than a configurable tolerance. Adequate for the small integer test
  counts this dataset produces (`tests_total` between 6 and 9); would need revisiting if
  a future problem's test count grew large enough for floating-point drift to matter.
- **Resume is implemented but has no practical value at this data scale** — ranking 50
  candidates across 10 problems completes in roughly 8 seconds. It exists because the
  spec requires it and because it is the right shape for a future evaluation run large
  enough to matter, not because today's dataset benefits from it.
- **The cross-artifact validator (`_check_against_evaluation_run`) requires the original
  evaluation run's directory to still exist on disk.** If it has been moved or deleted,
  `rankings validate` still runs every other check but skips the §51/§52 recomputation,
  surfacing a clear issue rather than failing outright — but the strongest guarantee this
  stage offers (that a stored score has not silently drifted from its source evidence)
  is unavailable in that case.

Stopping here. Not starting Step 8 (DPO preference-pair generation) without explicit
approval.
