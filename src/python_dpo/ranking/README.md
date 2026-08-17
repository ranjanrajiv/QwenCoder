# src/python_dpo/ranking/

Candidate evaluation, scoring, and ranking — turns Stage 6's objective execution evidence
(`EvaluationResult`) into a correctness classification, a score, a deterministic
per-problem ranking with explicit tie groups, and pairwise comparisons.

This package answers **"was this candidate good, and how does it compare to its
siblings?"** — never **"which pair should DPO train on?"**. It produces no
`chosen`/`rejected` labels, no DPO JSONL, and calls no LLM judge. The ranking signal is
test-case performance alone; a tie stays a tie unless the evidence itself breaks it.

Candidates are ranked strictly **within their own problem** — this package never compares
a candidate from one problem against a candidate from another. Given identical evaluation
evidence and configuration, the output is byte-for-byte deterministic: no randomness, no
LLM calls, no timestamp-influenced ordering.

## Files

### `errors.py`

`RankingError` and its subclasses: `RankingConfigError`, `EvaluationRunNotFoundError`
(the Stage 6 evaluation run a ranking is requested against does not exist — distinct from
`run_repository.py`'s `RankingRunNotFoundError`, which is about this package's *own* run
directories). Ranking is pure computation over already-persisted evidence — it never
calls a model, never touches Docker, never executes candidate code, so there is no
machinery failure mode beyond a missing or malformed input artifact.

### `models.py`

The schema, mirroring `python_dpo.evaluation.models`: frozen dataclasses validating in
`__post_init__`, explicit `to_dict()`/`from_dict()` rejecting unknown/missing fields.

`CandidateAssessment` — one candidate's classification and score, plus secondary
candidate metadata (`code_sha256`, `duplicate_of`, `code_lines`, `code_chars`,
`strategy`, `syntax_valid`) recorded for traceability only, never scored.
`correctness == "indeterminate"` if and only if `tests_total == 0` — every indeterminate
path (no evaluation attempted, an infrastructure failure, or a problem with zero declared
tests) collapses to no usable test evidence, and `__post_init__` validates this rather
than assuming it.

`RankingResult` — one candidate's position within its problem's ranking. `rank`/
`tie_group` are `None` exactly for indeterminate candidates: recorded, never dropped, but
excluded from the ordering. `eligible_for_preference` is the *per-candidate* signal
(correct/incorrect → true, indeterminate → false) — a deliberately different notion from
`ComparisonResult.comparison_eligible`, which is *per-pair*, so the two can never be
conflated by name alone.

`ComparisonResult` — one pairwise comparison. `relation` is
`A_BETTER`/`B_BETTER`/`TIE`/`INDETERMINATE`, never `chosen`/`rejected`.
`comparison_eligible` is true only for a decisive relation — a tie or an indeterminate
pair never produces a preference.

`RankingManifest` and `RankingStatistics` mirror `EvaluationManifest`/
`EvaluationStatistics`: `with_status()` enforces the same closed transition graph shape,
and `RankingStatistics.from_records(...)` is always reconstructable from the persisted
JSONL files, never trusted from an in-memory counter.

### `classifier.py`

`CorrectnessClassifier.classify(result)` — the decision table, in precedence order:

| Condition | Result |
|---|---|
| No `EvaluationResult` at all | `indeterminate` |
| Infrastructure failure | `indeterminate` |
| `tests_total == 0` | `indeterminate` |
| Every test passed, no timeout | `correct` |
| Otherwise (failure, error, skip, or a candidate-caused timeout) | `incorrect` |

The ordering is what makes a candidate-caused timeout `incorrect`, never
`indeterminate` — only a genuine absence of usable test evidence is indeterminate.
`classify_missing(error_type)` handles the "no result at all" case, using an
`EvaluationFailure`'s own `error_type` as the specific reason.

### `scorer.py`

`CandidateScorer.score(...)` builds a `CandidateAssessment`. The score is deliberately
simple: `score == pass_rate == tests_passed / tests_total`, full stop. Duration, code
length, syntax validity, and generation strategy are joined in from the candidate's own
record (via the generation run's `CandidateRepository`, resolved through
`candidate_run_id`) purely as secondary metadata — every scorer test that varies one of
these fields while holding test results constant asserts the score is unchanged.

### `comparator.py`

`CandidateComparator.compare(a, b)` never invents a preference: either side
indeterminate, or an equal score, is never decided in favour of one candidate.
`build_matrix(assessments)` returns every `N(N-1)/2` in-problem comparison, ordered
deterministically by `candidate_id` — never a set/dict iteration order.

### `ranker.py`

`CandidateRanker` groups strictly by `problem_id`. Within a problem, ranked candidates
(`correct` before `incorrect`, then by `tests_passed` descending) receive a **competition
rank** — 1, 1, 3, 4, 5 — with `candidate_id` breaking ties only for *presentation*, never
turning a tie into a decision. Tie detection compares the **integer** `tests_passed`,
never a float, because every ranked candidate within one problem shares the same
`tests_total` — asserted, not assumed, so a mismatch raises rather than silently comparing
apples to oranges. `indeterminate` candidates are appended last with `rank = None`.

### `repository.py` / `run_repository.py`

`RankingRepository` — run-scoped persistence for `assessments.jsonl`, `rankings.jsonl`,
`comparisons.jsonl`, built on `atomic_io` exactly as the Stage 4/6 repositories are.
`ranked_problem_ids()` is the resume index: a problem is settled once any of its
candidates has a persisted ranking.

`RankingRunRepository` — the multi-run manager, mirroring
`evaluation.run_repository.EvaluationRunRepository`: mints `rank_YYYYMMDD_HHMMSS_xxxx`
ids, owns `manifest.json`/`statistics.json`, and the run status lifecycle. A third,
independent copy of this plumbing rather than a shared base extracted across Stages 4, 6,
and 7 — a deliberate deferral to keep this stage's blast radius contained; see the
implementation report for the reasoning.

### `statistics.py`

Text formatters only — `RankingStatistics` itself lives in `models.py`.
`format_ranking_statistics(stats)` renders the counters and per-problem distribution;
`format_ranking_table(rows)` renders the rank table `rankings show` displays, taking
`(RankingResult, CandidateAssessment)` pairs already joined by the caller (the two live
in separate files).

### `validation.py`

Mirrors `python_dpo.runs.validation`: `RankingValidationIssue(check, message)`,
`RankingValidationReport`, every issue fatal, every check runs to completion. Internal
self-consistency comes free from the dataclasses' own `__post_init__`; the module's real
job is the **cross-artifact** recheck the spec asks for explicitly — "do not trust the
stored score blindly" — which reloads the source Stage 6 evaluation run and re-runs the
classifier against it, catching an assessment that is internally coherent but no longer
reflects the evidence it claims to summarise. Also checks: exactly one assessment per
candidate, every evaluated candidate has an assessment, ranks are contiguous competition
ranks with internally consistent tie groups, and — the check that most directly protects
DPO label quality — **no tied pair appears in `comparisons.jsonl` with a winner**.

## Persistence layout

```
data/rankings/runs/rank_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json        # evaluation run covered, algorithm versions, status
├── assessments.jsonl    # one CandidateAssessment per candidate
├── rankings.jsonl       # one RankingResult per candidate
├── comparisons.jsonl    # one ComparisonResult per in-problem pair
└── statistics.json      # reconstructable from assessments.jsonl + rankings.jsonl
```

Every record carries `ranking_run_id` and `evaluation_run_id`; assessments additionally
carry `candidate_run_id`, so the full chain back to model and prompt stays traceable.
Historical ranking runs are immutable — a re-ranking with `--force`, or a run at a new
`scoring_version`, is a new run, never an overwrite.
