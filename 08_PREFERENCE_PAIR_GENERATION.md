# Stage 8 Implementation Details — DPO Preference Pair Generation

How `src/python_dpo/preferences/` implements the layer specified in
`.claude/specs/08_preference_pair_generation.md`. For usage, see the "Stage 8 — DPO
Preference Pair Generation" section of the root `README.md`. This file is about *how* it
is built and what was learned building it.

## Goal

Stage 7 produced a neutral ordering — `A_BETTER`/`B_BETTER`/`TIE`/`INDETERMINATE` — and
deliberately stopped short of a preference; its own spec reserved `chosen`/`rejected`
vocabulary for this stage. Stage 8 closes that gap: convert the ranking into
`{prompt, chosen, rejected}` DPO training records, backed entirely by execution evidence,
with full audit provenance, three selection policies, deduplication, problem-level
splitting, statistics, a quality report, and dataset validation.

Three constraints define the stage:

- **Model-free.** No Qwen, no Claude, no embeddings, no reward model, no LLM judge. Every
  label comes from Stage 6's pytest counts by way of Stage 7's `CandidateComparator`,
  re-run here rather than trusted from a persisted `ComparisonResult`.
- **Code is never touched.** `chosen`/`rejected` are the candidates' code verbatim, as
  persisted by Stage 3. No formatter, no repair, no fences, no explanation injected, no
  evaluation label leaked into the text.
- **No ties, ever.** Equal score is not a preference — this governs the majority of the
  real input, since 78 of the real ranking run's 100 candidate pairs are ties.

## 1. Preference data schema

`PreferencePair` — `preference_id`, `problem_id`, `candidate_run_id`, `ranking_run_id`,
`evaluation_run_id`, `chosen_candidate_id`, `rejected_candidate_id`, `prompt`, `chosen`,
`rejected`, `chosen_score`/`rejected_score`/`score_margin`,
`chosen_pass_rate`/`rejected_pass_rate`, `chosen_tests_passed`/`rejected_tests_passed`/
`chosen_tests_total`/`rejected_tests_total`, `chosen_correctness`/`rejected_correctness`,
`preference_strength`, `selection_policy`/`selection_policy_version`,
`preference_version`, `canonical_prompt_sha256`/`prompt_version`,
`chosen_generation_prompt_sha256`/`rejected_generation_prompt_sha256`,
`chosen_strategy`/`rejected_strategy`, `chosen_code_sha256`/`rejected_code_sha256`,
`duplicate_training_record`/`canonical_preference_id`, `created_at`. `__post_init__`
enforces pair validity (different candidates, different code, `chosen_score >
rejected_score`, `score_margin` arithmetic, `preference_strength` agreeing with the
correctness pair) as a construction-time invariant — an invalid pair cannot exist in
memory, only be caught after the fact.

`PreferenceRejection` — every candidate pair that did not become a preference, with a
`reason` from a closed set: `tie`, `indeterminate`, `identical_code`,
`insufficient_margin`, `not_correct_vs_incorrect`, `invalid_prompt_match`,
`integrity_failure`, `max_pairs_per_problem`.

`PreferenceManifest`/`PreferenceStatistics` mirror `RankingManifest`/`RankingStatistics`.
`QualityReport` holds score-margin/pass-rate/strategy distributions and a reason per
pairless problem — reported, never enforced.

## 2. Canonical prompt and lineage verification

Every candidate of a problem was generated under a *different*, strategy-specific prompt
(Stage 3's `Strategy:` block), so no two candidates share a raw `prompt_sha256`. Applied
literally, the spec's "chosen and rejected prompts must match" check would produce zero
pairs under every policy — confirmed by measurement against the real committed candidate
run: all 50 candidates have distinct `prompt_sha256`.

The resolution: `build_canonical_prompt(problem)` (added to
`generation/prompt_builder.py`, derived from the existing `_TEMPLATE` by removing its
strategy section as a substring, so the two renderings cannot drift independently) renders
a strategy-free prompt used as the pair's `prompt`. Before it is ever used,
`verify_prompt_lineage(problem, candidates)` re-derives every candidate's stored prompt
hash from `build_prompt(problem, candidate.strategy)` and requires an exact match plus a
current `prompt_version` — proving the canonical prompt is a genuine rendering of the same
template every candidate was actually generated under, not an invention. A failure is
recorded as an `integrity_failure` for every pair the affected problem could have
produced, never a silent fallback to an unverified prompt.

## 3. Selection policies

| Policy | `version` | Admits |
|---|---|---|
| `StrictPolicy` | `strict_v1` | `chosen.correctness == "correct"` and `rejected.correctness == "incorrect"`. **Ignores `minimum_score_margin` entirely.** |
| `MarginPolicy` | `margin_v1` | `chosen_score > rejected_score` and the margin clears `minimum_score_margin` (default `0.2`, configurable, never hard-coded). |
| `AllBetterPolicy` | `all_better_v1` | Any decisive comparison, however small the margin. Never the default. |

Each `admits(chosen, rejected, *, minimum_score_margin)` returns `(bool, reason | None)`
rather than a bare boolean, so a rejection's reason is never re-derived downstream. The
universal exclusions — ties, indeterminate candidates, cross-problem pairs, identical
code, invalid prompt provenance, integrity failures — live in the builder, not in any
policy, so no policy can accidentally omit one.

Strict deliberately ignores the margin (a decision confirmed during planning): the real
data's p008 has a 9/9-vs-8/9 correctness gap but only a 0.111 margin — well under the 0.2
default — and gating strict on the margin would drop it, cutting the strict dataset from
12 pairs to 6 and making the two datasets differ only by a subset relationship rather than
a genuine comparison.

## 4. Pair-building algorithm

`PreferencePairBuilder.build_problem(...)`, per problem (grouped strictly by
`problem_id`, so a cross-problem pair is structurally impossible, not merely checked for):

1. Verify prompt lineage once over every candidate the problem can actually join to a
   real `Candidate` record.
2. For every unordered pair `C(n, 2)` (all assessments, including indeterminate ones —
   dropped explicitly, never assumed to score 0):
   - Re-run Stage 7's `CandidateComparator.compare(a, b)`.
   - `TIE`/`INDETERMINATE` → a rejection, no pair.
   - A missing candidate join, or a candidate missing schema-2.0 provenance hashes →
     `integrity_failure`.
   - Identical `code_sha256` → `identical_code`.
   - A defensive re-statement of the prompt-match check (structurally guaranteed by
     grouping-by-problem; never observed to fire on real data) → `invalid_prompt_match`.
   - The configured policy decides admission; a decline is recorded with the policy's own
     reason.
   - Otherwise, build a `PreferencePair`.
3. Deterministically truncate to `max_pairs_per_problem` (sorted by descending
   `score_margin`, then candidate ids — no RNG anywhere in this module).
4. Deduplicate training records within the problem (`dedupe_training_records`): the first
   occurrence, by `preference_id` order, of each `(prompt, chosen, rejected)` triple
   survives; the rest are flagged `duplicate_training_record` with a
   `canonical_preference_id`, never dropped from `metadata.jsonl`.

`build_run(...)` loops this over every problem and reports `candidates_considered`,
derived (not separately tracked) as the count of distinct candidate ids appearing across
every pair and rejection.

## 5. Deduplication

Three deliberately separate notions:

- `pair_key` — `(problem_id, chosen_id, rejected_id)`, directional; `A>B` and `B>A` are
  different keys, never merged (a `B>A` alongside an existing `A>B` is an invalid reverse
  preference, caught by the validator, not a duplicate).
- `code_identical` — candidate-level `code_sha256` equality. Gates a single pair only;
  never removes a candidate from the pool entirely (it may still pair against a third,
  distinct candidate).
- `training_key` — `(prompt, chosen, rejected)` text triple, the identity a DPO trainer
  actually sees. This is what collapses the real strict run's 12 pairs to 3 distinct
  training records.

On real, deterministic evaluation, identical code always produces an identical score, so
the comparator's tie check intercepts *before* `code_identical` is ever consulted —
confirmed against the real ranking run (`duplicates: 0`). The check is retained as a
structural guard against a future non-deterministic evaluation (a flaky test, a timeout),
not because it fires today.

## 6. Splitting

`ProblemSplitter.split(problem_ids)` — the split unit is `problem_id`, never a pair, so
every pair from one problem lands in exactly one split. The pool is the problems that
actually produced a training pair (a decision confirmed during planning), not the entire
ten-problem dataset — splitting all ten when only two produce pairs was measured, over a
1000-seed sweep, to leave a split empty 2.1% of the time and put a pair-bearing problem
outside `train` 37.5% of the time. A floor rule keeps `train` non-empty whenever the pool
is non-empty, even when the arithmetic floor rounds to zero. `random.Random(seed)` over a
**sorted** pool is the only randomness anywhere in the package.

## 7. Dataset directory / run architecture

```
data/preferences/runs/pref_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json        # policy, versions, margin, split config, upstream run ids, status
├── metadata.jsonl       # one PreferencePair per generated pair, including collapsed ones
├── rejections.jsonl     # one PreferenceRejection per excluded candidate pair, with a reason
├── preferences.jsonl    # {prompt, chosen, rejected} training records, deduped
├── split_manifest.json  # train/validation/test problem-id membership, seed, ratios
├── train.jsonl / validation.jsonl / test.jsonl
├── statistics.json      # reconstructable from metadata.jsonl + rejections.jsonl
└── quality_report.json  # distributions and a reason per pairless problem
```

`metadata.jsonl` is both the durable, append-as-computed ledger (mirroring Stage 7's
never-rewrite append pattern, so a killed run leaves a resumable file behind) *and* the
full spec-required audit artifact — there is no separate `pairs.jsonl`.
`preferences.jsonl`/`train.jsonl`/`validation.jsonl`/`test.jsonl` are whole-file atomic
replacements, always fully rebuilt together from `metadata.jsonl` once a run completes,
never a partial JSONL append.

`PreferenceRunRepository` mints `pref_YYYYMMDD_HHMMSS_xxxx` ids and owns the run status
lifecycle (`created → running → completed/failed/interrupted/cancelled`), mirroring
`RankingRunRepository` — a fourth, independent copy of this plumbing (after Stages 4, 6,
7) rather than a shared base, the same deliberate deferral Stage 7 made to keep this
stage's blast radius contained.

## 8. Versioning strategy

`preference_version` (`v1`, the record-schema version), `selection_policy_version`
(`strict_v1`/`margin_v1`/`all_better_v1`, the algorithm version — distinct from the
dataset-format version), `dataset_schema_version` (`dpo_preference_v1`), and
`builder_version`. A new policy, margin, or algorithm produces a new preference run;
historical runs are immutable.

## 9. CLI commands added

| Command | Behavior |
|---|---|
| `preferences generate --ranking-run-id ID [--policy] [--margin] [--max-pairs-per-problem] [--split-seed] [--resume] [--force]` | Build preference pairs from a ranking run |
| `preferences list` | Preference runs, newest first |
| `preferences show --preference-run-id --preference-id [--show-code]` | Inspect one pair |
| `preferences stats --preference-run-id` | Counters plus train/validation/test counts |
| `preferences validate --preference-run-id` | The full cross-artifact validator; exit 1 on failure |

Follows `rank run`'s shape, not `evaluate run`'s resume-by-default: a bare invocation
always creates a new preference run; `--resume` is the only way to continue one; `--force`
mints a new run and never mutates an existing one, so `strict_v1`/`margin_v1`/`margin_v2`
from the same ranking run coexist. `preferences list` was added beyond the spec's four
named commands, since none of them is usable without a way to discover a
`preference_run_id`.

## 10. Configuration

A `preferences:` section in `config.yaml` — `policy`, `minimum_score_margin`,
`max_pairs_per_problem`, and a `split:` block (`train`/`validation`/`test` ratios,
`seed`) — parsed by `preferences/config.py`'s `PreferenceConfig` and wrapped by
`config.py`'s `_parse_preferences`, the same one-way-dependency, boundary-translation
pattern as `sandbox:`/`evaluation:`. Unlike Stage 7 (no tunable scoring parameters),
preference generation has four genuinely tunable knobs and the spec explicitly forbids
hard-coding the margin. Every field is overridable per-invocation by the matching CLI
flag. No data-path wiring was needed: `paths.preferences` and `data/preferences/.gitkeep`
already existed from Stage 1's skeleton.

## 11. Preference statistics

`PreferenceStatistics.from_records(manifest, pairs, rejections, candidates_considered=...)`
— always reconstructable from `metadata.jsonl` + `rejections.jsonl`, never trusted from an
in-memory counter. The five headline rejection counters (`ties`, `duplicates`,
`indeterminate`, `prompt_mismatches`, `integrity_failures`) are validated in
`__post_init__` to equal their matching `rejections_by_reason` entries, so the two views
of the same rejections can never silently disagree. `training_records` counts pairs with
`duplicate_training_record == False` — the size of `preferences.jsonl`.

## 12. Unit-test results

```
tests/preferences/test_models.py          32 passed
tests/preferences/test_prompt.py           8 passed
tests/preferences/test_policies.py        11 passed
tests/preferences/test_dedup.py            9 passed
tests/preferences/test_builder.py         17 passed
tests/preferences/test_splitter.py        15 passed
tests/preferences/test_repository.py       9 passed
tests/preferences/test_run_repository.py  13 passed
tests/preferences/test_statistics.py       7 passed
tests/preferences/test_validation.py      12 passed
```

133 unit tests (in the ten non-integration files above), 0 Docker, 0 model calls — the
whole suite is pure computation.

## 13. Integration-test results

```
tests/preferences/test_integration.py      4 passed
```

The spec's strict matrix (`A=10/10, B=8/10, C=10/10, D=5/10, E=0/10` at `tests_total=10`,
plus a deliberately pairless second problem to prove a zero-pair problem is processed, not
silently skipped) verified end-to-end through `validate_preference_run`; the margin
policy's additional partial-vs-partial pairs; the end-to-end provenance/split check
(chosen really outscores rejected, the split is problem-level); the reproducibility test.
137 tests total across the package, all pure computation — no `-m integration` marker
needed, since nothing here touches Docker.

## 14. Determinism-test results

`test_reproducibility_two_runs_over_identical_input_agree` runs the full
build → split → statistics pipeline twice against byte-identical upstream evidence into
two separate preference runs, then strips only `preference_run_id` and `created_at`
before comparing: pairs are asserted equal, and `preferences.jsonl` is asserted
byte-identical. Also verified directly against the real ranking run: `preferences generate
--policy strict --force` a second time produces a `preferences.jsonl` and (modulo
run/candidate id ordering) `metadata.jsonl` identical to the first run.

## 15. Example preference output

The real Stage 7 ranking run, `rank_20260817_161726_a84d` (100 candidate pairs across 10
problems, 78 ties), produced these two committed preference runs:

```
$ python -m python_dpo preferences generate --ranking-run-id rank_20260817_161726_a84d --policy strict
Preference run pref_20260818_031940_c4d1 created | ranking run rank_20260817_161726_a84d | policy=strict margin=0.2
Preference run pref_20260818_031940_c4d1 completed | 10 problem(s) processed this call | pairs=12 rejected=88 training_records=3

$ python -m python_dpo preferences stats --preference-run-id pref_20260818_031940_c4d1
Problems processed: 10
Candidates considered: 50
Candidate pairs considered: 100
Pairs generated: 12
Pairs rejected: 88
  Ties: 78
  Duplicate code: 0
  Indeterminate: 0
  Prompt mismatches: 0
  Integrity failures: 0
Strong pairs: 12
Medium pairs: 0
Distinct training records: 3

Policy/other exclusions:
  not_correct_vs_incorrect: 10

Split (seed=42): train=1 validation=0 test=1

$ python -m python_dpo preferences generate --ranking-run-id rank_20260817_161726_a84d --policy margin --margin 0.2
Preference run pref_20260818_031944_4e54 completed | 10 problem(s) processed this call | pairs=10 rejected=90 training_records=4

$ python -m python_dpo preferences validate --preference-run-id pref_20260818_031940_c4d1
Preference dataset validation passed.
$ python -m python_dpo preferences validate --preference-run-id pref_20260818_031944_4e54
Preference dataset validation passed.
```

|  | strict | margin (0.2) | all_better (measured, not committed) |
|---|---|---|---|
| candidate pairs considered | 100 | 100 | 100 |
| ties excluded | 78 | 78 | 78 |
| policy exclusions | 10 | 12 | 0 |
| pairs generated | 12 | 10 | 22 |
| strong / medium | 12 / 0 | 6 / 4 | 12 / 10 |
| distinct training records | 3 | 4 | 7 |
| problems with pairs | p004, p008 | p004, p010 | p004, p007, p008, p010 |

`strict` and `margin` overlap only on `p004` — `strict` includes `p008` (a correctness
gap with a 0.111 margin, which `margin` drops) and `margin` includes `p010` (both
candidates incorrect but 0.5 apart, which `strict` drops). Eight of the ten problems
produce zero pairs under `strict`: five collapse to an all-correct tie, one to an
all-incorrect tie at an identical rate, and two never entered the ranking run's
comparisons at a decisive margin either way. `strict`'s problem-level split (seed 42,
floor rule applied to the 2-problem pool) puts `p008` in `train` and `p004` in `test`,
with `validation` empty.

## 16. Files created/modified

**Created:**

- `src/python_dpo/preferences/` — `__init__.py`, `errors.py`, `models.py`, `prompt.py`,
  `policies.py`, `dedup.py`, `builder.py`, `splitter.py`, `repository.py`,
  `run_repository.py`, `statistics.py`, `validation.py`, `config.py`, `README.md`
- `tests/preferences/` — `__init__.py`, `test_models.py`, `test_prompt.py`,
  `test_policies.py`, `test_dedup.py`, `test_builder.py`, `test_splitter.py`,
  `test_repository.py`, `test_run_repository.py`, `test_statistics.py`,
  `test_validation.py`, `test_integration.py`
- `data/preferences/runs/pref_20260818_031940_c4d1/` — the real `strict` preference run
- `data/preferences/runs/pref_20260818_031944_4e54/` — the real `margin` (0.2) preference run
- `08_PREFERENCE_PAIR_GENERATION.md` (this file)

**Modified:**

- `src/python_dpo/generation/prompt_builder.py` / `generation/__init__.py` —
  `build_canonical_prompt`/`CANONICAL_PROMPT_VERSION`, derived from the existing
  `_TEMPLATE` rather than a second hand-authored literal; `build_prompt`'s own output is
  byte-for-byte unchanged
- `src/python_dpo/config.py` — the `preferences:` section (`PreferenceConfig`), threaded
  through `Config`; no new data path needed (already present since Stage 1)
- `config.yaml` — the `preferences:` settings section
- `src/python_dpo/cli.py` — the `preferences` command group; `_PLACEHOLDER_STAGES` loses
  `"preferences"`
- `tests/test_project.py` — `preferences` CLI parsing/error-path tests; the placeholder
  assertion updated
- `src/python_dpo/__init__.py` — version `0.7.0` → `0.8.0`
- `README.md`, `src/python_dpo/README.md`, `data/README.md`, `tests/README.md` — Stage 8
  documentation

## 17. Dependencies added

**None.** Preference generation is pure Python standard library computation over
already-persisted JSONL — no new runtime or dev dependency anywhere in the stack.

## 18. Deviations from the specification

- **The pair prompt is a canonical, strategy-free rendering of the problem**, not either
  candidate's stored generation prompt — the spec's literal prompt-equality check would
  produce zero pairs under every policy, since every candidate of a problem carries a
  distinct, strategy-specific `prompt_sha256`. Lineage is verified against the current
  generation template, not waived; both candidates' generation-prompt hashes are kept in
  `metadata.jsonl` (confirmed during planning).
- **`minimum_score_margin` does not gate the `strict` policy** — the spec's margin section
  reads as a global filter, but the strict-policy section defines it purely on
  correctness. The latter is adopted so `strict` and `margin` remain a genuine comparison
  rather than one being a strict subset of the other (confirmed during planning).
- **`preferences.jsonl`/the split files are deduplicated by `(prompt, chosen, rejected)`**
  while `metadata.jsonl` keeps every generated pair — the spec requires the validator to
  reject duplicate training records, and 12 real strict pairs are only 3 distinct triples
  (confirmed during planning).
- **The split pool is the pair-bearing problems, not the entire dataset**, plus a floor
  rule guaranteeing a non-empty `train` split — the spec is silent on an empty pool, and a
  1000-seed sweep showed splitting all ten problems leaves a split empty 2.1% of the time.
- **`rejections.jsonl` is persisted** although the spec's artifact tree omits it — required
  by the spec's own "record the reason" rule for excluded pairs and CLAUDE.md's
  data-integrity rule.
- **`preference_id` is `pref_<chosen_id>__<rejected_id>`**, dropping the spec's leading
  problem-id example segment since every `candidate_id` already carries it.
- **`max_pairs_per_problem` truncation is a deterministic margin-ordered sort, with no
  RNG** — the spec's seed requirement for random sampling therefore does not arise; the
  manifest records the ordering rule instead.
- **Pair-level code deduplication never fires on the current data** — identical code
  implies an identical score implies a `TIE`, already excluded upstream. Retained as a
  structural guard against a future non-deterministic evaluation, not because it is
  load-bearing today.
- **The comparator is re-run rather than read from Stage 7's persisted
  `comparisons.jsonl`** — the spec explicitly says not to trust a stored score blindly;
  the persisted comparisons were not consulted at all in this implementation (an even
  stricter reading than "cross-check" — the builder never opens that file).
- **A fourth, independent copy of the run-directory-management plumbing**
  (`run_repository.py`) rather than a shared base extracted across Stages 4, 6, 7, and 8 —
  the same deliberate deferral Stage 7 made, to keep this stage's blast radius contained.
- **A `preferences:` config section exists**, where Stage 7 added none — this stage has
  four genuinely tunable parameters and the spec forbids hard-coding the margin.
- **`preferences list` was added** beyond the spec's four named commands, since none of
  them is usable without a way to discover a `preference_run_id`.
- **Resume is implemented but has no practical value at this data scale** — required by
  the spec's resumability and acceptance-criteria sections; building 12 pairs from 50
  assessments is milliseconds of in-memory work.

## 19. Known limitations

- **The indeterminate and prompt-mismatch paths have zero coverage from real data.** The
  committed ranking run has `indeterminate: 0`, so both branches — in the builder and the
  validator's cross-checks — are exercised only by synthetic fixtures in the test suite.
- **`identical_code` is unreachable on any deterministic evaluation.** Since identical
  code always produces an identical score, and equal score is always a `TIE`, the
  `identical_code` rejection reason can only fire if a future evaluation run is
  non-deterministic (a flaky test, a timeout that varies by run). The test suite exercises
  it only by constructing an artificial mismatch between candidate code and assessment
  score.
- **`derive_candidates_considered` under-counts a problem with fewer than two assessed
  candidates.** It counts distinct candidate ids appearing in at least one considered
  pair; a lone candidate with no pairing partner never appears in any pair or rejection
  record. Every problem in the real dataset has 5 candidates, so this never manifests
  today, but a future problem generated with `candidates_per_problem = 1` would silently
  exclude that candidate from the count.
- **At ten problems, no policy's split is large enough for meaningful DPO training.** The
  committed `strict` dataset is 3 distinct training records from 2 problems, with
  `validation` empty. This is the correct output for this input — the spec explicitly
  says to optimize for label confidence over pair count, and its own §65 already concedes
  ten problems cannot support real training — but it means Stage 8 validates the
  *pipeline*, not yet a usable training dataset. `margin_v1`'s 4 records are similarly
  small. Both are committed as evidence the pipeline is byte-for-byte correct, not as a
  training-ready dataset.
