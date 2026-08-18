# Stage 8 — DPO Preference Pair Generation

## Context

Stage 7 stopped one deliberate step short of a preference. `data/rankings/runs/rank_20260817_161726_a84d/`
holds 50 `CandidateAssessment` records, 50 `RankingResult` records and 100 `ComparisonResult`
records that say *A is better than B* in neutral terms — `A_BETTER`, `B_BETTER`, `TIE`,
`INDETERMINATE` — and nothing that says `chosen` or `rejected`. Stage 7's spec §67 forbade
that vocabulary precisely so this stage could own it.

`.claude/specs/08_preference_pair_generation.md` asks for the layer that converts that
objective ordering into DPO training data: `{prompt, chosen, rejected}` triples where
`chosen` is better than `rejected` **because execution evidence says so**, plus the audit
trail, splits, statistics and validation that make the claim checkable.

Three boundaries define the stage:

- **Model-free (§87).** No Qwen, no Claude, no embeddings, no reward model, no judge. Every
  label comes from Stage 6's pytest counts by way of Stage 7's comparator.
- **Code is never touched (§12, §13, §14, §97, §99).** `chosen` and `rejected` are the exact
  bytes Qwen emitted, as persisted by Stage 3. No formatter, no repair, no fences, no
  "here is the correct solution", no "passes 10/10 tests".
- **No ties, ever (§17).** Equal score is not a preference. This matters more here than
  anywhere: 78 of the 100 comparisons in the real run are ties.

**Outcome:** `preferences generate --ranking-run-id rank_20260817_161726_a84d --policy strict`
produces `data/preferences/runs/pref_.../` with training JSONL, audit metadata, splits,
statistics and a quality report.

---

### What exploration established

Everything below is **measured from the committed artifacts**, not assumed. Several
findings change the shape of the stage.

| Finding | Consequence |
|---|---|
| **All 50 candidates have distinct `prompt_sha256`.** The generation prompt embeds the strategy (`"Strategy: Solve the problem using a clear, correct Python implementation."`) and each of the 5 candidates per problem used a different strategy | §41's literal check (`chosen.prompt_sha256 == rejected.prompt_sha256`) yields **zero pairs under every policy**. This is the blocking finding; decision 1 resolves it |
| 100 comparisons: **78 `TIE`, 16 `A_BETTER`, 6 `B_BETTER`** | Only 22 pairs are orderable at all. §17's tie exclusion governs 78% of the input |
| **strict → 12 pairs across 2 problems** (p004, p008); **margin 0.2 → 10 pairs across 2 problems** (p004, p010); **all_better → 22 pairs across 4 problems** (p004, p007, p008, p010) | These are the exact acceptance numbers |
| **strict is not a superset of margin.** strict includes p008 (9/9 vs 8/9, margin 0.111) which margin drops; margin includes p010 (4/6 vs 1/6, both incorrect) which strict drops | §93's "compare the two datasets" is meaningful — they overlap only on p004 |
| **The pairs collapse hard at the text level**: strict's 12 pairs are only **3 distinct** `(prompt, chosen, rejected)` triples; margin's 10 are 4; all_better's 22 are 7. p008 alone contributes 6 pairs that are 6 byte-identical records (3 candidates share code `a3c68dbb`, 2 share `1ff6fbd6`) | §72 requires the validator to reject exact duplicate records — so without decision 3 the validator fails on our own output |
| **Pair-level code dedup (§33, §34) never fires on this data.** Identical code ⇒ identical score ⇒ `TIE` ⇒ already excluded upstream | The check stays, as a structural guard against a future non-deterministic evaluation (flaky test, timeout). Honest framing: defensive, not load-bearing |
| **Zero `indeterminate` candidates and zero timeouts in the real run** (`indeterminate: 0` in `statistics.json`) | §38/§88's indeterminate exclusion has no real coverage — synthetic fixtures only, exactly as Stage 7 found |
| Only **2 of 10 problems** produce strict pairs | Any problem-level 80/10/10 split is degenerate. Measured over 1000 seeds on the all-10 pool: 375 put a strict problem outside train, 21 leave train empty. Decision 4 picks the pool; §65 already says to document rather than engineer this away |
| `config.yaml` **already declares `paths.preferences: data/preferences`**, `_REQUIRED_PATH_KEYS` already lists it, and `data/preferences/.gitkeep` is committed | Unlike Stage 7, **no data-path wiring is needed**. `tests/test_project.py`'s three literal path enumerations are untouched |
| `cli.py:91` `_PLACEHOLDER_STAGES` still contains `"preferences"` | Must be removed, and `_add_preferences_parser` added at `cli.py:1818` |
| `Candidate` already carries `prompt`, `prompt_sha256`, `code`, `code_sha256`, `strategy`, `prompt_version` — and `__post_init__` **recomputes and verifies all three hashes** on load | §42's candidate-integrity check is largely free: a tampered `candidates.jsonl` cannot be loaded at all. The pair builder still asserts run/problem membership explicitly |
| `generation/prompt_builder.build_prompt(problem, strategy)` is a pure function; `PROMPT_VERSION = "v1"` | The canonical prompt of decision 1 can be derived in the same module, and the derivation *verified* against every candidate's stored `prompt_sha256` |
| `ComparisonResult` is already persisted with `relation` and `score_margin` | Tempting, but §43 says "do not blindly trust stored rank numbers": the builder **re-runs `CandidateComparator`** over freshly loaded assessments (§44) and treats `comparisons.jsonl` as a cross-check, not a source |

---

### Decisions confirmed with the user

1. **Canonical problem prompt.** The pair prompt is a strategy-free prompt derived per
   *problem*, not the per-candidate generation prompt. §41's equality check becomes equality
   of `canonical_prompt_sha256`, which is trivially satisfied within a problem and still
   fails loudly across problems. Both candidates' generation-prompt hashes and strategies are
   preserved in `metadata.jsonl`. Rationale beyond unblocking: a prompt reading *"Strategy:
   explore an alternative approach"* would make generation-strategy conditioning part of the
   training input, which contradicts §60 ("do not use strategy as a preference signal") and
   §95 (model-agnostic representation). **Integrity is preserved, not waived** — see
   `prompt.py` below, which proves the canonical prompt derives from the same template the
   candidates were generated under before emitting it.

2. **Strict ignores `minimum_score_margin`.** §25 defines strict purely as correct vs
   incorrect; the margin is §26's instrument. Keeping them independent is what makes the two
   datasets a real comparison. Concretely this keeps p008 (correctness gap, margin 0.111) in
   the strict dataset: 12 pairs rather than 6.

3. **Dedup the training file, keep every pair in the audit file.** `preferences.jsonl`,
   `train/validation/test.jsonl` carry only distinct `(prompt, chosen, rejected)` triples;
   `metadata.jsonl` carries all 12 rows, the 9 collapsed ones marked
   `duplicate_training_record` with a `canonical_preference_id` pointing at the survivor.
   Satisfies §72 without violating the repo's "never silently discard" rule.

4. **The split pool is the set of problems that produced at least one pair**, with a floor
   guaranteeing train is non-empty. Splitting all 10 would spend the entire 10% validation
   and 10% test budget on problems contributing zero rows.

---

## New package — `src/python_dpo/preferences/`

House style throughout, copied from `ranking/`: frozen dataclasses validating in
`__post_init__`, explicit `to_dict()`/`from_dict()` rejecting both unknown and missing
fields against a module-level `frozenset`, plain-string closed sets rather than `enum.Enum`,
`created_at` stamped via `object.__setattr__` with `utc_now_iso()`, two-tier repositories
over `atomic_io`, a per-package `README.md`.

**`errors.py`** — `PreferenceError` base; `PreferenceConfigError`, `PreferencePolicyError`,
`RankingRunNotFoundError`, `PreferenceRunNotFoundError`, `PreferenceStoreError`. Stage 7
errors are translated, never leaked.

**`models.py`** — the schema and the closed sets.

```
POLICIES             = {strict, margin, all_better}                    # §24
PREFERENCE_STRENGTHS = {strong, medium}                                # §45 — never "weak"
SPLITS               = {train, validation, test}                       # §64
PREFERENCE_RUN_STATUSES = {created, running, completed, failed, interrupted, cancelled}
DATASET_SCHEMA_VERSION = "dpo_preference_v1"                           # §85
PREFERENCE_VERSION     = "v1"                                          # §28
```

`PreferencePair` — §7's required fields (`preference_id`, `problem_id`, `candidate_run_id`,
`ranking_run_id`, `chosen_candidate_id`, `rejected_candidate_id`, `prompt`, `chosen`,
`rejected`) plus §7's metadata (`chosen_score`, `rejected_score`, `score_margin`,
`chosen_pass_rate`, `rejected_pass_rate`, `chosen_tests_passed`, `rejected_tests_passed`,
`selection_policy`, `preference_version`, `created_at`), plus `evaluation_run_id`,
`chosen_tests_total`/`rejected_tests_total` (§53), `chosen_correctness`/
`rejected_correctness`, `preference_strength` (§45), `selection_policy_version` (§86),
`canonical_prompt_sha256` + `prompt_version` (§11), decision-1's
`chosen_generation_prompt_sha256`/`rejected_generation_prompt_sha256`, decision-3's
`duplicate_training_record: bool` + `canonical_preference_id: str | None`, and §60's
`chosen_strategy`/`rejected_strategy` + `chosen_code_sha256`/`rejected_code_sha256`.

`__post_init__` enforces §16 as construction-time invariants, so an invalid pair cannot
exist in memory: `chosen_candidate_id != rejected_candidate_id`; `chosen != rejected`;
`chosen_code_sha256 != rejected_code_sha256`; `chosen_score > rejected_score`;
`score_margin == chosen_score - rejected_score`; both candidate ids prefixed by
`problem_id`; `preference_strength == "strong"` **iff**
`(chosen_correctness, rejected_correctness) == ("correct", "incorrect")`; neither
correctness is `indeterminate` (§38).

`PreferenceRejection` (§77, and CLAUDE.md's data-integrity rule) — `ranking_run_id`,
`problem_id`, `candidate_a`, `candidate_b`, `reason`, `detail`, `relation`, `score_a`,
`score_b`, `score_margin`, `created_at`. `reason` is a closed set:

```
tie · indeterminate · identical_code · insufficient_margin · not_correct_vs_incorrect
invalid_prompt_match · integrity_failure · max_pairs_per_problem · duplicate_training_record
```

`PreferenceManifest` (§51) — `preference_run_id`, `ranking_run_id`, `evaluation_run_id`,
`candidate_run_id`, `status`, `created_at`/`started_at`/`completed_at`,
`preference_version`, `selection_policy`, `selection_policy_version`,
`minimum_score_margin`, `max_pairs_per_problem`, `split_ratios`, `split_seed`,
`dataset_schema_version`, `builder_version`, `error`. `with_status()` reuses Stage 4/6/7's
transition graph shape.

`PreferenceStatistics` (§54) — the ten §54 counters (`problems_processed`,
`candidates_considered`, `candidate_pairs_considered`, `pairs_generated`, `pairs_rejected`,
`ties`, `duplicates`, `indeterminate`, `prompt_mismatches`, `integrity_failures`) plus
`strong_pairs`, `medium_pairs`, `training_records`, `rejections_by_reason: dict[str, int]`,
`per_problem: dict[str, dict]`, `computed_at`, `statistics_version`. Built by
`from_records(manifest, pairs, rejections, *, computed_at=None)` — keyword-only
`computed_at` so the validator can recompute against the on-disk timestamp and compare with
`==`, exactly as `ranking/models.py` does.

`QualityReport` (§75) — `total_pairs`, `strong_pairs`, `medium_pairs`,
`score_margin_distribution`, `chosen_pass_rate_distribution`,
`rejected_pass_rate_distribution`, `strategy_distribution` (§61: chosen-strategy and
rejected-strategy histograms), `problems_with_pairs`, `problems_without_pairs` with §76's
reason per problem. **Reported, never enforced** (§92).

**`prompt.py`** — decision 1, and the reason it is a reconstruction rather than an invention.

`CANONICAL_PROMPT_VERSION = "v1"`. `build_canonical_prompt(problem)` is added to
`generation/prompt_builder.py` alongside the existing `build_prompt(problem, strategy)`, so
both renderings live in the one module the `PROMPT_VERSION` docstring already governs; this
module wraps it with the verification step:

```
verify_prompt_lineage(problem, candidates) -> canonical_prompt, canonical_prompt_sha256
```

For every candidate of the problem it asserts
`sha256_text(build_prompt(problem, candidate.strategy)) == candidate.prompt_sha256` and
`candidate.prompt_version == PROMPT_VERSION`. Passing means the canonical prompt is
demonstrably the same template, minus the strategy block, that these candidates were
actually generated under. Failing raises `integrity_failure` and the problem produces no
pairs — never a silent fallback. This is what keeps §10 ("do not invent a new problem
statement") and §42 honest under decision 1.

**`policies.py`** — §24–§28. A `PreferencePolicy` protocol:

```
name: str · version: str
admits(comparison, chosen, rejected, *, minimum_score_margin) -> (bool, reason | None)
```

Returning the rejection *reason* rather than a bare bool is what makes §77 and §54's
`rejections_by_reason` possible without the builder re-deriving why.

| Policy | `version` | Admits | §refs |
|---|---|---|---|
| `StrictPolicy` | `strict_v1` | `chosen.correctness == "correct"` and `rejected.correctness == "incorrect"`. **No margin gate** (decision 2) | §25, §46 |
| `MarginPolicy` | `margin_v1` | `chosen_score > rejected_score` and `margin >= minimum_score_margin` | §26, §47 |
| `AllBetterPolicy` | `all_better_v1` | `chosen_score > rejected_score`. Never the default (§27) | §27 |

`minimum_score_margin` defaults to **0.2** (§22) and is read from config, never hard-coded
in the builder. Policies are pure and stateless; the universal exclusions (tie,
indeterminate, cross-problem, identical code, prompt mismatch, integrity) live in the
builder so no policy can accidentally omit one.

**`builder.py`** — `PreferencePairBuilder`, `BUILDER_VERSION = "v1"`.

Per problem (§37 — grouping by `problem_id` first is what makes §36 structural rather than
a check that could be forgotten):

1. Load the problem's assessments; run `verify_prompt_lineage` once.
2. Drop `indeterminate` candidates up front (§38) — recorded as `indeterminate` rejections,
   never coerced to score 0 (§38 is explicit about this).
3. For every unordered pair `C(n,2)` (§29), call **Stage 7's** `CandidateComparator.compare`
   (§44) — re-derived from the assessments, not read from `comparisons.jsonl` (§43).
4. `TIE`/`INDETERMINATE` → rejection, no pair (§17). Otherwise the relation fixes the
   direction; only one direction is ever emitted (§30, §71).
5. Universal exclusions, each with its own reason: identical `code_sha256` (§33, §34),
   `canonical_prompt_sha256` mismatch → `invalid_prompt_match` (§41), integrity failure
   (§42).
6. Apply the configured policy.
7. Truncate to `max_pairs_per_problem` (§57) **deterministically** (§58): sort surviving
   pairs by `(-score_margin, chosen_candidate_id, rejected_candidate_id)` and keep the first
   N; the rest become `max_pairs_per_problem` rejections. No RNG at all — §58's seed
   requirement therefore does not arise, and the manifest records the ordering rule instead.
   A `PairSelector` seam exists so §59's future balancing policy has somewhere to live; v1
   preserves all pairs by default (`max_pairs_per_problem` defaults to `None`).
8. Text-level dedup (decision 3): first occurrence in `preference_id` order wins; the rest
   are emitted to `metadata.jsonl` with `duplicate_training_record = True` and
   `canonical_preference_id` set.

`preference_id` is `pref_<chosen_candidate_id>__<rejected_candidate_id>`, e.g.
`pref_p004_c001__p004_c004` — deterministic, never a UUID (§31). §31's example repeats the
problem id, which every `candidate_id` already carries; recorded as a deviation.

**`dedup.py`** — the three dedup notions kept deliberately separate, because §32/§33/§72/§73
mean three different things and conflating them is how a dataset silently loses rows:

| Function | Identity | Spec |
|---|---|---|
| `pair_key(pair)` | `(problem_id, chosen_candidate_id, rejected_candidate_id)` — an ordered pair; `A>B` and `B>A` are **not** duplicates of each other (§32), the second is an invalid reverse preference (§71) | §32, §73 |
| `code_identical(a, b)` | `a.code_sha256 == b.code_sha256` | §33, §34, §74 |
| `training_key(pair)` | `(prompt, chosen, rejected)` — the §72 identity, and the one that collapses 12 → 3 | §72 |

§74's asymmetry is preserved: candidates with identical code may not pair with *each other*
but each may still pair against a third candidate.

**`splitter.py`** — `ProblemSplitter`, `SPLITTER_VERSION = "v1"`. §62–§68.

The split unit is `problem_id` (§63); all pairs of a problem land in one split, which is
what prevents the same prompt appearing in train and validation (§62). Pool = problems with
≥1 pair (decision 4). Deterministic: `random.Random(seed).shuffle(sorted(pool))` — sorted
first so pool ordering cannot leak in, no timestamps, no UUIDs (§67). Sizes are
`floor(n * ratio)` for train and validation with the remainder to test, then a **floor rule**
moves one problem into train if train would otherwise be empty. Ratios and seed come from
config (`0.8 / 0.1 / 0.1`, seed `42`) and are persisted in both `manifest.json` and
`split_manifest.json` (§66, §68).

**`repository.py`** — `PreferenceRepository(directory)`, run-scoped, over `atomic_io`
`append_jsonl`/`iter_jsonl`, mirroring `ranking/repository.py`. §82's API: `save(pair)`,
`get(preference_id)`, `list()`, `list_by_problem(problem_id)`, `count()`,
`exists(preference_id)`, plus `save_rejection`, `load_rejections`, `write_dataset(...)` for
the derived JSONL files and `paired_problem_ids()` as the resume index (§83). No filesystem
access to the run directory happens anywhere outside this class (§82). A malformed line
raises `PreferenceStoreError(f"{path}:{number}: {exc}")` — never skipped.

**`run_repository.py`** — `PreferenceRunRepository(preferences_root)`, mirroring
`ranking/run_repository.py`: mints `pref_YYYYMMDD_HHMMSS_xxxx` (§50) via the same
collision-checked `secrets.token_hex(2)` template, owns `manifest.json`/`statistics.json`/
`quality_report.json`, `create_run`/`get_run`/`list_runs`/`results`, and the status
lifecycle. **Fourth copy of the run plumbing** — Stage 7 deliberately deferred extracting a
shared base; that decision stands here so Stage 8's blast radius stays contained, and it is
re-flagged as debt.

**`statistics.py`** — `format_preference_statistics(stats) -> str`,
`format_quality_report(report) -> str`, `format_pair_table(rows) -> str`. Following
Stage 7's (newer) convention of formatters living in the package rather than inline in the
CLI, since `preferences stats` and `preferences show` share renderers.

**`validation.py`** — `PreferenceDatasetValidator` (§69), mirroring `ranking/validation.py`:
`PreferenceValidationIssue(check, message)`, `PreferenceValidationReport(preference_run_id,
issues, .valid)`, `validate_preference_run(run_dir, ranking_run_dir=None,
candidate_run_dir=None) -> report`, `format_preference_report(report) -> str`. Reads raw
JSONL directly rather than through the repository, so one call collects every issue instead
of raising on the first bad record. All issues fatal, no severity levels.

Checks, and the two levels they operate at. Internal coherence (§16, §70) comes free from
`PreferencePair.__post_init__` — a tampered record fails to construct. The checks that earn
their keep are the **cross-artifact** ones (§69's "provenance", "objective preference"),
which reload the ranking run and re-run Stage 7's comparator against the pair's claim:

- schema: every training record has `prompt`/`chosen`/`rejected`, none empty (§69)
- `chosen != rejected` in every training record (§69)
- every `metadata.jsonl` row references a candidate that exists in the named candidate run,
  belongs to the named problem, and whose `code_sha256` matches (§42, §69)
- `chosen_score`/`rejected_score` re-derived from `assessments.jsonl`, and
  `comparator.compare()` re-run to confirm the direction still holds (§43, §69, §70)
- policy-specific direction (§70): strict ⇒ `correct` vs `incorrect`; margin/all_better ⇒
  `chosen_score > rejected_score`, and margin ⇒ `margin >= minimum_score_margin`
- **no reverse pairs** — `(problem, A, B)` and `(problem, B, A)` both present (§71)
- **no duplicate training records** — the §72 `(prompt, chosen, rejected)` check over
  `preferences.jsonl`, which decision 3 is what makes passable
- same prompt is *not* a duplicate (§73) — asserted as a positive test, since the naive
  reading of §72 would wrongly reject a legitimate multi-pair problem
- no problem id appears in more than one split; every split member has ≥1 pair; splits
  reproduce from `split_manifest.json`'s seed and ratios (§62, §67, §68)
- `statistics.json` recomputed via `from_records(..., computed_at=<on-disk>)`, compared `==`

---

## Persistence layout (§49, §102)

```
data/preferences/runs/pref_20260818_090000_a123/
├── manifest.json        # policy, versions, margin, split config, upstream run ids, status
├── preferences.jsonl    # §52 training records: {prompt, chosen, rejected} ONLY, deduped
├── metadata.jsonl       # §53 full audit row per generated pair, including collapsed ones
├── rejections.jsonl     # §77 every excluded candidate pair with its reason
├── statistics.json      # §54 counters
├── quality_report.json  # §75 distributions
├── split_manifest.json  # §68 split membership, seed, ratios
├── train.jsonl          # §64 — same shape as preferences.jsonl
├── validation.jsonl
└── test.jsonl
```

`preferences.jsonl` and the three split files contain **exactly three keys** (§52, §94) — no
candidate ids, no run ids. Everything else lives in `metadata.jsonl`, keyed by
`preference_id`. Historical preference runs are immutable (§84): a re-generation is a new
run, so `strict_v1`, `margin_v1` and `margin_v2` coexist (§48) without rerunning Qwen,
Docker or pytest.

`rejections.jsonl` is not in §102's list; it is added because §77 requires the reason to be
recorded and CLAUDE.md forbids silently discarding evaluation outcomes. Recorded as a
deviation.

---

## Modifications to existing code

**`src/python_dpo/generation/prompt_builder.py`** — add
`build_canonical_prompt(problem) -> str` and `CANONICAL_PROMPT_VERSION = "v1"` beside the
existing `build_prompt(problem, strategy)`, sharing one template so the two renderings
cannot drift. The module docstring's rule ("bumping the template requires bumping
`PROMPT_VERSION`") is extended to cover it. No change to `build_prompt` — Stage 3's output
must stay byte-reproducible.

**`config.yaml` + `src/python_dpo/config.py`** — a new `preferences:` section, following the
`evaluation:` pattern exactly (the sub-package owns `PreferenceConfig` and
`PreferenceConfigError`; `config.py` adds `_parse_preferences` translating the error so the
dependency stays one-way):

```yaml
preferences:
  policy: strict
  minimum_score_margin: 0.2      # §21, §22 — configurable, never hard-coded
  max_pairs_per_problem: null    # §57 — null keeps all valid pairs
  split:
    train: 0.8
    validation: 0.1
    test: 0.1
    seed: 42                     # §66
```

`Config.preferences: PreferenceConfig` sits alongside `Config.paths.preferences: Path` —
the same benign pairing `Config.evaluation` / `Config.paths.evaluations` already has. **No
data-path wiring is needed**: `paths.preferences` and `data/preferences/.gitkeep` already
exist, so `tests/test_project.py`'s three literal path enumerations are untouched.

**`src/python_dpo/cli.py`** — drop `"preferences"` from `_PLACEHOLDER_STAGES` (line 91) and
add `_add_preferences_parser(subparsers)` at line 1818. CLI flags override config (§78).

| Command | Behavior |
|---|---|
| `preferences generate --ranking-run-id ID [--policy strict\|margin\|all_better] [--margin M] [--max-pairs-per-problem N] [--split-seed S] [--resume PREF_ID] [--force]` | §78, §83, §84 |
| `preferences validate --preference-run-id ID` | §79; prints `Preference dataset validation passed.`; exit 1 on failure |
| `preferences stats --preference-run-id ID` | §80's counters plus train/validation/test counts |
| `preferences show --preference-run-id ID --preference-id PREF [--show-code]` | §81 |
| `preferences list` | Preference runs, newest first — mirrors `rankings list`; needed to discover an id for the other three |

Handlers keep the house contract: `(args, config) -> int`, user data to `sys.stdout` with
fixed-width f-string columns, errors via `logger.error`, exit codes 0/1/2/130. Resume
(§83) skips problems already in `metadata.jsonl`; `--force` mints a new run and never
mutates an existing one (§84). As in Stage 7, resume has **no practical value at this scale**
— building 12 pairs is milliseconds of in-memory work — and is implemented because §83 and
the §100 acceptance criteria require it.

**`src/python_dpo/__init__.py`** — `__version__` → `0.8.0`.

**Docs** — `src/python_dpo/preferences/README.md` (new), plus Stage 8 sections in `README.md`
(roadmap row, repository layout, CLI block, configuration, a `## Stage 8` section shaped like
Stage 7's), `src/python_dpo/README.md`, `data/README.md` (replacing the `### preferences/`
stub), `tests/README.md`.

---

## Tests

`tests/preferences/`. **Pure computation — no Docker, no model**, like `tests/ranking/`, so
everything runs in the default offline suite. Plain module-level functions, no classes, no
`conftest.py`, `make_X(**overrides)` factories defined per module, `tmp_path` for disk.

- **`test_models.py`** — §16 invariants as construction failures: same candidate id;
  identical code; `chosen_score <= rejected_score`; wrong `score_margin`; indeterminate side;
  `preference_strength` disagreeing with the correctness pair. Round-trip
  `to_dict`/`from_dict`, unknown and missing field rejection.
- **`test_prompt.py`** — decision 1: the canonical prompt is identical for all 5 strategies
  of a problem and differs across problems; `verify_prompt_lineage` passes on real committed
  candidates and raises when a candidate's `prompt_sha256` doesn't match its rebuilt prompt
  or its `prompt_version` has drifted; the canonical prompt contains no `Strategy:` block and
  no §98 label leakage.
- **`test_policies.py`** — §88's table exactly. strict: correct vs incorrect include,
  correct vs correct exclude, incorrect vs incorrect exclude; **and decision 2's case,
  9/9 vs 8/9 (margin 0.111) included under strict**. margin @0.2: 1.0v0.8 include,
  0.8v0.7 exclude, 0.9v0.7 include, 0.8v0.6 include. all_better: 0.9v0.8 include.
  Margin threshold read from config, not literal, in every case (§22).
- **`test_builder.py`** — ties excluded (§17); indeterminate vs correct and indeterminate vs
  incorrect both excluded (§38, §88); identical code excluded (§33) including the
  cross-strategy case (§34); cross-problem never generated (§36, §37); prompt mismatch →
  `invalid_prompt_match` (§41); one direction only (§30); deterministic `preference_id`
  (§31); `max_pairs_per_problem` truncation is deterministic and margin-ordered (§57, §58);
  every exclusion produces a `PreferenceRejection` — asserted by counting
  `pairs_generated + pairs_rejected == candidate_pairs_considered`, which is the mechanical
  form of "never silently discard".
- **`test_dedup.py`** — §74's worked example (A=code X 1.0, B=code X 1.0, C=code Y 0.5 ⇒
  `A>C` and `B>C` valid, `A>B` invalid); `A>B` vs `B>A` are not duplicates of each other
  (§32); `training_key` collapse; same prompt is not a duplicate (§73).
- **`test_splitter.py`** — §89: determinism (same seed ⇒ identical split); no problem in two
  splits; all pairs of a problem together; ratios; a different seed gives a different split;
  the floor rule keeps train non-empty at pool size 2; sorted-pool independence (shuffling
  the input list does not change the result).
- **`test_repository.py`** — §82's API, persistence and retrieval, duplicate-id prevention,
  malformed-line rejection with a line number, the resume index, three-key-only training
  records.
- **`test_run_repository.py`** — id minting and collision retry, status lifecycle and illegal
  transitions, `--force` creating a second run that leaves the first byte-identical (§84).
- **`test_statistics.py`** — every §54 counter and §75 distribution against hand-counted
  fixtures; §76's problems-without-pairs reasons.
- **`test_validation.py`** — one test per §69–§72 check, each built by **mutating a real
  valid run** produced by the actual builder (the `tests/ranking/test_validation.py` pattern:
  `build_valid_run(tmp_path)` + `_rewrite_line` / `_append_line`): a reversed pair appended
  (§71), a duplicated training record appended (§72), a flipped score, a chosen code that no
  longer matches its candidate, a problem in two splits, drifted `statistics.json`.
- **`test_integration.py`** — §90's fixture (A=10/10, B=8/10, C=10/10, D=5/10, E=0/10) under
  strict, asserting exactly `A>B, A>D, A>E, C>B, C>D, C>E` and **no A/C pair**; then margin
  0.2 adding the qualifying partial-vs-partial pairs; §91's end-to-end problem → candidates →
  evaluations → ranking → `preferences.jsonl`, verifying chosen really scores higher, no
  ties, no duplicates, correct provenance and a problem-level split; and a **reproducibility
  test** — two runs over identical input differ only in run id and timestamps.

**Synthetic fixtures are mandatory for the indeterminate and prompt-mismatch paths** — the
real ranking run has zero indeterminate candidates, so those branches would otherwise ship
untested.

**Extended** — `tests/test_project.py` gains `preferences` CLI-parsing tests mirroring
`test_rankings_subcommands_parse`, and its placeholder-stage assertion loses `preferences`.

---

## Execution order

1. Write this plan to `.claude/plans/08_preference_pair_generation_plan.md` and add its entry
   to `.claude/plans/README.md`.
2. `errors.py`, `models.py`, `config.py` + tests — pure schema, no dependencies.
3. `prompt.py` + `build_canonical_prompt` in `generation/prompt_builder.py` + tests. **First
   real code, because decision 1 is the stage's load-bearing assumption** — if lineage
   verification fails against the committed candidates, everything downstream changes.
4. `policies.py` + tests — §88's table.
5. `dedup.py` + tests — §74's worked example.
6. `builder.py` + tests, including §90's strict matrix.
7. `splitter.py` + tests — §89.
8. `repository.py`, `run_repository.py` + tests.
9. `statistics.py`, `validation.py` + tests.
10. Config section wiring and CLI wiring; drop the placeholder.
11. `test_integration.py` — §90, §91 and reproducibility last, since they exercise everything.
12. Generate the real `strict` and `margin 0.2` runs; commit both
    `data/preferences/runs/<pref_id>/`; docs; the §106 report.

---

## Verification

```bash
source .venv/bin/activate
pytest -q                                   # offline, zero skips, no Docker

python -m python_dpo rankings list          # discover the ranking run
python -m python_dpo preferences generate --ranking-run-id rank_20260817_161726_a84d --policy strict
python -m python_dpo preferences validate --preference-run-id PREF_ID    # must exit 0
python -m python_dpo preferences stats    --preference-run-id PREF_ID
python -m python_dpo preferences show     --preference-run-id PREF_ID \
    --preference-id pref_p004_c001__p004_c004 --show-code

python -m python_dpo preferences generate --ranking-run-id rank_20260817_161726_a84d \
    --policy margin --margin 0.2
python -m python_dpo preferences validate --preference-run-id PREF_ID_2
```

**Expected output on the real ranking run — computed from the committed artifacts, so these
are exact acceptance numbers, not estimates:**

```
                       strict      margin 0.2   all_better
candidate pairs considered  100         100          100
ties excluded                78          78           78
policy exclusions            10          12            0
preference pairs generated   12          10           22
  strong                     12           6           12
  medium                      0           4           10
distinct training records      3           4            7
problems with pairs            2           2            4
                        p004,p008   p004,p010   p004,p007,p008,p010
problems without pairs         8           8            6
```

`strict`, seed 42, pool `{p004, p008}` → `train = [p008]` (1 record),
`validation = []`, `test = [p004]` (2 records).

```bash
# determinism (§67): a second run differs only in run id and timestamps
python -m python_dpo preferences generate --ranking-run-id rank_20260817_161726_a84d \
    --policy strict --force
diff RUN_A/preferences.jsonl RUN_B/preferences.jsonl                   # empty
diff <(jq -S 'del(.preference_run_id,.created_at)' RUN_A/metadata.jsonl) \
     <(jq -S 'del(.preference_run_id,.created_at)' RUN_B/metadata.jsonl)   # empty

# no problem id in more than one split (§62, the critical one)
jq -r '.train_problem_ids[],.validation_problem_ids[],.test_problem_ids[]' \
   RUN/split_manifest.json | sort | uniq -d                            # empty

# training records carry exactly three keys (§52, §94)
jq -r 'keys|join(",")' RUN/preferences.jsonl | sort -u                 # chosen,prompt,rejected

# nothing upstream was mutated
git diff --stat data/problems/ data/candidates/ data/evaluations/ data/rankings/   # empty
```

Scope containment:

```bash
grep -rniE "\b(qwen|openai|anthropic|llm|judge|embedding|reward)\b" src/python_dpo/preferences/  # none (§87)
grep -rniE "\b(train|lora|qlora|optimizer|torch)\b"  src/python_dpo/preferences/  # none (§3)
grep -rniE "\b(black|autopep8|ruff format|textwrap)\b" src/python_dpo/preferences/  # none (§14)
grep -rn "random\|shuffle" src/python_dpo/preferences/   # only the seeded splitter (§58, §67)
grep -rniE "CORRECT|INCORRECT|CHOSEN|REJECTED|passes .* tests" RUN/preferences.jsonl  # none (§98, §99)
```

Then produce the §106 report in `08_PREFERENCE_PAIR_GENERATION.md` and **stop — do not start
Step 9 (DPO/QLoRA training) without explicit approval** (§106).

**The honest headline for that report (§92, §105):** the strict dataset is **3 distinct
training records drawn from 2 problems**, with validation empty. That is the correct output
for this input, not a bug — §105 says to optimize for high-confidence labels rather than pair
count, and §65 already states that ten problems cannot support meaningful DPO training. The
pipeline is what is being validated here; the dataset is not yet a dataset.

---

## Deviations to record in the report

- **The pair prompt is a canonical, strategy-free problem prompt** rather than the stored
  per-candidate generation prompt (decision 1). §41's literal check would yield zero pairs
  because all 50 candidates have distinct prompt hashes. Lineage is verified, not waived, and
  both generation-prompt hashes are kept in `metadata.jsonl`.
- **`minimum_score_margin` does not gate the strict policy** (decision 2) — §21 reads as a
  global filter, §25/§26 as a margin-policy instrument; the latter is adopted.
- **`preferences.jsonl` and the split files are deduplicated by `(prompt, chosen, rejected)`**
  while `metadata.jsonl` keeps every pair (decision 3), because 12 strict pairs are only 3
  distinct triples and §72 forbids duplicate records.
- **The split pool is the pair-bearing problems, not all problems** (decision 4), plus a
  floor rule guaranteeing a non-empty train split. Not in §64, which is silent on empty pools.
- **`rejections.jsonl` is persisted** although §102's tree omits it — required by §77 and by
  CLAUDE.md's data-integrity rule.
- **`preference_id` is `pref_<chosen_id>__<rejected_id>`**, dropping §31's leading problem id
  since every `candidate_id` already carries it.
- **`max_pairs_per_problem` truncation is a deterministic margin-ordered sort, with no RNG**,
  so §58's seed requirement does not arise; the manifest records the ordering rule instead.
- **Pair-level code deduplication never fires on the current data** — identical code implies
  identical score implies `TIE`, already excluded upstream. It is retained as a structural
  guard against non-deterministic evaluation, and stated as defensive rather than oversold.
- **The comparator is re-run rather than read from `comparisons.jsonl`** (§43, §44); the
  persisted comparisons become a validator cross-check.
- **A fourth copy of the run-directory plumbing**, rather than the shared base Stage 7
  deferred. Re-flagged as debt; extracting it would touch four stages at once.
- **A `preferences:` config section exists**, where Stage 7 added none — this stage has four
  genuinely tunable parameters and §22 forbids hard-coding the margin.
- **`preferences list` was added** beyond §78–§81, since none of the four specified commands
  is usable without a way to discover a `preference_run_id`.
- **Resume is implemented but has no practical value at this data scale** — required by §83
  and the §100 acceptance criteria.
