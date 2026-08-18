# src/python_dpo/preferences/

DPO preference pair generation — turns Stage 7's objective ranking (`CandidateAssessment`,
`ComparisonResult`) into `{prompt, chosen, rejected}` training data, with full audit
provenance, deduplication, splitting, statistics, and dataset validation.

This is the first package allowed to say **`chosen`**/**`rejected`** (Stage 7's spec
reserved that vocabulary for here). A pair exists only when the evidence is decisive: a
tie never produces a pair, an indeterminate candidate never participates, and identical
code never produces a pair against itself. No model is ever called — every label comes
from Stage 6's pytest counts by way of Stage 7's `CandidateComparator`, re-run here rather
than trusted from a persisted `ComparisonResult`.

## Files

### `errors.py`

`PreferenceError` and every preference-specific exception in one place —
`PreferenceConfigError`, `PreferencePolicyError`, `RankingRunNotFoundError` (the Stage 7
ranking run a preference generation is requested against does not exist — distinct from
`run_repository.py`'s `PreferenceRunNotFoundError`, about this package's own run
directories), `PreferenceStoreError`. A deliberate consolidation compared to Stage 7,
which splits its exceptions across `errors.py`/`repository.py`/`run_repository.py`.

### `models.py`

The schema, mirroring `python_dpo.ranking.models`: frozen dataclasses validating in
`__post_init__`, explicit `to_dict()`/`from_dict()` rejecting unknown/missing fields.

`PreferencePair` — the DPO record. `prompt`/`chosen`/`rejected` are exactly a training
JSONL row; everything else (candidate ids, scores, provenance hashes, strategies) is audit
metadata. `chosen`/`rejected` are the candidates' code verbatim — never reformatted,
repaired, or wrapped in fences. `prompt` is the *canonical*, strategy-free problem prompt
(see `prompt.py`), not either candidate's own generation-time prompt. `__post_init__`
enforces pair validity as a construction-time invariant: different candidates, different
code, `chosen_score > rejected_score`, `score_margin` arithmetic, `preference_strength`
agreeing with the correctness pair (`strong` iff correct-vs-incorrect, never `weak`).

`PreferenceRejection` — one candidate pair that did not become a preference, with a
specific `reason` from a closed set (`tie`, `indeterminate`, `identical_code`,
`insufficient_margin`, `not_correct_vs_incorrect`, `invalid_prompt_match`,
`integrity_failure`, `max_pairs_per_problem`). Every pair the builder considers becomes
either a `PreferencePair` or one of these — nothing is silently dropped.

`PreferenceManifest`/`PreferenceStatistics` mirror `RankingManifest`/`RankingStatistics`:
`with_status()` enforces the same closed transition graph, and
`PreferenceStatistics.from_records(...)` is always reconstructable from
`metadata.jsonl`/`rejections.jsonl`, never trusted from an in-memory counter. The five
headline rejection counters (`ties`, `duplicates`, `indeterminate`, `prompt_mismatches`,
`integrity_failures`) are validated to equal the matching entries of
`rejections_by_reason`, so the two views can never silently disagree.

`QualityReport` — score-margin/pass-rate/strategy distributions and a reason per
pairless problem. Reported, never enforced: the ten-problem dataset is far too small for
a statistically meaningful gate.

### `prompt.py`

`verify_prompt_lineage(problem, candidates)` — the resolution to the stage's central
tension: every candidate of a problem was generated under a *different*, strategy-specific
prompt (see `generation/prompt_builder.py`), so no two candidates share a
`prompt_sha256`, and spec section 41's literal "chosen and rejected prompts must match"
check would produce zero pairs under every policy. This function proves the canonical,
strategy-free prompt is a genuine rendering of the same template every candidate was
actually generated under — by re-deriving each candidate's stored prompt hash from
`build_prompt(problem, candidate.strategy)` and requiring an exact match — before it is
ever used as the pair's `prompt`. A failure is recorded as an `integrity_failure`, never a
silent fallback to an unverified prompt.

### `policies.py`

Three pure, stateless policies, each returning `(admitted, rejection_reason)` rather than
a bare boolean:

| Policy | Admits |
|---|---|
| `StrictPolicy` | `correct` vs `incorrect` only. **Ignores** `minimum_score_margin` entirely — correctness alone is the signal. |
| `MarginPolicy` | `chosen_score > rejected_score` and the margin clears `minimum_score_margin` (default `0.2`). |
| `AllBetterPolicy` | Any decisive comparison, however small the margin. Never the default. |

The universal exclusions — ties, indeterminate candidates, cross-problem pairs, identical
code, invalid prompt provenance, integrity failures — live in `builder.py`, not here, so
no policy can accidentally omit one.

### `dedup.py`

Three deliberately separate deduplication notions:

| Function | Identity | Purpose |
|---|---|---|
| `pair_key` | `(problem_id, chosen_id, rejected_id)` | Directional — `A>B` and `B>A` are different keys, not duplicates of each other |
| `code_identical` | Candidate-level `code_sha256` equality | Gates a *single pair*; never removes a candidate from the pool — it may still pair against a third candidate |
| `training_key` | `(prompt, chosen, rejected)` text triple | The identity a DPO trainer actually sees — the one `dedupe_training_records` collapses on |

`dedupe_training_records(pairs)` marks every pair beyond the first (by
`preference_id` order) occurrence of its `training_key` as `duplicate_training_record`,
pointing at the survivor's id. Nothing is dropped from `metadata.jsonl`; only
`preferences.jsonl` and the split files are filtered down to survivors.

### `builder.py`

`PreferencePairBuilder.build_problem(...)` — the per-problem pipeline: verify prompt
lineage once, then for every unordered candidate pair `C(n, 2)`, re-run Stage 7's
`CandidateComparator`, exclude ties/indeterminate/identical-code/integrity failures, apply
the configured policy, deterministically truncate to `max_pairs_per_problem` (largest
score margin first, no RNG anywhere), and dedupe training records within the problem.
`build_run(...)` loops this over every problem, strictly grouped — a pair can never cross
a problem boundary, because pairing only ever happens *within* one `build_problem` call.

### `splitter.py`

`ProblemSplitter.split(problem_ids)` — deterministic, problem-level train/validation/test
splitting. The pool is deliberately the *pair-bearing* problems (passed in by the caller),
not the entire dataset — splitting all ten problems when only two produce pairs would
spend the validation/test budget on problems contributing nothing. A floor rule keeps
`train` non-empty whenever the pool is non-empty. `random.Random(seed).shuffle(...)` over
a **sorted** pool is the only randomness anywhere in this package — the input's own
ordering can never leak into the result.

### `repository.py` / `run_repository.py`

`PreferenceRepository` — run-scoped persistence. `metadata.jsonl` is both the durable,
append-as-computed ledger (mirroring `ranking/repository.py`'s never-rewrite append
pattern) *and* the full audit artifact; `rejections.jsonl` is the same shape for excluded
pairs. `write_dataset(split_manifest)` (re)writes the three-key training files
(`preferences.jsonl`, `train.jsonl`, `validation.jsonl`, `test.jsonl`) as whole-file atomic
replacements from what is already durably in `metadata.jsonl` — never a partial append,
since these must always reflect a fully-built state.

`PreferenceRunRepository` — the multi-run manager, mirroring
`ranking.run_repository.RankingRunRepository`: mints `pref_YYYYMMDD_HHMMSS_xxxx` ids, owns
`manifest.json`/`statistics.json`/`quality_report.json`/`split_manifest.json`, and the run
status lifecycle. A fourth, independent copy of this plumbing rather than a shared base
extracted across Stages 4, 6, 7, and 8 — the same deliberate deferral Stage 7 made, kept
here to contain this stage's blast radius.

### `statistics.py`

Text formatters only — `PreferenceStatistics`/`QualityReport` themselves live in
`models.py`. `format_preference_statistics`, `format_quality_report`, `format_pair_table`,
`format_pair_detail`.

### `validation.py`

Mirrors `python_dpo.ranking.validation`: `PreferenceValidationIssue(check, message)`,
`PreferenceValidationReport`, every issue fatal, every check runs to completion. Internal
self-consistency comes free from `PreferencePair.__post_init__`; the module's real job is
the **cross-artifact** checks — reloading the candidate run to confirm `chosen`/`rejected`
still match a real candidate's stored code, reloading the ranking run to re-derive each
pair's scores and re-run the comparator to confirm the claimed direction still holds, and
re-applying the recorded policy to confirm the pair is still admitted. Also checks: no
duplicate `preference_id` in `metadata.jsonl`, no reverse pairs (`A>B` and `B>A` both
present), no duplicate `(prompt, chosen, rejected)` triple in `preferences.jsonl`, no
problem in more than one split (structural, via `SplitManifest`'s own construction), every
split member has training pairs, the split reproduces from its own seed and ratios, and
`statistics.json` recomputed and compared `==`.

## Persistence layout

```
data/preferences/runs/pref_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json        # policy, versions, margin, split config, upstream run ids, status
├── metadata.jsonl       # one PreferencePair per generated pair, including collapsed ones
├── rejections.jsonl     # one PreferenceRejection per excluded candidate pair
├── preferences.jsonl    # {prompt, chosen, rejected} training records, deduped
├── split_manifest.json  # train/validation/test problem-id membership, seed, ratios
├── train.jsonl          # same three-key shape as preferences.jsonl
├── validation.jsonl
├── test.jsonl
├── statistics.json      # reconstructable from metadata.jsonl + rejections.jsonl
└── quality_report.json  # distributions and per-problem reasons; reported, not enforced
```

Every record in `metadata.jsonl`/`rejections.jsonl` carries `ranking_run_id`; pairs
additionally carry `candidate_run_id` and `evaluation_run_id`, so the full chain back to
model and prompt stays traceable. Historical preference runs are immutable — a
re-generation with `--force`, or a different policy/margin, is a new run, never an
overwrite, so `strict_v1`/`margin_v1`/`margin_v2` can coexist without rerunning Qwen,
Docker, or pytest.
