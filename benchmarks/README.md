# benchmarks/

Held-out evaluation benchmarks for Stage 10 (`.claude/specs/10_model_evaluation.md`).

Each subdirectory is one benchmark, named by its `benchmark_version`:

```
benchmarks/<name>/manifest.json
```

The manifest holds `problem_ids`, a SHA-256 `dataset_hash` over their canonical content,
and `source_dataset_version` — never a copy of the problems themselves.
`data/problems/problems.jsonl` stays the single source of truth; `benchmark validate`
recomputes the hash from the live dataset and fails loudly on drift (spec section 10),
which is what makes "the benchmark must not change between model comparisons" (spec
section 91) a real guarantee rather than a stated intention.

## `python_eval_v1`

The 7 problems (`p001, p002, p003, p004, p005, p006, p009`) never assigned to any Stage 8
preference split for `pref_20260818_074347_5eff` — Stage 8's problem-level splitter only
assigns *pair-bearing* problems to train/validation/test, so the six problems that
produced no preference pairs (all their candidates tied) are equally untrained and equally
valid as held-out data. `p007`/`p008` (train) and `p010` (validation) are excluded.

**Ceiling effect, reported rather than hidden:** from the committed Stage 6/7 evidence, 5
of these 7 problems (`p001, p003, p005, p006, p009`) are already solved by every one of
the base model's 5 candidates. Only `p002` (0/5) and `p004` (3/5) have real headroom. This
means the benchmark can mostly only detect *regression*, not improvement — selecting only
the 2 problems with headroom was rejected as benchmark contamination (spec sections 134,
135): a benchmark chosen by inspecting base-model results is no longer held out.

Rebuild or re-check it with:

```bash
python -m python_dpo benchmark build --name python_eval_v1 \
    --exclude-preference-run-id pref_20260818_074347_5eff
python -m python_dpo benchmark validate --benchmark python_eval_v1
python -m python_dpo benchmark check-leakage --benchmark python_eval_v1 \
    --preference-run-id pref_20260818_074347_5eff
```

Per spec section 144: 7 problems is suitable for **pipeline validation**, not for reliable
model-performance conclusions. Spec section 145 recommends hundreds to 500-1,000+ problems
for serious experimentation.
