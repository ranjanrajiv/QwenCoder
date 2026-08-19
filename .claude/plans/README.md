# .claude/plans/

Approved implementation plans, tracked in git despite the rest of `.claude/` being
ignored (see `.gitignore`'s `.claude/*` + `!.claude/plans/` rule).

## Files

### `01_project_skeleton_plan.md`

The concrete implementation plan for Step 1, derived from
[`.claude/specs/01_project_skeleton.md`](../specs/01_project_skeleton.md) and
confirmed with the user before implementation started. It records the three
decisions made beyond what the spec itself dictates (skeleton at the repo root
rather than a nested `python-dpo/` directory; PyYAML as the sole runtime
dependency; `.claude/specs/` and `.claude/plans/` becoming tracked while the rest of
`.claude/` stays ignored), an exact file-by-file list of what to create and why,
the verification commands to run afterward, and the deviations from the spec's
literal file tree to flag in the final report (`__main__.py` and `config.py`
weren't in the spec's tree but are required for `python -m python_dpo` and the
config abstraction it calls for). This plan has been fully executed — see the root
[`README.md`](../../README.md) for current project status.

### `02_problem_dataset_plan.md`

The implementation plan for Stage 2 — the problem dataset and ground-truth layer —
derived from [`.claude/specs/02_problem_dataset.md`](../specs/02_problem_dataset.md)
and confirmed with the user before implementation started. It covers the new
`src/python_dpo/problems/` subpackage (schema, catalog, reference solutions,
JSONL storage, a swappable `ReferenceExecutor`, and dataset validation), the
`problems build` / `problems validate` CLI commands, and the unit plus integration
test suites. It pins down the three approved design decisions (frozen dataclasses
rather than Pydantic, reference solutions authored as real Python functions with
their JSONL text derived via `inspect.getsource()`, and the validation summary
printed to stdout from the CLI layer), plus the semantics chosen for each of the ten
problems where the spec required an explicit ruling on ties, ordering, and
invalid-input behavior. **Approved but not yet implemented.**

### `04_candidate_persistence_plan.md`

The implementation plan for Stage 4 — candidate persistence, runs and reproducibility —
derived from
[`.claude/specs/04_candidate_presistence.md`](../specs/04_candidate_presistence.md) and
confirmed with the user before implementation started. It turns Stage 3's flat
append-only `candidates.jsonl` into per-run artifact directories
(`data/candidates/runs/<run_id>/` with a manifest, candidates, failures, statistics and
a prompts artifact), adds SHA-256 hashes for code, prompt and raw output, a
`schema_version` on candidate records, atomic/durable persistence with torn-tail
detection, a retry policy for infrastructure failures, statistics reconstructable from
disk, a run integrity validator, and the `runs` / `candidates` CLI command groups. It
records the three approved design decisions (`generate` always mints a new run with
`--resume RUN_ID` as the only resume path; the existing 50-record flat file is migrated
into a run directory by an explicit `candidates migrate` command rather than discarded;
run IDs adopt the spec's `run_YYYYMMDD_HHMMSS_xxxx` format), and lists the deviations to
flag in the final report — chiefly the run-scoped repository API, run-scoped duplicate
detection, and the decision not to persist tracebacks. This plan has been fully executed —
see [`04_CANDIDATE_PERSISTENCE.md`](../../04_CANDIDATE_PERSISTENCE.md) for the
implementation report.

### `05_docker_sandbox.md`

The implementation plan for Stage 5 — the isolated Docker sandbox — derived from
[`.claude/specs/05_docker_sandbox.md`](../specs/05_docker_sandbox.md) and confirmed with
the user before implementation started. It builds the execution boundary every previous
stage deliberately stopped short of: a new `src/python_dpo/sandbox/` package whose
`SandboxExecutor` runs arbitrary Python inside a locked-down container (no network, no
host filesystem, non-root, capabilities dropped, with CPU/memory/PID/output/time limits)
and returns a structured `ExecutionResult` that reports *what happened* without ever
judging correctness. It records the three approved design decisions (Docker CLI subprocess
with a fixed argv rather than the Python SDK, so zero dependencies are added and the whole
isolation posture is one auditable list; stock `python:3.12-slim` with `--user
65534:65534` rather than a custom Dockerfile; integration tests deselected by default so
`pytest -q` keeps its zero-skip, Docker-free property), and lists the deviations to flag
in the final report — chiefly the streaming-oriented `ContainerRuntime` shape, workspaces
in system temp rather than `data/sandbox/jobs/`, and the hardening added beyond the spec's
literal text (`--memory-swap`, `--security-opt no-new-privileges`). This plan has been
fully executed — see [`05_DOCKER_SANDBOX.md`](../../05_DOCKER_SANDBOX.md) for the
implementation report.

### `06_candidate_test_executor_plan.md`

The implementation plan for Stage 6 — the candidate test executor — derived from
[`.claude/specs/06_candidate_test_executor.md`](../specs/06_candidate_test_executor.md)
and confirmed with the user before implementation started. It is what the Stage 5 sandbox
was built for: a new
`src/python_dpo/evaluation/` package that turns a problem's declared test cases into a
deterministic pytest suite, runs it against a persisted candidate inside the sandbox, and
persists per-candidate and per-test execution evidence — counts, statuses, durations,
error types — without ever deciding `correct`/`incorrect` or producing preference pairs.

It records the three approved design decisions (results returned as nonce-prefixed JSON
lines from a generated `conftest.py`, so no dependency and no report-file extraction from
a read-only container; resume-by-default per the spec, deliberately diverging from Stage
4's explicit-resume `generate`; and verification evaluating all 50 real Qwen candidates
with the artifacts committed). It also captures the dataset findings that drive the
generator — `TestCase.input` is a kwargs mapping rather than the positional list the spec's
examples imply, p010 is `async`, p005/p006/p009 use `expected_exception`, p009 returns a
generator, and no float expected values exist — plus the reference-solution self-check that
proves the generator's comparison semantics still match the ones the dataset was validated
under. This plan has been fully executed — see
[`06_CANDIDATE_TEST_EXECUTOR.md`](../../06_CANDIDATE_TEST_EXECUTOR.md) for the
implementation report.

### `07_candidate_ranking_plan.md`

The implementation plan for Stage 7 — candidate evaluation, scoring and ranking — derived
from [`.claude/specs/07_candidate_ranking.md`](../specs/07_candidate_ranking.md) and
confirmed with the user before implementation started. It turns Stage 6's objective
execution evidence into a judgement: a new `src/python_dpo/ranking/` package that
classifies each candidate `correct`/`incorrect`/`indeterminate`, scores it as
`tests_passed / tests_total`, ranks candidates independently **per problem** with
competition ranking and explicit tie groups, and exposes pairwise comparisons with score
margins — while producing no `chosen`/`rejected` labels, no DPO pairs, and calling no LLM
judge.

It records the four approved design decisions (assessments join back to the generation run
so `code_sha256`/`duplicate_of` reach Step 8; `comparisons.jsonl` is persisted; a candidate
with no evaluation result becomes an `indeterminate` assessment with an explicit reason
rather than being skipped; and the run-directory plumbing is mirrored a third time rather
than extracted into a shared base). Its most consequential finding is measured from the
committed evaluation run: **6 of the 10 problems yield no ordering at all** — five have all
five candidates fully passing and one has all five failing identically — so 78 of the 100
candidate pairs are ties, and 31 of 50 candidates are duplicate code. Tie handling is
therefore the majority behaviour rather than an edge case, and the `indeterminate` path has
zero real coverage and must be driven by synthetic fixtures. The plan carries exact
expected acceptance numbers for the real run. **Approved but not yet implemented.**

### `08_preference_pair_generation_plan.md`

The implementation plan for Stage 8 — DPO preference pair generation — derived from
[`.claude/specs/08_preference_pair_generation.md`](../specs/08_preference_pair_generation.md)
and confirmed with the user before implementation started. It turns Stage 7's neutral
`A_BETTER`/`TIE` orderings into `{prompt, chosen, rejected}` training records: a new
`src/python_dpo/preferences/` package with three selection policies (`strict`, `margin`,
`all_better`), a configurable minimum score margin, three separate deduplication notions,
problem-level train/validation/test splitting, a dataset validator, and full audit
provenance — while calling no model of any kind and never altering a byte of the candidate
code.

Its blocking finding is measured from the committed candidate run: **all 50 candidates have
distinct `prompt_sha256`**, because the generation prompt embeds the per-candidate strategy,
so §41's literal prompt-equality check would yield **zero pairs under every policy**. The
four approved decisions follow from that and from the rest of the real data (a canonical,
strategy-free problem prompt whose lineage is *verified* against every candidate's stored
hash rather than waived; `minimum_score_margin` gating the margin policy but not strict, so
p008's 9/9-vs-8/9 correctness gap survives; the training JSONL deduplicated by
`(prompt, chosen, rejected)` while `metadata.jsonl` keeps every pair; and the split pool
being the pair-bearing problems with a floor rule keeping train non-empty). The second
consequential finding is scale: 78 of the 100 comparisons are ties, strict yields 12 pairs
across just 2 problems, and those 12 collapse to **3 distinct training records**. The plan
carries exact expected acceptance numbers per policy and states plainly that the pipeline —
not the dataset — is what this stage validates. This plan has been fully executed — see
[`08_PREFERENCE_PAIR_GENERATION.md`](../../08_PREFERENCE_PAIR_GENERATION.md) for the
implementation report.

### `09_dpo_qlora_training_plan.md`

The implementation plan for Stage 9 — Qwen Coder DPO/QLoRA training — derived from
[`.claude/specs/09_dpo_qlora_training.md`](../specs/09_dpo_qlora_training.md) and confirmed
with the user before implementation started. It fine-tunes a LoRA adapter over a frozen,
4-bit NF4-quantized Qwen Coder base using TRL's `DPOTrainer`: a new
`src/python_dpo/training/` package covering hardware and package-version capture, dataset
validation and hashing, token-length and truncation analysis, quantization and LoRA
application with parameter-count safety checks, metrics persistence, run manifests, and the
mandatory adapter-reload verification — while executing no candidate code and making no
claim about Python ability, which belongs to Step 10.

Its blocking finding is measured from the committed Stage 8 output: **every preference
dataset has an empty `validation.jsonl`** (strict train=1/val=0/test=2, margin
train=2/val=0/test=2), because the splitter floors validation at `floor(n × 0.1)` and only
2–4 problems ever produce pairs — so §24.9's "train and validation must be non-empty" would
fail preflight on every dataset that exists. The four approved decisions follow (mint an
`all_better` dataset at 0.5/0.25/0.25 ratios, the only reachable non-empty split at
train=3/val=2/test=2, *and* tolerate an empty validation behind `--allow-small-dataset`;
run the spec's full six-step verification including a real training run; commit provenance
and the ~15 MB adapter while gitignoring checkpoints; and apply the Qwen chat template
during training because Stage 3 generation demonstrably did). Hardware and dependencies
were verified up front: an RTX 3060 with ~11.7 GiB free, the base model already cached, and
trl 1.10 / peft 0.20 / bitsandbytes 0.50 / datasets 5.0 resolving without downgrading torch
or transformers. The plan states plainly that three training records is under one
gradient-accumulation cycle — this stage validates the QLoRA/DPO stack, not the model.
This plan has been fully executed — see
[`09_DPO_QLORA_TRAINING.md`](../../09_DPO_QLORA_TRAINING.md) for the implementation
report, including the two TRL 1.10 API changes (`max_prompt_length` and `warmup_ratio`
no longer exist on `DPOConfig`) that the plan flagged as its biggest unknown.

### `10_model_evaluation_plan.md`

The implementation plan for Stage 10 — base vs DPO model evaluation — derived from
[`.claude/specs/10_model_evaluation.md`](../specs/10_model_evaluation.md) and confirmed
with the user before implementation started. It is where the question every prior stage
deferred finally gets asked: a new `src/python_dpo/model_evaluation/` package that runs
base Qwen and base + LoRA adapter over the same held-out problems with the same prompts,
seeds and sandbox, then measures the difference with a correct pass@k estimator, paired
bootstrap confidence intervals, win/tie/loss and McNemar — reusing the Stage 5 sandbox and
Stage 6 executor for all execution, and calling no LLM judge anywhere.

Its two consequential findings are measured from the committed artifacts. Stage 9 trained
on only three problems, so the leakage-clean held-out set is **seven** problems rather than
the single formally designated test problem — but **five of those seven are already solved
perfectly by the base model** (5/5 candidates passing every test), leaving real headroom on
only `p002` and `p004`. The benchmark can therefore mostly detect regression, not
improvement, and the plan makes that a first-class headroom analysis rather than burying
it; selecting only the two problems with headroom was explicitly rejected as benchmark
contamination. Combined with a Stage 9 adapter that trained for one optimizer step, the
expected verdict is `DPO_SUCCESS = false` with a paired CI straddling zero — and the plan
declines to adjust the success thresholds to produce a pass.

It also records the three approved decisions (all seven held-out problems with the ceiling
reported; the first full run treated as pipeline validation reported honestly; and a
benchmark manifest that references problem ids and hashes their content rather than
snapshotting them), the reuse survey that shows `CandidateEvaluator` accepts an in-memory
`Candidate` so Stage 6's execution *and classification* can be reused without Stage 4's run
plumbing, and the finding that a new inference layer is unavoidable because
`QwenModelClient` supports seeding but not adapters or quantization while
`training/verify.py` supports adapters but is greedy-only and unseeded. pass@k, bootstrap
and McNemar are hand-rolled in pure stdlib, adding no dependencies.
**Approved but not yet implemented.**

### `11_error_analysis_and_iteration_plan.md`

The implementation plan for Stage 11 — error analysis, preference refinement and iterative
improvement — derived from
[`.claude/specs/11_error_analysis_and_iteration.md`](../specs/11_error_analysis_and_iteration.md)
and confirmed with the user before implementation started. Stage 10 answered *"did DPO make
Qwen better at Python?"* with a number; Stage 11 asks what to change next. A new
`src/python_dpo/analysis/` package classifies every failure against a deterministic
taxonomy, sorts problems into improvements and regressions, computes test-level failure
frequencies, diversity and category/difficulty coverage gaps, and emits an evidence-backed
recommendation set plus a refined preference dataset — with **no LLM judge anywhere** (§12,
§295 forbid it as the primary classifier), correlation never stated as causation (§38, a
wording rule enforced in `report.py` and asserted in tests), and no automatic retraining
(§5, §113): the stage emits `next_experiment.yaml` and stops. It is the first stage in the
back half of the pipeline that is **pure computation over persisted artifacts** — no model,
no GPU, no Docker, no new dependencies — so the whole thing runs in the default offline
suite with no new markers.

Its blocking finding is that the real analysis is degenerate before it starts:
`eval_20260818_155511_1633` has **0 DPO wins, 0 losses, 7 ties**, both `improvements.jsonl`
and `regressions.jsonl` are 0 bytes, and `failure_analysis.json` has exactly one non-zero
bucket. The honest iteration decision is therefore `insufficient_evidence`, and the plan has
it *gate* the others rather than reporting the better-supported `refine_data` as a headline
over a 7-problem benchmark with a 4.3pp-wide CI. The one genuinely informative finding is
structural rather than statistical: only `p007`/`p008` were ever trained on (categories
`edge_cases`, `exceptions`), and the benchmark's seven categories share **zero** overlap with
them, so every benchmark category scores `coverage_ratio = 0.00`. Two further findings shape
the schema — `coverage_ratio` has degenerate cases JSON cannot represent (`inf` when a
category is absent from the benchmark, `nan` when absent from both), so it is `float | None`
beside an explicit five-value verdict enum; and the hierarchical taxonomy is reachable only
through `evaluations/_sandbox/<variant>/test_results.jsonl`, which the already-public
`sandbox_repository()` exposes, so §45–§49 need no change to Stage 10.

It records three approved decisions (defer every explicitly-optional capability — LLM
semantic analysis, `HardProblemGenerator`, and flaky-test detection, the last because it
would turn a pure-computation stage into a Docker-dependent one for a property this dataset
cannot exhibit; emit both a re-versioned `refined_preferences.jsonl` carrying
`parent_preference_run_id` and the three example datasets, never overwriting Stage 8; and
analyse the real run reporting `insufficient_evidence` rather than committing a fabricated
one beside it, with the improvement, regression, mode-collapse and non-degenerate coverage
paths exercised by synthetic fixtures instead). It also claims `analysis` as the **tenth**
`paths` entry — a wiring point Stage 12's plan shares, since that plan claims `experiments`
as the same kind of addition. **Approved but not yet implemented.**

### `12_pipeline_orchestration_and_productionization_plan.md`

The implementation plan for Stage 12 — end-to-end pipeline orchestration, model packaging
and productionization — derived from
[`.claude/specs/12_pipeline_orchestration_and_productionization.md`](../specs/12_pipeline_orchestration_and_productionization.md)
and confirmed with the user before implementation started. It turns eleven stage-shaped CLI
command groups into a single reproducible experiment: a new `src/python_dpo/pipeline/`
package with a `PipelineOrchestrator`, a nine-stage dependency graph, an immutable resolved
configuration, per-stage state and manifests, a derived cache with automatic invalidation
cascade, resume/retry/force/dry-run, signal handling, artifact hashing and lineage, plus a
new `src/python_dpo/packaging/` package that packages the LoRA adapter, verifies it by
generating Python and executing it through the Stage 5 sandbox, and registers it in a local
model registry that never promotes automatically.

Its blocking finding is that **Stage 11 does not exist** — `src/python_dpo/analysis/` is
absent and only its plan is committed — so `error_analysis` ships as a registered but
disabled stage whose adapter fails loudly if enabled and whose state is persisted as
`SKIPPED` with a reason, rather than being silently omitted. Two further findings reshape
the spec's literal text: the spec's nine stages do not map 1:1 onto the repo's commands
(Stage 6 executes *and* runs pytest, Stage 7 judges, so `candidate_execution` → Stage 6 and
`candidate_evaluation` → Stage 7), and `problem_generation` is not generation at all — Stage
2's catalog is ten hand-authored problems, so the spec's `problem_count: 1000` is
unimplementable and `problem_count` can only select a subset.

It records the four approved decisions (Stage 11 registered-but-disabled; stage artifacts
staying in their canonical `data/<stage>/runs/` stores with the experiment directory holding
manifests, SHA-256 pointers, logs, reports and the packaged model rather than duplicating
gigabytes; the full spec executed in four phases with explicitly-optional capabilities
deferred with reasoning; and two smoke tiers — an offline mock-model pipeline test in the
default suite plus a real GPU/Docker `--smoke-test` whose artifacts are committed). It also
resolves that the stage bodies currently buried in `cli.py`'s private helpers **move** into
`pipeline/stages/` so the orchestrator and the CLI share one implementation rather than the
orchestrator shelling out or faking an `argparse.Namespace`, and that the git SHA is recorded
in the manifest but deliberately excluded from the cache key, since including it would
invalidate every stage on every commit and defeat the spec's own cache-invalidation
requirement. **Approved but not yet implemented.**
