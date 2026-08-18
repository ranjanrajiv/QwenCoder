# Stage 9 — Qwen Coder DPO/QLoRA Training

## Context

Stage 8 produced the first artifact that can legitimately be called DPO training data:
`data/preferences/runs/pref_*/train.jsonl` etc., `{prompt, chosen, rejected}` triples
backed by execution evidence. Nothing in the repository has ever loaded a model for
*training* — Stage 3 loads Qwen only for inference, and `tests/test_no_heavy_imports.py`
asserts that importing the package never pulls in `torch`.

`.claude/specs/09_dpo_qlora_training.md` asks for the layer that fine-tunes a Qwen Coder
model on that preference data: 4-bit NF4 quantization, a LoRA adapter over a frozen base,
TRL's `DPOTrainer`, and the whole provenance/preflight/verification apparatus around it.

Three boundaries define the stage:

- **The base model stays frozen (§3, §20).** Only LoRA parameters train. If
  `trainable_parameters == total_parameters` the run **fails** rather than silently
  performing full fine-tuning.
- **No candidate code is ever executed (§100, §101).** `chosen`/`rejected` are read as
  text and tokenized. Execution belongs to Stages 5/6 alone — this is a CLAUDE.md
  security rule, not merely a spec line.
- **No performance claim (§95, §110).** A falling DPO loss is not evidence of better
  Python. Stage 9 produces `base model + adapter`; Stage 10 evaluates it.

**Outcome:** `train dpo --config configs/training/dpo_qlora.yaml --preference-run-id PREF`
produces `data/training/runs/dpo_.../` with a manifest, hardware/dataset provenance,
metrics, a LoRA adapter, and a passing adapter-reload check.

---

### What exploration established

Measured on this machine, not assumed. Several findings are load-bearing.

| Finding | Consequence |
|---|---|
| **Every committed Stage 8 dataset has an EMPTY `validation.jsonl`** — strict: train=1/val=0/test=2; margin: train=2/val=0/test=2 | §24.9 ("verify train and validation are non-empty") would **fail preflight on every dataset that exists**. The blocking finding; decision 1 resolves it |
| Stage 8's splitter floors validation at `floor(n_problems × 0.1)`, which is **0 for any pool under 10 problems** — and only 2–4 problems ever produce pairs | No seed change helps. The *only* reachable non-empty validation is `all_better` + ratios `0.5/0.25/0.25` → **train=3, val=2, test=2 records** over 4 problems (2/1/1) |
| **GPU present**: one RTX 3060, 12288 MiB, ~495 MiB already held by Xorg/gnome-shell/firefox/vscode → **~11.7 GiB usable**. Driver 580.82.07, CUDA 13.0 | §11's 12 GB budget is real but the desktop's ~0.5 GB is unrecoverable — `hardware-check` must report **free** VRAM (`torch.cuda.mem_get_info()`), not total |
| **Installed**: `torch 2.13.0+cu130` (CUDA available: True), `transformers 5.15.0`, `accelerate 1.14.0`, `safetensors 0.8.0`. **Missing**: trl, peft, bitsandbytes, datasets | A `training` extra is needed |
| `pip install --dry-run trl peft bitsandbytes datasets` resolves **cleanly**: `trl 1.10.0`, `peft 0.20.0`, `bitsandbytes 0.50.1`, `datasets 5.0.1` — and **downgrades nothing** (torch and transformers stay put) | The stack is installable as-is. But TRL **1.x** has a different API surface from the 0.x releases most documentation describes — the `DPOTrainer`/`DPOConfig` signature must be verified against the installed version at implementation time, not assumed |
| **`Qwen/Qwen2.5-Coder-3B-Instruct` is already cached** (5.8 GB in `~/.cache/huggingface`); 105 GB disk free | No multi-GB download; smoke and real runs can proceed immediately |
| Architecture: `Qwen2ForCausalLM`, hidden 2048, **36 layers**, 16 heads, **2 KV heads (GQA)** — so `k_proj`/`v_proj` are 2048→256, not 2048→2048 | LoRA r=16 over q/k/v/o = **7,372,800 params ≈ 7.4M**, ~0.24% of ~3.09B, ~15 MB in bf16. §19's "substantially smaller" is satisfied by two orders of magnitude |
| **Stage 3 generation applied Qwen's chat template** — `qwen.py:_render` wraps every prompt as `[{"role":"user","content":prompt}]` with `add_generation_prompt=True` when the tokenizer declares a template | §30's consistency requirement is concrete: training **must** apply the same template, or the model sees a prompt format it was never generated under. Decision 4 |
| `ModelConfig.quantization` **actively rejects any non-null value** ("not implemented in Stage 3") | Stage 9 must not reuse `config.model` for quantization. The spec's own separate `configs/training/dpo_qlora.yaml` (§62) sidesteps this entirely |
| `_REQUIRED_PATH_KEYS` has **no `training` entry**, and `data/training/` does not exist | Unlike Stage 8, an **eighth data path** must be threaded through `config.py`, `config.yaml`, `data/`, and the **three literal enumerations** in `tests/test_project.py` (lines ~34, ~62, ~85) |
| `tests/test_no_heavy_imports.py` subprocess-imports `python_dpo.cli` and asserts `torch`/`transformers`/`accelerate` are absent from `sys.modules` | Any `training` module reachable from `cli.py` must use the `qwen.py` `_import_backend()` deferred-import idiom. The guard must also **grow** to cover `trl`/`peft`/`bitsandbytes`/`datasets` and the new package |
| `sandbox/health.py` already establishes the exact report shape §13 wants: frozen `HealthCheck(name, passed, detail)` / `HealthReport`, short-circuit on first failure, every failure detail ending in a remediation command, a separate `format_*_report()` | `hardware-check` mirrors it rather than inventing a shape |
| `evaluation/probe.py` establishes "record what genuinely ran, not what the config asked for" | §71's package-version capture is the same principle via `importlib.metadata`, tolerating a missing optional package as `None` |
| `preferences generate` exposes `--policy/--margin/--max-pairs-per-problem/--split-seed` but **no `--split-ratios`** | Minting the decision-1 dataset would otherwise require hand-editing `config.yaml`. A small backward-compatible flag is added — see "Modifications" |
| `.gitignore` excludes only `data/raw/*`; everything else under `data/` is tracked | Training checkpoints are a new size class and need an explicit exclusion (decision 3) |

---

### Decisions confirmed with the user

1. **Regenerate a validation-bearing dataset AND degrade gracefully.** Mint a third
   Stage 8 run — `all_better`, ratios `0.5/0.25/0.25` — giving the only reachable
   non-empty split (train=3, val=2, test=2 records). *Also* make Stage 9 tolerate an
   empty validation behind `--allow-small-dataset`, running with `eval_strategy="no"` and
   warning loudly, so the already-committed strict/margin datasets stay trainable
   (§96 wants several datasets trainable without code changes).

2. **Run the spec's full §108 verification**, all six steps: `hardware-check` → dataset
   validate → `--dry-run` → `--smoke-test` → a real training run → `train verify`
   (the mandatory §74 adapter reload). At three training records the "full" run is a
   handful of optimizer steps — pipeline validation, not model adaptation, and the report
   says so plainly.

3. **Commit provenance + the final adapter; gitignore the weights-heavy rest.**
   `manifest.json`, `config.yaml`, `dataset_manifest.json`, `hardware.json`, `metrics/`,
   `logs/`, `final_report.json`, and `adapter/` (~15 MB) are tracked;
   `checkpoints/` and `tokenizer/` are ignored. §58's "preserve all checkpoints" still
   holds on disk — just not in git history.

4. **Training applies the Qwen chat template** (implied by §30/§31 plus the Stage 3
   evidence above), converting the model-agnostic string prompt into conversational form
   so TRL applies the identical template the candidates were generated under.
   Configurable, defaulting to on, and recorded in the manifest.

---

## New package — `src/python_dpo/training/`

House style throughout, copied from `preferences/`/`ranking/`: frozen dataclasses
validating in `__post_init__`, explicit `to_dict()`/`from_dict()` rejecting unknown and
missing fields, plain-string closed sets, two-tier repositories over `atomic_io`, a
per-package `README.md`.

**The overriding architectural constraint is the lazy-import discipline.** Every module
that touches torch/transformers/trl/peft/bitsandbytes/datasets does so **inside
functions**, via the `qwen.py` idiom:

```python
_INSTALL_HINT = "install the training backend with: pip install -e '.[training]'"

def _import_backend():
    try:
        import bitsandbytes, peft, torch, transformers, trl
    except ImportError as exc:
        raise TrainingDependencyError(f"{exc}; {_INSTALL_HINT}") from exc
    return ...
```

`training/__init__.py` re-exports only names whose modules are import-cheap.

**`errors.py`** — `TrainingError` base; `TrainingConfigError`, `TrainingDependencyError`,
`HardwareError`, `DatasetValidationError`, `TargetModuleError`, `FullFineTuneError`
(§20's safety trip), `TruncationThresholdError`, `TrainingRunNotFoundError`,
`TrainingStoreError`, `AdapterReloadError`, `CheckpointCompatibilityError`.

**`config.py`** — the §62 experiment schema, loaded from a **standalone YAML file**, not
from root `config.yaml`. Frozen dataclasses: `ModelSpec` (name, revision),
`QuantizationSettings` (enabled, bits, quant_type, double_quant, compute_dtype),
`LoraSettings` (r, alpha, dropout, bias, target_modules, task_type),
`DpoSettings` (beta, loss_type), `TrainingSettings` (max_length, max_prompt_length,
learning_rate, num_train_epochs, max_steps, per_device batch sizes,
gradient_accumulation_steps, gradient_checkpointing, warmup_ratio, max_grad_norm,
lr_scheduler_type, seed, data_seed, save_steps, eval_steps, logging_steps,
max_truncation_rate, min_training_pairs, apply_chat_template),
`OptimizerSettings` (name), `DatasetSpec` (preference_run_id), plus `experiment_name`
(§98) and `distributed.enabled` (§87 — accepted, and **rejected if true**, since Step 9
must not implement it).

`ExperimentConfig.load(path)` + `with_overrides(**cli_flags)` so §63's CLI overrides
never require editing YAML *or* Python. `to_dict()` is what gets copied into the run
directory as `config.yaml` (§69).

**`versions.py`** — `capture_versions() -> dict` (§71): torch, transformers, trl, peft,
bitsandbytes, accelerate, datasets, safetensors via `importlib.metadata.version`,
recording `None` for a missing optional package rather than raising; plus CUDA runtime and
driver version. Pure stdlib at module scope.

**`hardware.py`** — §12/§13, mirroring `sandbox/health.py` exactly.
`HardwareCheck(name, passed, detail)` / `HardwareReport(checks, .passed, .failures)` plus
a `HardwareInfo` record persisted as `hardware.json`: GPU name, device count, VRAM
**total and free**, CUDA version, torch CUDA version, compute capability, BF16 support,
4-bit (bitsandbytes) availability. `check_hardware(config=None, probe=None)` takes an
**injectable probe** so the whole thing is unit-testable with a fake and no GPU;
`format_hardware_report(report) -> str` renders §13's shape.

`resolve_compute_dtype(...)` implements §10/§51: bfloat16 when
`torch.cuda.is_bf16_supported()`, else float16, never both, never blindly requesting BF16.

**`dataset.py`** — §21–§27, §102–§106. `load_training_dataset(preference_run_dir, ...)`:

1. Verify the run and its `train`/`validation`/`test.jsonl` exist (§24.1, §24.8).
2. Parse each line, validating `prompt`/`chosen`/`rejected` are present, non-empty
   **strings**, and `chosen != rejected` (§23, §24.2–§24.7, §101).
3. SHA-256 each split file via the existing `candidates.hashing.sha256_text` (§26).
4. Read Stage 8's `split_manifest.json` and assert the three problem-id sets are
   **disjoint** — abort on any overlap (§102).
5. Read Stage 8's `manifest.json` for `preference_run_id`/`preference_version`/
   `selection_policy`/`dataset_schema_version` (§25), and `metadata.jsonl` for the §105
   balance stats (mean chosen/rejected score, mean margin, strong/medium share) and §106
   histograms.
6. Character-level statistics (§27); emit the §103 small-dataset warning below
   `min_training_pairs` (default 500 — this dataset will always trip it).

**The test split is loaded only to hash it and prove disjointness (§26, §102). It is never
handed to the trainer (§21, §22)** — enforced by `TrainingDataset` exposing `train` and
`validation` as datasets but `test` only as a hash and an id set.

**`lengths.py`** — §34–§36. `analyze_lengths(records, tokenizer, max_length,
max_prompt_length)` returns p50/p90/p95/p99/max for prompt, chosen, rejected,
prompt+chosen, prompt+rejected, plus `truncated_examples` and `truncation_rate`. Exceeding
`max_truncation_rate` (default **0.05**, §36) raises `TruncationThresholdError` with the
recommendation to adjust sequence lengths — overridable by an explicit flag. The tokenizer
is a parameter, so tests inject a trivial fake and this runs offline.

**`loader.py`** — §6–§20, all deferred-import.

- `load_tokenizer(model_spec)` — from the **exact** base model (§28), recording name,
  revision, vocab size, special tokens. Pad-token handling per §29: if absent, fall back
  to EOS and **record the decision explicitly**; set `padding_side="left"` as TRL's
  processing class expects.
- `build_quantization_config(settings, compute_dtype)` — the §9 `BitsAndBytesConfig`,
  every option configurable.
- `load_model(...)` — 4-bit load, then `prepare_model_for_kbit_training`, then §6's
  compatibility checks (causal LM, sequence length, architecture) which **fail before any
  artifact is written** (§6).
- `validate_target_modules(model, configured)` — checks against `model.named_modules()`;
  **raises if none match** (§16). Never trains with zero LoRA targets.
- `apply_lora(model, settings)` — PEFT `LoraConfig` with `task_type="CAUSAL_LM"` (§18),
  `bias` configurable defaulting to `"none"` (§17).
- `count_parameters(model) -> ParameterCounts` — total/trainable/percentage (§19), and
  **`FullFineTuneError` if `trainable == total`** (§20). This is the single most important
  safety check in the stage and gets its own test.

**`callbacks.py`** — a `transformers.TrainerCallback` appending every log to
`metrics/metrics.jsonl` (§78, §93): step, epoch, train_loss, eval_loss, learning_rate,
plus whatever DPO reward metrics the installed TRL emits (`rewards/chosen`,
`rewards/rejected`, `rewards/accuracies`, `rewards/margins` — **names vary by version, so
they are passed through rather than hard-coded**, §78). Also records allocated/reserved/
peak GPU memory (§84) and never logs candidate code (§55).

**`trainer.py`** — `DpoTrainingJob`, `TRAINING_VERSION = "v1"`. One class with an explicit
phase sequence so `--dry-run` and `--smoke-test` are the *same* code path stopped early:

```
preflight:  hardware -> versions -> dataset -> tokenizer -> lengths/truncation
            -> model -> quantization -> LoRA -> parameter counts -> memory estimate
dry-run:    stop here (§65)
smoke-test: subset to 2-10 train / 1-2 validation, max_steps 1-5 (§66)
train:      DPOConfig -> DPOTrainer -> train() -> save adapter -> reload check (§67, §74)
```

Reference model: **`ref_model=None` with `peft_config` supplied**, so TRL uses the
adapter-disabled base as the implicit reference — §46/§47's "do not instantiate
unnecessary duplicate full-precision models on a 12 GB GPU". Gradient checkpointing on by
default with `use_cache=False` (§48, §49). Optimizer `paged_adamw_8bit` with a recorded
fallback to AdamW when bitsandbytes cannot provide it (§50).

`CUDA OOM` is caught and re-reported with model, sequence length, batch size, gradient
accumulation, LoRA config and GPU memory (§83) — and **never silently retried with a
smaller configuration** (§83 is explicit). Any failure marks the run `failed` with
error type, message, traceback and last completed step (§82), and the adapter is **not**
marked final unless it saved *and* passed reload (§82).

**`verify.py`** — §74/§75. `verify_adapter(training_run_dir)` loads base + saved adapter +
tokenizer, runs a fixed test prompt, and asserts generation succeeds.
`run_inference(training_run_dir, prompt)` backs `train inference`. `baseline_response(...)`
covers §97's pre-training baseline capture on the same fixed prompt set — recorded, and
explicitly **not** a benchmark.

**`models.py`** — the persisted schema: `TrainingManifest` (§70 — every field, including
package versions, seeds, hardware, upstream `preference_run_id`/`ranking_run_id`/
`evaluation_run_id`, start/end time, and the status lifecycle mirroring Stage 4/6/7/8),
`DatasetManifest` (§25/§26 — provenance and the three hashes), `FinalReport` (§94).

**`run_repository.py`** — `TrainingRunRepository(training_root)`: mints
`dpo_YYYYMMDD_HHMMSS_xxxx` (§68) via the same collision-checked `secrets.token_hex(2)`
template, owns `manifest.json`/`dataset_manifest.json`/`hardware.json`/`final_report.json`,
and the status lifecycle. Resume support (§90–§92): `--resume-from-checkpoint` verifies
model, tokenizer, LoRA config, quantization mode **and dataset hashes** against the
original manifest, refusing on mismatch unless `--force-resume` (§91), and failing outright
on a critical config difference (§92). **Fifth copy of the run plumbing** — re-flagged as
debt; extracting a shared base would now touch five stages.

**`statistics.py`** — formatters: `format_hardware_report` lives in `hardware.py` (with its
dataclasses, like `sandbox/health.py`); this holds `format_dataset_statistics`,
`format_length_analysis`, `format_parameter_counts`, `format_final_report`.

---

## Persistence layout (§69, §109)

```
data/training/runs/dpo_20260818_101500_a91f/
├── manifest.json          # §70 versions, seeds, hardware, upstream ids, status
├── config.yaml            # the resolved experiment config actually used
├── dataset_manifest.json  # §25/§26 preference provenance + three split hashes
├── hardware.json          # §12 GPU/CUDA/BF16/4-bit capability at run start
├── metrics/metrics.jsonl  # §78/§93 per-log-step metrics
├── logs/training.log      # §93
├── adapter/               # §73 PEFT-format final adapter  [tracked, ~15 MB]
├── checkpoints/           # §57 periodic checkpoints        [gitignored]
├── tokenizer/             # §69 tokenizer snapshot          [gitignored]
└── final_report.json      # §94
```

`.gitignore` gains `data/training/runs/*/checkpoints/` and
`data/training/runs/*/tokenizer/` (decision 3) — the first exclusion under `data/` since
`data/raw/`, and for the same reason: reproducible bulk, not a deliverable.

---

## Modifications to existing code

**`configs/training/dpo_qlora.yaml`** (new top-level `configs/` directory, per §62) — the
experiment defaults, RTX-3060-conservative: 4-bit NF4 double-quant bfloat16 compute; LoRA
r=16/alpha=32/dropout=0.05/bias=none over `q_proj,k_proj,v_proj,o_proj`; beta 0.1, loss
sigmoid; max_length 1024, max_prompt_length 512, lr 1e-5, 1 epoch, batch 1, grad-accum 8
(effective 8, §38), gradient checkpointing on, warmup 0.05, max_grad_norm 1.0, seed 42;
`paged_adamw_8bit`. Deliberately a **separate file** from root `config.yaml`, which stays
the pipeline config — this keeps per-experiment hyperparameters (the things you iterate
on) out of it, and sidesteps `ModelConfig.quantization`'s Stage-3 rejection entirely.

**`src/python_dpo/config.py` + `config.yaml` + `data/`** — the **eighth** data path.
`_REQUIRED_PATH_KEYS` gains `training`; `Paths` gains `training: Path` and includes it in
`ensure_exists()`; `config.yaml` gains `training: data/training`; `data/training/.gitkeep`
is created. **No `training:` settings section in root `config.yaml`** — those live in the
experiment file above.

**`src/python_dpo/cli.py`** — a new `train` command group:

| Command | Behavior |
|---|---|
| `train hardware-check` | §13's report; exit 1 if unusable |
| `train dpo --config PATH --preference-run-id ID [--dry-run] [--smoke-test] [--allow-small-dataset] [--resume-from-checkpoint PATH] [--force-resume] [--experiment-name NAME] [--learning-rate/--beta/--epochs/--max-steps/--seed ...]` | §64, §65, §66, §90 |
| `train verify --training-run-id ID` | §74's mandatory adapter reload; prints `Adapter reload successful.` |
| `train inference --training-run-id ID --prompt TEXT` | §75 |
| `train list` | Training runs, newest first — mirrors `preferences list`, needed to discover an id |

`_PLACEHOLDER_STAGES` is untouched (it holds only `run`; there is no `train` placeholder
today). Handlers keep the house contract: `(args, config) -> int`, data to `sys.stdout`,
errors via `logger.error`, exit codes 0/1/2/130.

**`src/python_dpo/preferences/` — one small addition.** `preferences generate` gains
`--split-ratios TRAIN,VALIDATION,TEST`, so decision 1's dataset can be minted from the CLI
rather than by hand-editing `config.yaml`. Backward compatible (defaults to the config
value), mirrors the existing `--split-seed`, and is the same "don't require editing
config to change a hyperparameter" principle Stage 8 already applies to `--margin`.

**`pyproject.toml`** — a `training` extra, floored at the versions actually verified here,
because TRL 1.x's `DPOTrainer`/`DPOConfig` API differs materially from 0.x:

```toml
training = [
    "torch>=2.2", "transformers>=4.56", "accelerate>=0.30",
    "trl>=1.10", "peft>=0.20", "bitsandbytes>=0.50", "datasets>=5.0",
]
```

Also a `gpu` marker, with `addopts = "-ra -m 'not integration and not gpu'"` so `pytest -q`
stays offline and zero-skip.

**`tests/test_no_heavy_imports.py`** — `HEAVY_MODULES` grows to include `trl`, `peft`,
`bitsandbytes`, `datasets`; the probe additionally imports `python_dpo.training` and every
submodule. This is what keeps the deferred-import discipline honest as the package grows.

**`src/python_dpo/__init__.py`** — `__version__` → `0.9.0`.

**Docs** — `src/python_dpo/training/README.md` (new), plus Stage 9 sections in `README.md`,
`src/python_dpo/README.md`, `data/README.md`, `tests/README.md`, and a `configs/README.md`.

---

## Tests

Two tiers, because most of this stage cannot run without a GPU.

**Offline (`tests/training/`, default suite — no GPU, no model, no heavy imports):**

- **`test_config.py`** — the §62 schema: defaults, unknown-key rejection, every validation
  rule, `--` override precedence over YAML, `distributed.enabled: true` rejected (§87),
  round-trip `to_dict`.
- **`test_versions.py`** — version capture with a faked `importlib.metadata`, including a
  missing optional package recorded as `None` rather than raising.
- **`test_hardware.py`** — `check_hardware` against an **injected fake probe**: no CUDA;
  CUDA but insufficient free VRAM; BF16 supported vs not (§10 fallback to fp16, never
  both); bitsandbytes absent. Report formatting and the short-circuit-on-first-failure
  behavior, mirroring `tests/sandbox/test_config.py`'s approach to `health.py`.
- **`test_dataset.py`** — §24's nine checks one at a time against fixtures built with
  `tmp_path`: missing file, malformed JSONL, missing field, empty prompt/chosen/rejected,
  `chosen == rejected`, empty train, empty validation (and that it *passes* with
  `allow_small_dataset=True`), hash stability, §102 split-overlap abort, §103 small-dataset
  warning, and — the security-relevant one — that `chosen`/`rejected` are only ever
  handled as `str` (§101).
- **`test_lengths.py`** — percentile arithmetic and truncation rate against a trivial fake
  tokenizer; the §36 threshold raising, and the explicit override suppressing it.
- **`test_loader.py`** — `validate_target_modules` against a fake module tree (all match,
  some match, **none match → raises**, §16); `count_parameters` against a fake model with
  known frozen/trainable splits, including **`trainable == total` → `FullFineTuneError`**
  (§20). No torch needed — the fakes expose `named_modules()`/`parameters()`.
- **`test_run_repository.py`** — id format and collision retry, status lifecycle, resume
  compatibility checks (§90–§92): dataset-hash mismatch refused, `--force-resume`
  overriding it, a LoRA-r/target-module/quantization/base-model change failing outright.
- **`test_models.py`** — manifest/dataset-manifest/final-report schema round-trips and
  validation.
- **`test_statistics.py`** — the formatters.
- **`tests/test_project.py`** — the eighth data directory in all three literal
  enumerations; `train` CLI parsing and error paths.

**GPU-gated (`tests/training/test_gpu_integration.py`, `pytestmark = pytest.mark.gpu`):**
mirrors the Docker suites' philosophy — a session fixture that **`pytest.fail()`s rather
than skips** when CUDA or the training extra is absent, with a message naming the fix.
Covers: real hardware check; 4-bit NF4 load; LoRA application and a real parameter count
(asserting the ~0.24% figure); `DPOTrainer` initialization; one forward+backward step;
**LoRA parameters actually changed** (compare a tensor before/after); checkpoint save;
adapter save; adapter reload and generation (§67's full list).

---

## Execution order

1. Write this plan to `.claude/plans/09_dpo_qlora_training_plan.md` and add its entry to
   `.claude/plans/README.md`.
2. `pip install -e '.[training]'` and **verify the installed TRL's `DPOTrainer`/`DPOConfig`
   signature by introspection** before writing `trainer.py` against it. Record the exact
   resolved versions. **This is the first real step because the TRL 1.x API is the stage's
   biggest unknown** — if it differs from expectation, `trainer.py`'s shape changes.
3. `errors.py`, `config.py`, `models.py`, `versions.py` + tests — pure schema.
4. `hardware.py` + tests; wire `train hardware-check`; run it for real.
5. Mint the decision-1 dataset: add `--split-ratios`, then
   `preferences generate --policy all_better --split-ratios 0.5,0.25,0.25` — and
   `preferences validate` it.
6. `dataset.py`, `lengths.py` + tests.
7. `loader.py` + tests (the §16/§20 safety checks first).
8. `run_repository.py` + tests.
9. `callbacks.py`, `trainer.py`, `verify.py`; wire `train dpo`/`verify`/`inference`/`list`.
10. Config path wiring (eighth path, three `test_project.py` enumerations), `.gitignore`,
    `pyproject.toml` extra + `gpu` marker, `test_no_heavy_imports.py` extension.
11. `tests/training/test_gpu_integration.py`.
12. Run §108's six steps for real; commit the training run per decision 3; docs; the §111
    report in `09_DPO_QLORA_TRAINING.md`.

---

## Verification

```bash
source .venv/bin/activate
pip install -e '.[training]'
pytest -q                     # offline, zero skips, no GPU, no Docker
pytest -q -m gpu              # the real stack on the RTX 3060

# §108 step 1
python -m python_dpo train hardware-check

# §108 step 2 — the decision-1 dataset
python -m python_dpo preferences generate --ranking-run-id rank_20260817_161726_a84d \
    --policy all_better --split-ratios 0.5,0.25,0.25
python -m python_dpo preferences validate --preference-run-id PREF_ID

# §108 steps 3-5
python -m python_dpo train dpo --config configs/training/dpo_qlora.yaml \
    --preference-run-id PREF_ID --dry-run
python -m python_dpo train dpo --config configs/training/dpo_qlora.yaml \
    --preference-run-id PREF_ID --smoke-test
python -m python_dpo train dpo --config configs/training/dpo_qlora.yaml \
    --preference-run-id PREF_ID --allow-small-dataset

# §108 step 6 — mandatory
python -m python_dpo train verify --training-run-id TRAIN_ID   # "Adapter reload successful."
python -m python_dpo train inference --training-run-id TRAIN_ID --prompt "Write a Python function that reverses a string."
```

**Expected, computed from the measured facts above:**

```
hardware      RTX 3060 · 12288 MiB total · ~11.7 GiB free · CUDA 13.0 · BF16 yes · 4-bit yes
dataset       all_better 0.5/0.25/0.25 -> train=3  validation=2  test=2 records
LoRA          r=16 over q,k,v,o x 36 layers = 7,372,800 trainable of ~3.09B  (~0.24%)
adapter       ~15 MB bf16
effective bs  1 x 8 x 1 = 8   ->  3 training records is well under one full accumulation cycle
```

The §103 small-dataset warning **will** fire (3 ≪ 500). That is expected and correct.

```bash
# the base model is untouched — only the adapter is an output (§3, §72)
ls data/training/runs/TRAIN_ID/adapter/     # adapter_config.json, adapter_model.safetensors
du -sh data/training/runs/TRAIN_ID/adapter/ # ~15 MB, not GB

# nothing upstream was mutated
git diff --stat data/problems/ data/candidates/ data/evaluations/ data/rankings/   # empty

# the test split never reached the trainer (§21, §22)
grep -rn "test" data/training/runs/TRAIN_ID/manifest.json   # only a hash + id list
```

Scope containment:

```bash
grep -rn "exec(\|eval(\|subprocess" src/python_dpo/training/          # none (§100)
grep -rn "torchrun\|DistributedDataParallel\|FSDP\|deepspeed" src/python_dpo/training/  # none (§86)
grep -rn "merge_and_unload" src/python_dpo/training/                  # none (§76)
grep -rnE "^import (torch|trl|peft|bitsandbytes)|^from (torch|trl|peft|bitsandbytes)" \
     src/python_dpo/training/                                          # none — all deferred
pytest -q tests/test_no_heavy_imports.py                               # the real guard
```

Then produce the §111 report in `09_DPO_QLORA_TRAINING.md` and **stop — do not start
Step 10 (the programming benchmark and trained-vs-base evaluation) without explicit
approval** (§111).

**The honest headline for that report (§95, §110):** three training records over two
problems is under one gradient-accumulation cycle. Stage 9 will have demonstrated that the
QLoRA/DPO stack loads, quantizes, attaches an adapter, takes a real optimizer step, saves,
and reloads — and **nothing whatsoever about whether the model writes better Python**.
That claim belongs to Step 10, on the test split this stage deliberately never touched.

---

## Deviations to record in the report

- **A third preference dataset is minted** (`all_better`, ratios 0.5/0.25/0.25) purely to
  obtain a non-empty validation split — no committed Stage 8 dataset has one, and no seed
  choice can produce one below 10 pair-bearing problems (decision 1).
- **An empty validation split is tolerated behind `--allow-small-dataset`**, running with
  `eval_strategy="no"`, rather than failing §24.9 outright — otherwise the two committed
  datasets would be permanently untrainable (decision 1).
- **`preferences generate` gains `--split-ratios`** — a small backward-compatible Stage 8
  amendment so the decision-1 dataset is reproducible from the CLI rather than by editing
  `config.yaml`.
- **Training applies the Qwen chat template** to the model-agnostic stored prompt, because
  Stage 3 generation demonstrably did (`qwen.py:_render`); training on the bare string
  would train on a format the candidates were never produced under (§30, §31, decision 4).
- **Hyperparameters live in `configs/training/dpo_qlora.yaml`, not root `config.yaml`** —
  per §62, and because `ModelConfig.quantization` still rejects any non-null value as
  "not implemented in Stage 3".
- **Checkpoints and the tokenizer snapshot are gitignored** while the adapter is tracked —
  the first exclusion under `data/` since `data/raw/`, since checkpoints are a
  reproducible size class rather than a deliverable (decision 3).
- **Version floors are set at the versions actually verified** (`trl>=1.10`, `peft>=0.20`,
  `bitsandbytes>=0.50`, `datasets>=5.0`) rather than the lowest plausible, because TRL 1.x
  changed the `DPOTrainer`/`DPOConfig` surface and this stage is only tested against 1.x.
- **A fifth copy of the run-directory plumbing**, rather than the shared base deferred at
  Stages 7 and 8. Extracting it would now touch five stages at once; re-flagged as debt.
- **`train list` was added** beyond §64/§74/§75, since none of the specified commands is
  usable without a way to discover a `training_run_id`.
- **§58's "preserve all checkpoints" holds on disk but not in git**, per decision 3.
- **Resume is implemented but barely exercisable at this data scale** — three records
  produce too few steps for a mid-run interruption to be meaningful; required by §90–§92
  and the §107 acceptance criteria.
