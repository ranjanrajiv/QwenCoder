# src/python_dpo/training/

DPO/QLoRA training — fine-tunes a LoRA adapter over a frozen, 4-bit quantized Qwen Coder
base using TRL's `DPOTrainer`, from the `{prompt, chosen, rejected}` preference data
Stage 8 produced.

Three rules govern the package:

- **The base model stays frozen.** Only LoRA parameters train, and `count_parameters`
  *refuses to proceed* if every parameter is trainable — that would be a full fine-tune,
  which this stage must never perform by accident.
- **No candidate code is executed.** `chosen`/`rejected` are validated to be strings and
  are only ever tokenized. Execution belongs exclusively to the Stage 5 sandbox and the
  Stage 6 evaluator. There is no `exec`, no `eval`, no `subprocess` on candidate text
  anywhere in this package.
- **No performance claim.** A falling DPO loss is not evidence of better Python. This
  stage produces `base model + adapter`; Step 10 evaluates it.

**Import discipline.** Nothing here imports torch, transformers, trl, peft, bitsandbytes
or datasets at module scope — every heavy import is deferred into the function that needs
it, via the same `_import_backend()` idiom `models/qwen.py` uses. `import python_dpo` must
stay cheap, and `tests/test_no_heavy_imports.py` enforces it.

## Files

### `errors.py`

`TrainingError` and its subclasses, grouped by what actually went wrong: **environment**
(`TrainingDependencyError`, `HardwareError`), **preflight** (`DatasetValidationError`,
`TargetModuleError`, `TruncationThresholdError`, `FullFineTuneError`), and **run**
(`TrainingRunNotFoundError`, `CheckpointCompatibilityError`, `AdapterReloadError`).

`FullFineTuneError` is the safety trip: reaching it means LoRA silently failed to freeze
the base model.

### `config.py`

The experiment schema, loaded from a **standalone YAML file** (`configs/training/
dpo_qlora.yaml`) rather than the root `config.yaml`. Two reasons: these are the values an
experimenter iterates on, so a second experiment should be a second file; and
`models.base.ModelConfig` actively rejects a non-null `quantization`, so the root config
cannot express 4-bit NF4 at all.

`with_overrides()` applies CLI flags by re-running the whole schema, so a flag can no more
produce an invalid configuration than the YAML could.

**Two fields are deliberately not passed to TRL.** `max_prompt_length` and `warmup_ratio`
are spec-mandated but absent from TRL 1.10's `DPOConfig`; see `trainer.py`.

### `versions.py`

Package and driver version capture. Follows `evaluation/probe.py`'s principle — record
what *genuinely ran*, not what the dependency pins asked for — via
`importlib.metadata`. A missing optional package is recorded as `None` rather than
raising, so `train hardware-check` still works on a partial install.

### `hardware.py`

Mirrors `sandbox/health.py` exactly: frozen `HardwareCheck`/`HardwareReport`, checks that
short-circuit at the first failure, every failure detail ending in something actionable,
and a separate formatter. `check_hardware()` takes an **injectable probe**, so the report
shapes and the BF16/FP16 fallback are unit-testable with no GPU.

**Free VRAM, not total, is what is checked.** A 12 GB card running a desktop session
permanently holds several hundred MB; reporting total would overstate the budget by
exactly the amount most likely to cause an OOM.

`resolve_compute_dtype` implements the bf16→fp16 fallback: bfloat16 where the GPU
supports it, float16 otherwise, never both, never blindly requested.

### `dataset.py`

Loads, validates, hashes and describes a Stage 8 preference run — all **before the model
is loaded**, since there is no point spending minutes on 4-bit weights for a dataset that
was never trainable.

Two rules are structural rather than merely checked:

- **The test split never reaches the trainer.** `TrainingDataset` exposes `train` and
  `validation` as records but `test` only as a hash, a count and a problem-id set. There
  is no attribute a caller could hand to `DPOTrainer` by mistake.
- **`chosen`/`rejected` are only ever text.** They are validated to *be* `str`, and are
  never parsed, compiled or executed.

`allow_small_dataset` relaxes exactly one rule: the requirement that the validation split
be non-empty. Stage 8's problem-level splitter floors validation at
`floor(n_problems × 0.1)`, which is zero below ten pair-bearing problems, so without the
escape hatch most datasets this project can produce would be permanently untrainable. An
empty *training* split is always fatal.

### `lengths.py`

Token-length percentiles and truncation counting. An example counts as truncated when
*either* `prompt + chosen` or `prompt + rejected` exceeds `max_length` — DPO sees both
sequences, so losing either damages the pair. Exceeding `max_truncation_rate` (default
0.05) fails preflight with the p95 length in the message, so the remedy is obvious.

The tokenizer is a parameter, so the whole analysis runs offline against a trivial fake.

### `loader.py`

Tokenizer, quantization, model and LoRA loading. Two functions here are the real safety
net, and both fail loudly rather than warn:

- `validate_target_modules` — raises when **no** configured LoRA target exists on the
  model. Training with zero targets would produce an adapter that changes nothing while
  reporting success.
- `count_parameters` — raises when `trainable == total` (LoRA never froze the base, so
  this is a full fine-tune) *or* when `trainable == 0` (LoRA never attached, so training
  is a silent no-op).

`resolve_optimizer` falls back from `paged_adamw_8bit` to AdamW when bitsandbytes is
unavailable, and **records** the substitution rather than hiding it — it changes the run's
memory profile.

### `callbacks.py`

Appends every trainer log to `metrics/metrics.jsonl` as it happens, so an interrupted run
still leaves behind everything it measured. **DPO reward metric names are passed through,
not enumerated** — they vary by TRL version, and hard-coding the list would silently drop
metrics on any other one. Non-numeric values are dropped, which is what keeps candidate
code out of the metrics file.

### `trainer.py`

`--dry-run`, `--smoke-test` and a real run are the **same code path stopped at different
points**, which is the whole value of the dry run: it exercises the code that will
actually train.

Reference model: `ref_model=None` with a `peft_config`, so TRL uses the adapter-disabled
base as the implicit reference — no second full-precision model on a 12 GB card. A CUDA
OOM is re-reported with the whole configuration and **never silently retried smaller**.

Two TRL 1.10 realities are handled here, both verified by introspection rather than
assumed: `DPOConfig` has no `max_prompt_length` (prompt and completion truncate together
against `max_length`, so ours drives the preflight analysis instead) and no
`warmup_ratio` (the configured ratio is converted to `warmup_steps` once the schedule
length is known).

### `verify.py`

The mandatory adapter reload. Loads the base model **fresh** rather than reusing the
in-memory training model: the point is to prove the *saved artifact* works, which an
in-memory handle would not establish. An adapter that does not reload is not a training
artifact, whatever the loss curve said, and the run is not marked successful.

`_render_prompt` applies the model's chat template, matching both Stage 3 generation and
training — verifying under a format the model was never trained on would measure the
wrong thing.

### `run_repository.py`

Mints `dpo_YYYYMMDD_HHMMSS_xxxx` ids and owns the run's JSON artifacts and status
lifecycle. A **fifth** copy of this plumbing after Stages 4, 6, 7 and 8; the extraction has
been deferred at every stage since Stage 7, and doing it now would touch five stages at
once, so the debt is carried deliberately.

Resume compares manifests, so it needs no torch and is testable offline. A **dataset**
difference is refusable but overridable with `--force-resume` (and the override is
recorded); a **critical configuration** difference — base model, LoRA rank, target
modules, quantization — is never overridable, because the saved optimizer and adapter
state do not describe the same training problem.

## Persistence layout

```
data/training/runs/dpo_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json          # versions, seeds, hardware, upstream ids, status
├── config.yaml            # the resolved experiment config actually used
├── dataset_manifest.json  # preference provenance + all three split hashes
├── hardware.json          # GPU/CUDA/BF16/4-bit capability at run start
├── metrics/metrics.jsonl  # per-log-step metrics and GPU memory
├── logs/training.log      # the human-readable narrative
├── adapter/               # the PEFT adapter          [tracked, ~14 MB]
├── checkpoints/           # periodic checkpoints      [gitignored]
└── final_report.json
```

All three split hashes are recorded including `test` — not because test is used, but
because reproducing a run means proving the same test split was held out.

`checkpoints/`, `tokenizer/`, and TRL's `adapter/ref/` (a frozen fp32 reference copy, not
needed to load the adapter) are gitignored: reproducible bulk rather than deliverables.
The trained adapter itself is tracked — it is the artifact.
