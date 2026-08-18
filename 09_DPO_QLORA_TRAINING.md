# Stage 9 Implementation Details — Qwen Coder DPO/QLoRA Training

How `src/python_dpo/training/` implements the layer specified in
`.claude/specs/09_dpo_qlora_training.md`. For usage, see the "Stage 9 — DPO/QLoRA
Training" section of the root `README.md`. This file is about *how* it is built and what
was learned building it.

## Goal

Stage 8 produced `{prompt, chosen, rejected}` records backed by execution evidence. Stage 9
is the first stage that changes a model: it trains a LoRA adapter over a frozen, 4-bit
NF4-quantized Qwen Coder base using TRL's `DPOTrainer`, with the preflight, provenance and
verification apparatus that makes the result reproducible.

Three constraints define the stage:

- **The base model stays frozen.** Only LoRA parameters train. `count_parameters` refuses
  to proceed when `trainable == total` — that is a full fine-tune, which Step 9 must never
  perform by accident — and equally when `trainable == 0`, which would be a silent no-op.
- **No candidate code is executed.** `chosen`/`rejected` are validated to *be* strings and
  are only ever tokenized. Verified: no builtin `exec`/`eval` anywhere in the package, no
  `shell=True`, and the single `subprocess` call is a fixed-argv `nvidia-smi`.
- **No performance claim.** A falling DPO loss is not evidence of better Python. This stage
  produces `base model + adapter`; Step 10 evaluates it on the test split this stage never
  touched.

## 1–3. Base model, revision, tokenizer

`Qwen/Qwen2.5-Coder-3B-Instruct`, revision `null` (the default branch; whatever it resolves
to is recorded in the run manifest). The tokenizer is loaded from the **same** model id and
revision, never another Qwen checkpoint. It declares a pad token, so the EOS fallback path
was not taken — and would have been recorded if it were. `padding_side` is set to `left`,
which TRL's DPO processing class expects.

## 4–11. Environment

| | |
|---|---|
| Python | 3.12.3 |
| torch | 2.13.0+cu130 |
| transformers | 5.15.0 |
| trl | **1.10.0** |
| peft | 0.20.0 |
| bitsandbytes | 0.50.1 |
| accelerate / datasets / safetensors | 1.14.0 / 5.0.1 / 0.8.0 |
| CUDA | 13.0 (driver 580.82.07) |
| GPU | NVIDIA GeForce RTX 3060, compute capability 8.6 |
| VRAM | 12,478,906,368 bytes total; ~11.2 GiB free with a desktop session running |

## 12. TRL 1.10's API, and two spec fields that no longer exist

The spec's biggest unknown was the TRL API, so the first implementation step was to install
the stack and **introspect it** rather than code against documentation for an older release.
Two spec-mandated `DPOConfig` fields are simply absent from TRL 1.x:

- **`max_prompt_length` (§33) does not exist.** Prompt and completion are truncated together
  against `max_length`, with `truncation_mode` choosing which end. It is kept in this
  stage's own configuration because it remains a genuinely useful *diagnostic* — a prompt
  that alone consumes most of `max_length` leaves no room for a response — and it drives the
  §34/§35 length analysis. It is simply not passed to the trainer.
- **`warmup_ratio` (§53) does not exist**; `DPOConfig` exposes `warmup_steps`. The
  configured ratio is converted to a step count once the schedule length is known.

Also worth recording: `loss_type` is now a *list* internally (`['sigmoid']`), though a bare
string is accepted and normalized, so the spec's `loss_type: sigmoid` works unchanged.

## 13–14. Quantization and LoRA configuration

4-bit NF4, double quantization, bfloat16 compute (resolved against the device — the RTX
3060 supports BF16, so no FP16 fallback was needed). LoRA `r=16`, `alpha=32`,
`dropout=0.05`, `bias=none`, `task_type=CAUSAL_LM`, targeting `q_proj,k_proj,v_proj,o_proj`.

## 15–18. Target modules and parameter counts

All four configured targets exist on the real Qwen2 module tree; `validate_target_modules`
resolved them and would have raised had none matched.

```
Total parameters:       1,706,045,440
Trainable parameters:       7,372,800
Trainable percentage:         0.4322%
```

**The trainable count is exactly 7,372,800**, matching the figure predicted from the
architecture before any code ran: 36 layers × (q: 16·2048+2048·16, k: 16·2048+256·16,
v: same, o: 65,536) = 204,800 per layer. Note the grouped-query attention — `k_proj`/
`v_proj` are 2048→256, not 2048→2048 — which is why the count is not simply 4×65,536×36.

The total reads 1.7B rather than the model's nominal 3.09B because bitsandbytes packs two
4-bit weights per `uint8` element, so `numel()` counts packed storage. The trainable
percentage is therefore 0.43% against packed storage (≈0.24% against unpacked). Either way
the §19/§20 requirement is satisfied by more than two orders of magnitude.

## 19–20. Dataset

| | |
|---|---|
| Preference run | `pref_20260818_074347_5eff` (`all_better`, ratios 0.5/0.25/0.25) |
| Train / validation / test | **3 / 2 / 2** records |
| Problems | train `p007,p008` · validation `p010` · test `p004` |
| `train.jsonl` SHA-256 | `2bec77c843269e31…` |
| Source pair balance | mean margin 0.2235 · 54.5% strong · 45.5% medium |

**Why a new dataset was generated.** Every previously committed Stage 8 dataset has an
*empty* validation split — strict is 1/0/2, margin is 2/0/2 — because Stage 8's splitter
floors validation at `floor(n_problems × 0.1)`, which is zero below ten pair-bearing
problems, and only 2–4 problems ever produce pairs. §24.9 requires a non-empty validation,
so **no dataset that existed before this stage was trainable under a literal reading of the
spec**. The `all_better` policy at 0.5/0.25/0.25 ratios is the only reachable configuration
that yields one.

## 21–23. Sequence lengths and truncation

```
QUANTITY             P50     P90     P95     P99     MAX
prompt               193     215     215     215     215
chosen                86     123     123     123     123
rejected              83     145     145     145     145
prompt_chosen        279     338     338     338     338
prompt_rejected      276     360     360     360     360

Truncation: 0/5 examples (0.0%) exceed max_length=1024
```

`max_length=1024` is generous for this data — the longest sequence is 360 tokens — so the
5% truncation gate never engaged. It is nonetheless enforced, and tested.

## 24–31. Training configuration and results

| | |
|---|---|
| DPO beta / loss | 0.1 / sigmoid |
| Learning rate | 1e-5, cosine schedule |
| Batch × accumulation | 1 × 8 = **effective 8** |
| Epochs / steps | 1 / **1** |
| Optimizer | `paged_adamw_8bit` (no fallback needed) |
| Precision | bf16 (fp16 off — never both) |
| Gradient checkpointing | on, `use_cache=False` |
| Final train loss | 0.693147 |
| Final eval loss | 0.676852 |
| DPO reward metrics | `rewards/chosen` 0.0 · `rewards/rejected` 0.0 · `rewards/margins` 0.0 · `rewards/accuracies` 0.0 |
| Peak GPU memory | **4.66 GiB** |
| Duration | 26.3s |
| Adapter | 14.1 MiB, reload **OK** |

**Read the reward metrics honestly.** `0.693147` is `ln(2)` — the DPO loss at
initialization — and all four reward metrics are exactly zero. That is not a bug: with 3
training records and an effective batch size of 8, the run is a **single optimizer step**,
and the metrics are logged at that step while the policy still equals the reference, so
chosen and rejected rewards are identically zero by construction.

The smoke test, which runs 2 steps over 2 epochs, does show movement —
`rewards/margins` 0.0151, `rewards/accuracies` 0.667, eval margin 0.0271 — confirming the
objective does what it should once there is more than one step. Neither result says
anything about Python ability.

## 32. Verification procedure (§108), all six steps

| Step | Result |
|---|---|
| 1. `train hardware-check` | passed — CUDA 13.0, RTX 3060, 11.2 GiB free, BF16 yes, 4-bit yes |
| 2. `preferences validate` | `Preference dataset validation passed.` |
| 3. `train dpo --dry-run` | preflight only; parameter counts reported; no training |
| 4. `train dpo --smoke-test` | completed; forward, backward, LoRA update, checkpoint, reload |
| 5. `train dpo` | completed; `dpo_20260818_081231_a91d` |
| 6. `train verify` | **`Adapter reload successful.`** (64 tokens generated) |

`train inference` was also exercised and produced coherent Python from base + adapter.

## 33. Test results

```
pytest -q                    1052 passed, 51 deselected      (offline, zero skips)
pytest -q -m gpu                5 passed                      (real GPU)
```

Of those, 155 are new Stage 9 offline tests plus 5 GPU tests:

```
tests/training/test_config.py           25 passed
tests/training/test_dataset.py          25 passed
tests/training/test_hardware.py         19 passed
tests/training/test_lengths.py          15 passed
tests/training/test_loader.py           14 passed
tests/training/test_models.py           19 passed
tests/training/test_run_repository.py   20 passed
tests/training/test_statistics.py        9 passed
tests/training/test_versions.py          9 passed
tests/training/test_gpu_integration.py   5 passed  (-m gpu)
```

The GPU suite covers spec §67's whole acceptance list in one end-to-end test, plus a check
that **the LoRA B matrices are no longer all zero** after training — a step that does not
move the weights has achieved nothing, and nothing else would catch that.

**A real bug the tests found.** The GPU suite initially used a session-scoped model
fixture, which held ~7 GiB for the whole run and starved the later full-job tests: they
failed the 6 GiB VRAM preflight rather than the thing they meant to test. The fixture is
now function-scoped with explicit `empty_cache()` teardown — slower, and correct.

## 34. Files created/modified

**Created:**

- `src/python_dpo/training/` — `__init__.py`, `errors.py`, `config.py`, `models.py`,
  `versions.py`, `hardware.py`, `dataset.py`, `lengths.py`, `loader.py`, `callbacks.py`,
  `trainer.py`, `verify.py`, `run_repository.py`, `statistics.py`, `README.md`
- `tests/training/` — nine offline test modules plus `test_gpu_integration.py`
- `configs/training/dpo_qlora.yaml` — the experiment configuration
- `data/preferences/runs/pref_20260818_074347_5eff/` — the validation-bearing dataset
- `data/training/runs/dpo_20260818_081231_a91d/` — the trained adapter and provenance
- `09_DPO_QLORA_TRAINING.md` (this file)

**Modified:**

- `src/python_dpo/config.py` + `config.yaml` + `data/training/.gitkeep` — the eighth data
  path (`training`), threaded through `_REQUIRED_PATH_KEYS`, `Paths` and `ensure_exists()`
- `src/python_dpo/cli.py` — the `train` command group; `preferences generate` gains
  `--split-ratios`
- `pyproject.toml` — the `training` extra and the `gpu` marker; `addopts` now deselects both
- `tests/test_no_heavy_imports.py` — `trl`/`peft`/`bitsandbytes`/`datasets` added to the
  guard, and every `training` submodule added to the probe
- `tests/test_project.py` — the eighth data path in all three enumerations; `train` CLI tests
- `tests/sandbox/test_config.py`, `tests/evaluation/test_config.py` — their fixture YAML
  gained the required `paths.training` key
- `.gitignore` — checkpoints, tokenizer snapshot and TRL's `adapter/ref/` excluded; the
  training log un-ignored against the global `*.log` rule
- `src/python_dpo/__init__.py` — version `0.8.0` → `0.9.0`
- `README.md`, `src/python_dpo/README.md`, `data/README.md`, `tests/README.md`

## 35. Dependencies added

The `training` extra: `trl>=1.10`, `peft>=0.20`, `bitsandbytes>=0.50`, `datasets>=5.0`,
alongside the existing torch/transformers/accelerate. Floors are set at the versions
actually verified here rather than the lowest plausible, because TRL 1.x changed the
`DPOTrainer`/`DPOConfig` surface materially from 0.x and a lower floor would resolve to an
API this code does not speak. Installing the extra downgraded nothing.

## 36. Deviations from the specification

- **`max_prompt_length` is not passed to the trainer** (§33) — TRL 1.10's `DPOConfig` has
  no such field. Retained as a length-analysis diagnostic.
- **`warmup_ratio` is converted to `warmup_steps`** (§53) — same reason.
- **A new preference dataset was generated** to obtain a non-empty validation split, since
  no previously committed dataset had one and none could, below ten pair-bearing problems.
- **An empty validation split is tolerated behind `--allow-small-dataset`**, running with
  `eval_strategy="no"`, rather than failing §24.9 outright — otherwise the two committed
  strict/margin datasets would be permanently untrainable. An empty *train* split remains
  always fatal.
- **`preferences generate` gained `--split-ratios`** — a small backward-compatible Stage 8
  amendment, so the dataset above is reproducible from the CLI rather than by editing
  `config.yaml`.
- **Training applies the base model's chat template** (§30, §31), because Stage 3
  generation demonstrably did (`qwen.py:_render`); training on the bare string would train
  on a format the candidates were never produced under.
- **Hyperparameters live in `configs/training/dpo_qlora.yaml`**, not the root
  `config.yaml` — per §62, and because `ModelConfig.quantization` still rejects any
  non-null value, so the root config cannot express 4-bit NF4 at all.
- **Checkpoints, the tokenizer snapshot and TRL's `adapter/ref/` are gitignored** while the
  trained adapter is tracked. `ref/` is a frozen fp32 reference copy (29 MB); the adapter
  was verified to reload and generate without it.
- **`train list` and `train show` were added** beyond §64/§74/§75, since none of the
  specified commands is usable without a way to discover a `training_run_id`.
- **A fifth copy of the run-directory plumbing**, rather than the shared base deferred at
  Stages 7 and 8. Extracting it would now touch five stages at once.
- **`--override-truncation` was added** so §35's threshold can be bypassed deliberately
  rather than only by editing the configured rate.

## 37. Known limitations

- **The dataset is three training records over two problems — a single optimizer step.**
  Every acceptance criterion in §107 is met, and the reward metrics at that step are
  identically zero because the policy still equals the reference. Stage 9 has demonstrated
  that the QLoRA/DPO stack loads, quantizes, attaches an adapter, takes a real gradient
  step, saves and reloads. It has demonstrated **nothing whatsoever** about whether the
  model writes better Python. That claim belongs to Step 10.
- **Resume is implemented but barely exercisable.** One step leaves no meaningful mid-run
  interruption point. The compatibility logic (§90–§92) is fully unit-tested; the
  end-to-end resume path is not exercised against a real interrupted run.
- **The OOM path (§83) is untested against a real OOM.** The message is assembled from
  configuration that is verified, but 4.66 GiB peak against 11.2 GiB free means the
  condition never arose.
- **`derive`-style provenance depends on Stage 8's manifest.** If a preference run's
  `manifest.json` lacks a provenance field the loader raises rather than guessing, which is
  correct but means a hand-edited Stage 8 run cannot be trained on.
- **Single GPU only**, by specification (§86). `distributed.enabled: true` is rejected
  rather than silently ignored.
