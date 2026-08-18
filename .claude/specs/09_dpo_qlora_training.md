# Python DPO Data Generation Pipeline

## Step 9 — Qwen Coder DPO/QLoRA Training

**Specification Version:** 1.0
**Status:** Implementation Specification
**Step:** 9 of 12
**Depends On:** Step 1 — Project Skeleton
**Depends On:** Step 2 — Python Problem Dataset
**Depends On:** Step 3 — Qwen Candidate Generator
**Depends On:** Step 4 — Candidate Persistence
**Depends On:** Step 5 — Docker Sandbox
**Depends On:** Step 6 — Candidate Test Executor
**Depends On:** Step 7 — Candidate Evaluation and Ranking
**Depends On:** Step 8 — DPO Preference Pair Generation

---

# 1. Objective

Implement the model-training stage that fine-tunes a Qwen Coder model using the preference dataset generated in Step 8.

The initial implementation must use:

```
QLoRA
+
DPO
+
Hugging Face Transformers
+
TRL
+
PEFT
+
bitsandbytes
```

The objective is to train a lightweight LoRA adapter that improves Python programming behavior while keeping the base Qwen model frozen.

---

# 2. Core Training Architecture

The training pipeline must be:

```
DPO preference dataset
         │
         ▼
   Dataset Loader
         │
         ▼
   Schema Validator
         │
         ▼
    Tokenizer
         │
         ▼
  4-bit Qwen Model
         │
         ▼
    LoRA Adapter
         │
         ▼
    DPOTrainer
         │
         ▼
  LoRA Adapter
         │
         ▼
  Training Artifact
         │
         ▼
  Evaluation / Step 10
```

---

# 3. Critical Design Decision

The base model must remain frozen.

Only LoRA parameters are trainable.

Conceptually:

```
Qwen base model
     │
     ├── frozen weights
     │
     └── LoRA adapters
            │
            ▼
         trainable
```

Do NOT implement full-model fine-tuning in Step 9.

---

# 4. Training Method

The default training method is:

```
DPO
```

using:

```
QLoRA
```

The implementation must use the Hugging Face TRL DPO training stack.

Current TRL exposes `DPOTrainer`, `DPOConfig`, PEFT integration, and quantization configuration for this workflow.

---

# 5. Supported Base Model

The base model must be configurable.

Example:

```
model:
  name: "Qwen/<configured-coder-model>"
```

Do NOT hard-code a model name throughout the source code.

The model name must appear only in configuration.

---

# 6. Model Compatibility

Before training, verify:

* model exists
* tokenizer exists
* model is causal language model
* model supports the configured sequence length
* model supports the selected quantization mode
* model supports PEFT/LoRA
* model has the expected architecture

If any requirement fails:

```
training must stop before modifying artifacts.
```

---

# 7. Model Revision

Support an optional:

```
revision
```

field.

Example:

```
model:
  name: "Qwen/..."
  revision: "<commit-or-tag>"
```

For reproducibility, record the exact resolved model revision.

---

# 8. Quantization

The default configuration must use:

```
4-bit quantization
```

with:

```
NF4
```

and:

```
bfloat16 compute
```

when the GPU supports BF16.

Hugging Face documents NF4 as the recommended 4-bit type for training quantized models in the QLoRA workflow.

---

# 9. Quantization Configuration

Conceptually:

```
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

The exact options must be configurable.

---

# 10. Compute Dtype

Default:

```
bfloat16
```

Fallback:

```
float16
```

The implementation must detect whether BF16 is supported.

Do not blindly request BF16 on unsupported hardware.

---

# 11. RTX 3060 Requirement

The user's RTX 3060 has 12 GB VRAM.

The training configuration must therefore be conservative.

The default configuration must be designed to fit within approximately:

```
12 GB VRAM
```

where practical.

Do not assume that the full Qwen model can fit unquantized.

---

# 12. VRAM Awareness

Before training, perform a hardware check.

Record:

```
GPU name
GPU count
VRAM
CUDA version
PyTorch CUDA version
compute capability
```

If the configured model/training configuration is clearly incompatible with available VRAM:

```
fail early
```

with a useful error message.

---

# 13. Hardware CLI

Add:

```
python -m python_dpo train hardware-check
```

Output:

```
GPU:
VRAM:
CUDA:
PyTorch:
BF16 support:
4-bit support:
Device count:
```

---

# 14. LoRA

Use PEFT LoRA.

Initial configuration should be configurable:

```
lora:
  r: 16
  alpha: 32
  dropout: 0.05
```

Do not hard-code these values.

---

# 15. LoRA Target Modules

The target modules must be configurable.

For a Qwen architecture, the implementation should inspect the model and verify the configured module names exist.

Do not assume that every Qwen model has exactly the same module naming.

A typical initial configuration may target attention projections such as:

```
q_proj
k_proj
v_proj
o_proj
```

Optionally later:

```
gate_proj
up_proj
down_proj
```

The first implementation should start with attention projections unless the selected Qwen model requires otherwise.

---

# 16. Target Module Validation

Before training:

```
configured_target_modules
```

must be checked against:

```
model.named_modules()
```

If none of the configured modules exist:

```
fail immediately.
```

Do not silently train with zero LoRA target modules.

---

# 17. LoRA Bias

Default:

```
bias = "none"
```

Make configurable.

---

# 18. LoRA Task Type

Use:

```
CAUSAL_LM
```

unless the selected architecture requires another supported setting.

---

# 19. Trainable Parameter Check

Before training, calculate:

```
total_parameters
trainable_parameters
trainable_percentage
```

Example:

```
Total parameters:       X
Trainable parameters:   Y
Trainable percentage:   Z%
```

The trainable percentage must be substantially smaller than the total parameter count.

---

# 20. Critical Safety Check

Before starting training:

```
if trainable_parameters == total_parameters:

    FAIL
```

The system must never accidentally perform full-model training under the QLoRA configuration.

---

# 21. Dataset Input

The training dataset comes from Step 8:

```
train.jsonl
validation.jsonl
test.jsonl
```

The DPO trainer should receive:

```
train
validation
```

The test set must NOT be used during training.

---

# 22. Test Set Isolation

The test dataset must remain completely isolated from training.

Do not:

* tune hyperparameters against test results
* select checkpoints based on test results
* modify the training configuration based on test results

The test set is reserved for Step 10.

---

# 23. Dataset Schema

Training records must contain:

```
prompt
chosen
rejected
```

Example:

```
{
  "prompt": "Write a Python function...",
  "chosen": "def solution(...): ...",
  "rejected": "def solution(...): ..."
}
```

The training loader must validate this schema.

---

# 24. Dataset Validation

Before loading the model:

1. Verify dataset exists.
2. Verify JSONL syntax.
3. Verify required fields.
4. Verify non-empty prompt.
5. Verify non-empty chosen.
6. Verify non-empty rejected.
7. Verify chosen != rejected.
8. Verify dataset split exists.
9. Verify train and validation are non-empty.

If validation fails:

```
do not load the model.
```

---

# 25. Dataset Manifest

The training run must reference the exact preference dataset.

Record:

```
preference_run_id
preference_version
selection_policy
dataset_schema_version
```

Do not copy the dataset blindly without preserving its provenance.

---

# 26. Dataset Hash

Calculate SHA-256 hashes for:

```
train.jsonl
validation.jsonl
test.jsonl
```

Store them in:

```
manifest.json
```

This ensures that the training run can later be reproduced against the same dataset.

---

# 27. Dataset Statistics

Before training report:

```
training examples
validation examples
test examples
unique problems
average prompt length
average chosen length
average rejected length
maximum prompt length
maximum chosen length
maximum rejected length
```

Also report tokenized lengths after tokenizer loading.

---

# 28. Tokenizer

Load the tokenizer associated with the exact base model.

Do not use a tokenizer from another Qwen model.

Record:

```
tokenizer name
tokenizer revision
vocab size
special tokens
```

---

# 29. Padding

The tokenizer must have an appropriate:

```
pad_token
```

If no pad token exists, use the model's EOS token only when appropriate and explicitly record the decision.

Current TRL's `DPOTrainer` expects a processing class/tokenizer with a padding token configured; the documentation also specifies left padding for the processing class.

---

# 30. Chat Template

Do not manually reproduce the Qwen chat template.

Use the tokenizer/model's configured chat template where applicable.

The training pipeline must preserve consistency between:

```
prompt formatting
```

and:

```
model inference formatting.
```

---

# 31. Prompt Format

The dataset's prompt should remain model-agnostic.

The training layer is responsible for applying the selected model's conversational template where required.

Do not permanently embed Qwen-specific formatting into Step 8 artifacts.

---

# 32. Response Format

The `chosen` and `rejected` fields represent model responses.

They must be passed to the DPO training pipeline as responses rather than as independent prompts.

---

# 33. Sequence Length

Make sequence lengths configurable.

Example:

```
training:
  max_length: 1024
  max_prompt_length: 512
```

The initial RTX 3060 configuration should use conservative lengths.

---

# 34. Length Analysis

Before training, calculate the token-length distribution:

```
prompt
chosen
rejected
prompt + chosen
prompt + rejected
```

Report:

```
p50
p90
p95
p99
max
```

This is important before choosing `max_length`.

---

# 35. Truncation

Do not silently truncate large examples.

Report:

```
truncated_examples
```

and:

```
truncation_rate
```

If truncation exceeds a configured threshold:

```
training should fail validation
```

unless explicitly overridden.

---

# 36. Initial Truncation Threshold

Default:

```
max_truncation_rate = 0.05
```

If more than 5% of examples require truncation:

```
fail preflight
```

with a recommendation to adjust sequence lengths or clean the dataset.

---

# 37. Batch Size

For a 12 GB GPU, default conservatively:

```
per_device_train_batch_size: 1
```

Use:

```
gradient_accumulation_steps
```

to obtain an effective batch size.

---

# 38. Effective Batch Size

Calculate:

```
effective_batch_size =
    per_device_train_batch_size
    × gradient_accumulation_steps
    × number_of_devices
```

Record this value.

---

# 39. Initial Gradient Accumulation

Default:

```
gradient_accumulation_steps: 8
```

This must be configurable.

Do not assume this value is optimal.

---

# 40. Learning Rate

For LoRA/adapter training, use a configurable learning rate.

Initial default:

```
1e-5
```

Current TRL documentation notes that adapter training typically uses a higher learning rate than full fine-tuning, around 1e-5 as an example. Treat this as an initial experiment rather than a universal optimum.

---

# 41. Learning Rate Configuration

Example:

```
training:
  learning_rate: 1e-5
```

Do not hard-code the value in Python.

---

# 42. Number of Epochs

Initial default:

```
num_train_epochs: 1
```

Because the preference dataset is generated specifically for this task, overfitting is a significant risk.

Support:

```
1
2
3
```

as initial experimental values.

---

# 43. Training Steps

Allow:

```
max_steps
```

to override epoch-based training.

If:

```
max_steps > 0
```

use it instead of full epoch calculation.

---

# 44. DPO Beta

Expose:

```
dpo:
  beta: 0.1
```

as the initial default.

The implementation must make beta configurable.

Do not treat 0.1 as universally optimal.

---

# 45. DPO Loss

Default:

```
sigmoid
```

Use the standard DPO loss initially.

Do not introduce MPO or alternative DPO loss combinations in Step 9.

Current TRL supports multiple loss configurations, but this project should establish a simple baseline first.

---

# 46. Reference Model

The implementation must explicitly support:

```
reference model
```

and:

```
reference-free / implicit reference
```

depending on the TRL configuration.

For the initial QLoRA experiment, use the configuration recommended by the installed TRL version for PEFT-based DPO.

Do not instantiate unnecessary duplicate full-precision models on a 12 GB GPU.

---

# 47. Memory Requirement

The training implementation must avoid unnecessarily loading:

```
base model
reference model
optimizer states
```

multiple times in full precision.

Memory usage must be monitored.

---

# 48. Gradient Checkpointing

Enable:

```
gradient_checkpointing = true
```

by default for the RTX 3060 configuration.

This trades computation for lower activation memory.

---

# 49. Gradient Checkpointing Compatibility

If gradient checkpointing requires:

```
use_cache = false
```

configure it explicitly during training.

Restore appropriate inference settings after training if necessary.

---

# 50. Optimizer

Use a memory-efficient optimizer where compatible.

Initial preference:

```
paged_adamw_8bit
```

if supported by the installed stack.

Otherwise fall back to:

```
AdamW
```

The selected optimizer must be recorded.

---

# 51. Mixed Precision

Use:

```
bf16
```

when supported.

Otherwise:

```
fp16
```

Do not enable both simultaneously.

---

# 52. Gradient Clipping

Configure:

```
max_grad_norm: 1.0
```

as the initial default.

Make configurable.

---

# 53. Warmup

Use:

```
warmup_ratio: 0.05
```

as the initial default.

Make configurable.

---

# 54. Scheduler

Use:

```
cosine
```

or:

```
linear
```

as a configurable scheduler.

Initial default:

```
cosine
```

Record the scheduler.

---

# 55. Logging

Training must log:

```
loss
learning_rate
reward metrics if provided by DPOTrainer
chosen/rejected reward-related metrics if available
epoch
step
GPU memory
```

Do not log candidate code at every step.

---

# 56. Evaluation During Training

Run validation at a configurable frequency.

Example:

```
evaluation_strategy: "steps"
```

or the equivalent supported by the installed TRL/Transformers version.

Do not use the test set during training.

---

# 57. Checkpointing

Save checkpoints periodically.

Example:

```
save_steps: 100
```

The exact default must be configurable.

---

# 58. Best Checkpoint

The training pipeline should support selecting the best checkpoint based on validation metrics.

However, do NOT automatically assume:

```
lowest DPO loss
```

means:

```
best Python programming performance.
```

The actual programming benchmark evaluation occurs in Step 10.

Therefore the initial implementation should preserve all checkpoints or allow explicit selection rather than aggressively deleting them.

---

# 59. Early Stopping

Do not enable early stopping by default in v1.

The dataset is initially small and validation metrics may be noisy.

Make it configurable for later experiments.

---

# 60. Training Seed

Set:

```
seed: 42
```

by default.

Persist the seed.

Also configure:

```
data_seed
```

when supported.

---

# 61. Determinism

The training pipeline should attempt reproducibility.

Record:

```
seed
Python version
PyTorch version
Transformers version
TRL version
PEFT version
bitsandbytes version
CUDA version
GPU
model revision
dataset hashes
```

Do not claim bit-for-bit determinism unless it has been verified.

---

# 62. Experiment Configuration

Create:

```
configs/training/dpo_qlora.yaml
```

Example:

```
model:
  name: "Qwen/<model>"
  revision: null

quantization:
  enabled: true
  bits: 4
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

dpo:
  beta: 0.1
  loss_type: sigmoid

training:
  max_length: 1024
  max_prompt_length: 512
  learning_rate: 1.0e-5
  num_train_epochs: 1
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  warmup_ratio: 0.05
  max_grad_norm: 1.0
  seed: 42

optimizer:
  name: paged_adamw_8bit

dataset:
  preference_run_id: null
```

---

# 63. Configuration Overrides

Support CLI overrides.

Example:

```
--learning-rate 2e-5
```

or:

```
--config configs/training/dpo_qlora.yaml
```

Do not require editing Python source to change hyperparameters.

---

# 64. Training CLI

Add:

```
python -m python_dpo train dpo \
    --config configs/training/dpo_qlora.yaml \
    --preference-run-id PREF_RUN_ID
```

---

# 65. Dry Run

Support:

```
--dry-run
```

The dry run must:

1. Validate hardware.
2. Load tokenizer.
3. Validate dataset.
4. Calculate token-length statistics.
5. Load model.
6. Apply quantization.
7. Apply LoRA.
8. Report trainable parameters.
9. Estimate memory.
10. NOT start training.

---

# 66. Smoke Test

Support:

```
--smoke-test
```

The smoke test should use a tiny subset:

```
2–10 training examples
```

and:

```
1–2 validation examples
```

Train for:

```
1–5 steps
```

The objective is to verify that the complete stack works.

---

# 67. Smoke Test Acceptance

The smoke test must successfully:

```
load model
load tokenizer
load dataset
apply QLoRA
initialize DPOTrainer
perform forward pass
perform backward pass
update LoRA parameters
save checkpoint
reload adapter
```

If any of these fail:

```
do not run the full training job.
```

---

# 68. Training Run ID

Every training execution must have:

```
training_run_id
```

Example:

```
dpo_20260818_101500_a91f
```

---

# 69. Training Directory

Use:

```
data/training/runs/<training_run_id>/
```

Structure:

```
<training_run_id>/
    manifest.json
    config.yaml
    dataset_manifest.json
    hardware.json
    tokenizer/
    checkpoints/
    adapter/
    metrics/
    logs/
    evaluation/
    final_report.json
```

---

# 70. Training Manifest

Persist:

```
training_run_id
model_name
model_revision
tokenizer_revision
preference_run_id
dataset hashes
ranking run ID
evaluation run ID
hardware
configuration
package versions
random seeds
start time
end time
```

---

# 71. Package Version Capture

Capture:

```
torch
transformers
trl
peft
bitsandbytes
accelerate
datasets
safetensors
```

Also capture:

```
CUDA runtime
driver version
```

---

# 72. Model Artifact

Do not copy the entire base model into every training run.

Store:

```
model identifier
revision
checksum/revision information
```

The training artifact should primarily contain the LoRA adapter.

---

# 73. Adapter Output

Save the final adapter using PEFT-compatible format.

Example:

```
adapter/
    adapter_config.json
    adapter_model.safetensors
```

The exact files depend on the PEFT version.

---

# 74. Adapter Reload Test

After training:

1. Load base model.
2. Load saved LoRA adapter.
3. Load tokenizer.
4. Run a test prompt.
5. Verify generation succeeds.

This is mandatory.

---

# 75. Base + Adapter Inference

Support:

```
python -m python_dpo train inference \
    --training-run-id TRAINING_RUN_ID \
    --prompt "Write a Python function..."
```

This should load:

```
base Qwen model
+
trained LoRA adapter
```

and generate a response.

---

# 76. Adapter Merge

Do NOT merge LoRA weights into the base model by default.

The initial artifact should remain:

```
base model
+
adapter
```

A separate optional merge command may be implemented later.

---

# 77. Merge Command

If implemented:

```
python -m python_dpo train merge \
    --training-run-id TRAINING_RUN_ID
```

must create a separate artifact.

Never overwrite the adapter.

---

# 78. Training Metrics

Persist metrics in machine-readable format.

At minimum:

```
step
epoch
train_loss
eval_loss
learning_rate
```

Where supplied by DPOTrainer, also preserve relevant DPO metrics such as:

```
rewards/chosen
rewards/rejected
rewards/accuracies
rewards/margins
```

The exact metric names may vary with the installed TRL version.

---

# 79. DPO Reward Metrics

The training report should track whether:

```
chosen reward > rejected reward
```

is increasing.

The following metric is especially useful:

```
reward_margin
```

Conceptually:

```
reward_margin =
    chosen_reward - rejected_reward
```

Do not confuse this with the original test pass-rate margin from Step 8.

They are different quantities.

---

# 80. Training Loss

Track:

```
train_loss
```

but do not use it as the only measure of success.

A lower DPO loss does not necessarily mean better Python programming ability.

---

# 81. Validation Loss

Track:

```
eval_loss
```

but do not use it as the final model-selection criterion.

Step 10 must evaluate actual Python programming performance.

---

# 82. Training Failure Handling

If training fails:

```
mark training run as failed
```

Record:

```
error_type
error_message
stack trace
last completed step
```

Do not mark the adapter as final unless it successfully saved and passed reload validation.

---

# 83. Out-of-Memory Handling

If CUDA OOM occurs:

Report:

```
model
sequence length
batch size
gradient accumulation
LoRA configuration
GPU memory
```

Do not silently change configuration and restart.

A future automated tuner may do this, but v1 should remain deterministic.

---

# 84. Memory Monitoring

At training start and periodically record:

```
allocated GPU memory
reserved GPU memory
peak allocated memory
peak reserved memory
```

Use PyTorch CUDA APIs where available.

---

# 85. OOM Prevention

Before training, verify:

```
batch_size
sequence_length
quantization
gradient_checkpointing
```

are consistent with the configured GPU.

The preflight should warn when the configuration is aggressive.

---

# 86. Single-GPU Initial Scope

Step 9 v1 must support:

```
one GPU
```

only.

Do not implement distributed training yet.

The code should not assume:

```
torchrun
DDP
FSDP
```

are required.

---

# 87. Multi-GPU Architecture

The configuration may contain:

```
distributed:
  enabled: false
```

but distributed training must not be implemented in Step 9.

It can be introduced later.

---

# 88. Gradient Accumulation Verification

Log:

```
micro_batch_size
gradient_accumulation_steps
effective_batch_size
```

This prevents confusion when interpreting training logs.

---

# 89. Checkpoint Validation

Every checkpoint should contain enough metadata to identify:

```
training_run_id
step
epoch
base model
LoRA configuration
```

---

# 90. Checkpoint Resume

Support:

```
--resume-from-checkpoint
```

The resumed run must verify that:

```
model
tokenizer
dataset
LoRA configuration
```

are compatible with the checkpoint.

---

# 91. Dataset Compatibility on Resume

If the dataset hash differs from the original training dataset:

```
refuse resume
```

unless:

```
--force-resume
```

is explicitly provided.

---

# 92. Training Configuration Compatibility

If critical configuration differs:

```
LoRA r
target modules
quantization mode
base model
tokenizer
```

do not silently resume.

Fail with a compatibility error.

---

# 93. Training Logs

Store:

```
logs/training.log
```

and machine-readable:

```
metrics/metrics.jsonl
```

Do not depend solely on console output.

---

# 94. Final Training Report

Create:

```
final_report.json
```

with:

```
training_run_id
model
dataset
number_of_examples
epochs
steps
final_train_loss
final_eval_loss
peak_gpu_memory
trainable_parameters
total_parameters
adapter_path
checkpoint_path
training_duration
status
```

---

# 95. Model Evaluation Boundary

Step 9 must NOT claim:

```
Python accuracy improved
```

based only on:

```
DPO loss
reward margin
eval loss
```

Actual programming evaluation belongs to Step 10.

---

# 96. Training Dataset Variants

The pipeline must support training different preference datasets without changing the training code.

Examples:

```
strict_v1

margin_0.2_v1

all_better_v1
```

The training run manifest must record which preference dataset was used.

---

# 97. Baseline Model

Before training, Step 9 should support running a baseline inference test using the original Qwen model.

Record:

```
baseline_model_response
```

for a small fixed smoke-test prompt set.

Do not use this as the actual benchmark.

Step 10 will provide the proper evaluation.

---

# 98. Training Experiment Naming

Support:

```
experiment_name
```

Example:

```
qwen-python-dpo-strict-v1
```

This should appear in:

```
logs
checkpoints
manifest
final report
```

---

# 99. Reproducibility

A training run must be reconstructible from:

```
base model revision
tokenizer revision
preference dataset hash
training config
package versions
GPU information
seed
```

Do not depend on undocumented environment settings.

---

# 100. Security

The training process must NOT execute candidate code.

Candidate execution belongs exclusively to:

```
Step 5
Step 6
```

The training process reads:

```
prompt
chosen
rejected
```

as text.

It must never:

```
exec(chosen)
exec(rejected)
```

or otherwise execute generated code.

---

# 101. Training Data Integrity

Before training, validate that:

```
chosen
rejected
```

are ordinary text strings.

Do not interpret them as Python programs during training.

---

# 102. Dataset Leakage

Verify that:

```
train
validation
test
```

remain problem-disjoint.

Training must abort if the same problem ID appears across splits according to the Step 8 split manifest.

---

# 103. Small Dataset Warning

If:

```
number_of_training_pairs < configured_minimum
```

emit a warning.

Initial suggested threshold:

```
500
```

but do not necessarily prevent smoke testing.

For the current 10-problem development dataset, expect this warning.

---

# 104. Production Training Gate

Full training should require explicit confirmation when:

```
dataset is extremely small
```

For example:

```
WARNING:
Training dataset contains only 42 preference pairs.
This is suitable for pipeline validation but not for
meaningful model adaptation.
```

The CLI should allow:

```
--allow-small-dataset
```

for experimentation.

---

# 105. Training Dataset Balance

Report:

```
average chosen score
average rejected score
average score margin
strong-pair percentage
medium-pair percentage
```

This allows the training dataset to be understood before training.

---

# 106. Preference Distribution

Report:

```
chosen pass-rate histogram
rejected pass-rate histogram
score-margin histogram
```

Do not require visualization in Step 9.

Machine-readable statistics are sufficient.

---

# 107. Acceptance Criteria

Step 9 is complete only when:

* [ ] Qwen base model is configurable.
* [ ] Model revision is recorded.
* [ ] Tokenizer is loaded from the correct model.
* [ ] Preference dataset is validated.
* [ ] Dataset hashes are recorded.
* [ ] Train/validation/test isolation is verified.
* [ ] 4-bit quantization works.
* [ ] NF4 configuration works.
* [ ] BF16/FP16 fallback works.
* [ ] LoRA is applied.
* [ ] Target modules are validated.
* [ ] Only LoRA parameters are trainable.
* [ ] Trainable parameter count is recorded.
* [ ] DPOTrainer initializes.
* [ ] Smoke test completes.
* [ ] Forward pass works.
* [ ] Backward pass works.
* [ ] LoRA parameters update.
* [ ] Checkpoint saves.
* [ ] Adapter saves.
* [ ] Adapter reload succeeds.
* [ ] Training metrics are persisted.
* [ ] Hardware metrics are persisted.
* [ ] Training manifest is persisted.
* [ ] Training is resumable.
* [ ] Dataset compatibility is checked on resume.
* [ ] CUDA OOM is reported clearly.
* [ ] No candidate code executes during training.
* [ ] No test set is used for training.
* [ ] No distributed training is implemented.
* [ ] All unit tests pass.
* [ ] Smoke test passes.
* [ ] Adapter reload test passes.

---

# 108. Verification Procedure

## Step 1 — Hardware check

Run:

```
python -m python_dpo train hardware-check
```

Verify:

```
GPU detected
CUDA available
sufficient VRAM
BF16 capability reported
```

---

## Step 2 — Dataset validation

Run:

```
python -m python_dpo preferences validate \
    --preference-run-id PREF_RUN_ID
```

---

## Step 3 — Training dry run

Run:

```
python -m python_dpo train dpo \
    --config configs/training/dpo_qlora.yaml \
    --preference-run-id PREF_RUN_ID \
    --dry-run
```

Verify:

```
tokenizer
dataset
quantization
LoRA
parameter counts
sequence lengths
```

---

## Step 4 — Smoke test

Run:

```
python -m python_dpo train dpo \
    --config configs/training/dpo_qlora.yaml \
    --preference-run-id PREF_RUN_ID \
    --smoke-test
```

The smoke test must complete successfully.

---

## Step 5 — Full training

Only after the smoke test passes:

```
python -m python_dpo train dpo \
    --config configs/training/dpo_qlora.yaml \
    --preference-run-id PREF_RUN_ID
```

---

## Step 6 — Adapter reload

Run:

```
python -m python_dpo train verify \
    --training-run-id TRAINING_RUN_ID
```

Expected:

```
Adapter reload successful.
```

---

# 109. Expected Artifacts

After Step 9:

```
data/
└── training/
    └── runs/
        └── <training_run_id>/
            ├── manifest.json
            ├── config.yaml
            ├── dataset_manifest.json
            ├── hardware.json
            ├── tokenizer/
            ├── checkpoints/
            │   ├── checkpoint-...
            │   └── ...
            ├── adapter/
            │   ├── adapter_config.json
            │   └── adapter_model.safetensors
            ├── metrics/
            │   └── metrics.jsonl
            ├── logs/
            │   └── training.log
            └── final_report.json
```

---

# 110. What Step 9 Produces

Step 9 produces:

```
Qwen base model
    +
trained LoRA adapter
```

It does NOT yet produce the final claim:

```
"Python coding performance improved."
```

That must be established by Step 10.

---

# 111. Final Implementation Report

After implementation, report:

1. Base Qwen model.
2. Model revision.
3. Tokenizer.
4. Python version.
5. PyTorch version.
6. Transformers version.
7. TRL version.
8. PEFT version.
9. bitsandbytes version.
10. CUDA version.
11. GPU model.
12. GPU VRAM.
13. Quantization configuration.
14. LoRA configuration.
15. Target modules.
16. Total parameters.
17. Trainable parameters.
18. Trainable percentage.
19. Dataset name.
20. Dataset hash.
21. Number of training examples.
22. Number of validation examples.
23. Sequence-length statistics.
24. Truncation statistics.
25. DPO beta.
26. Learning rate.
27. Batch size.
28. Gradient accumulation.
29. Effective batch size.
30. Number of epochs.
31. Number of steps.
32. Final training loss.
33. Final validation loss.
34. DPO reward metrics.
35. Peak GPU memory.
36. Training duration.
37. Checkpoint path.
38. Adapter path.
39. Adapter reload result.
40. Smoke-test result.
41. Any deviations from the specification.
42. Known limitations.

Do NOT implement Step 10 automatically.

Wait for explicit approval before implementing the programming benchmark and trained-vs-base model evaluation.
