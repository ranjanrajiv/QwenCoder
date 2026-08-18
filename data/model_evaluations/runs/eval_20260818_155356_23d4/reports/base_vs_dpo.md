# Python DPO Evaluation

Evaluation run: `eval_20260818_155356_23d4`

## Executive Summary

DPO showed no measurable difference in pass@1 (66.7% for both models).

## Benchmark Description

- Benchmark: `python_eval_v1`
- Problems: 3
- Paired problems (evaluated for both variants): 3
- Base successfully evaluated: 3 / DPO successfully evaluated: 3

## Model Configuration

- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct` (revision: `default`)
- Adapter: `/home/rajiv/QwenCoder/QwenCoder/data/training/runs/dpo_20260818_081231_a91d/adapter`
- Training run: `dpo_20260818_081231_a91d`

## Generation Configuration

- base_seed: `1000`
- do_sample: `True`
- max_new_tokens: `512`
- num_samples: `1`
- repetition_penalty: `1.0`
- temperature: `0.2`
- top_p: `0.95`

## pass@k Results

| k | Base | DPO | 95% CI (Base) | 95% CI (DPO) |
|---|------|-----|----------------|----------------|
| 1 | 66.7% | 66.7% | [0.0%, 100.0%] | [0.0%, 100.0%] |

## Test Pass Rate

- Base mean test pass rate: 81.0%
- DPO mean test pass rate: 81.0%

Test failure distribution (Base): {'0%': 0, '1-20%': 0, '20-40%': 0, '40-60%': 1, '60-80%': 0, '80-99%': 0, '100%': 2}

Test failure distribution (DPO): {'0%': 0, '1-20%': 0, '20-40%': 0, '40-60%': 1, '60-80%': 0, '80-99%': 0, '100%': 2}

## Win/Tie/Loss

- DPO wins: 0
- Ties: 3
- DPO losses: 0
- DPO win rate (ties excluded): 0.0%

## Statistical Analysis

- Paired bootstrap, pass@1 difference (DPO - Base): +0.0 pp, 95% CI [+0.0, +0.0] pp (1000 iterations, seed 42)
- McNemar's exact test: base-only=0, dpo-only=0, p=1.0000

## Regression Analysis

- pass_at_1_improves: False
- pass_at_5_not_regressed: True
- syntax_success_not_regressed: True
- timeout_rate_not_increased: True
- paired_ci_supports_improvement: True
- catastrophic_regression_detected: False
- overfitting_check: not_applicable: Stage 10 does not evaluate training-set performance (spec section 69)
- **DPO_SUCCESS: False**

## Latency

- Base Qwen: mean 2166 ms, p50 2060 ms, p95 2462 ms, 36.9 tokens/s
- DPO Qwen: mean 2708 ms, p50 3003 ms, p95 3891 ms, 29.5 tokens/s

## Memory

- Base Qwen peak GPU memory: 1.99 GiB
- DPO Qwen peak GPU memory: 3.37 GiB

## Failure Analysis

- Base Qwen: {'generation_error': 0, 'syntax_error': 0, 'import_error': 0, 'runtime_error': 0, 'assertion_failure': 1, 'timeout': 0, 'infrastructure_error': 0}
- DPO Qwen: {'generation_error': 0, 'syntax_error': 0, 'import_error': 0, 'runtime_error': 0, 'assertion_failure': 1, 'timeout': 0, 'infrastructure_error': 0}

## Conclusions

DPO showed no measurable difference in pass@1 (66.7% for both models).

This benchmark contains only 3 problem(s). Per spec section 144, a benchmark this size is suitable for pipeline validation, not for reliable model-performance conclusions.

