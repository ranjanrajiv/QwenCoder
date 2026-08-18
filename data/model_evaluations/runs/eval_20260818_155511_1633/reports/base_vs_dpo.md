# Python DPO Evaluation

Evaluation run: `eval_20260818_155511_1633`

## Executive Summary

DPO improved pass@1 from 77.1% to 78.6%, an absolute improvement of 1.4 percentage points.

## Benchmark Description

- Benchmark: `python_eval_v1`
- Problems: 7
- Paired problems (evaluated for both variants): 7
- Base successfully evaluated: 7 / DPO successfully evaluated: 7

## Model Configuration

- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct` (revision: `default`)
- Adapter: `/home/rajiv/QwenCoder/QwenCoder/data/training/runs/dpo_20260818_081231_a91d/adapter`
- Training run: `dpo_20260818_081231_a91d`

## Generation Configuration

- base_seed: `1000`
- do_sample: `True`
- max_new_tokens: `512`
- num_samples: `10`
- repetition_penalty: `1.0`
- temperature: `0.2`
- top_p: `0.95`

## pass@k Results

| k | Base | DPO | 95% CI (Base) | 95% CI (DPO) |
|---|------|-----|----------------|----------------|
| 1 | 77.1% | 78.6% | [48.6%, 98.6%] | [50.0%, 100.0%] |
| 5 | 85.7% | 85.7% | [57.1%, 100.0%] | [57.1%, 100.0%] |
| 10 | 85.7% | 85.7% | [57.1%, 100.0%] | [57.1%, 100.0%] |

## Test Pass Rate

- Base mean test pass rate: 90.2%
- DPO mean test pass rate: 90.2%

Test failure distribution (Base): {'0%': 0, '1-20%': 0, '20-40%': 0, '40-60%': 10, '60-80%': 3, '80-99%': 3, '100%': 54}

Test failure distribution (DPO): {'0%': 0, '1-20%': 0, '20-40%': 0, '40-60%': 10, '60-80%': 4, '80-99%': 1, '100%': 55}

## Win/Tie/Loss

- DPO wins: 0
- Ties: 7
- DPO losses: 0
- DPO win rate (ties excluded): 0.0%

## Statistical Analysis

- Paired bootstrap, pass@1 difference (DPO - Base): +1.4 pp, 95% CI [+0.0, +4.3] pp (1000 iterations, seed 42)
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

- Base Qwen: mean 1420 ms, p50 1233 ms, p95 2433 ms, 47.2 tokens/s
- DPO Qwen: mean 2183 ms, p50 1914 ms, p95 3826 ms, 30.5 tokens/s

## Memory

- Base Qwen peak GPU memory: 2.00 GiB
- DPO Qwen peak GPU memory: 3.37 GiB

## Failure Analysis

- Base Qwen: {'generation_error': 0, 'syntax_error': 0, 'import_error': 0, 'runtime_error': 0, 'assertion_failure': 16, 'timeout': 0, 'infrastructure_error': 0}
- DPO Qwen: {'generation_error': 0, 'syntax_error': 0, 'import_error': 0, 'runtime_error': 0, 'assertion_failure': 15, 'timeout': 0, 'infrastructure_error': 0}

## Conclusions

DPO improved pass@1 from 77.1% to 78.6%, an absolute improvement of 1.4 percentage points.

This benchmark contains only 7 problem(s). Per spec section 144, a benchmark this size is suitable for pipeline validation, not for reliable model-performance conclusions.

