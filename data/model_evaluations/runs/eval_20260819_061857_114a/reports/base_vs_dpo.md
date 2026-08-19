# Python DPO Evaluation

Evaluation run: `eval_20260819_061857_114a`

## Executive Summary

DPO showed no measurable difference in pass@1 (50.0% for both models).

## Benchmark Description

- Benchmark: `python_eval_v1`
- Problems: 2
- Paired problems (evaluated for both variants): 2
- Base successfully evaluated: 2 / DPO successfully evaluated: 2

## Model Configuration

- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct` (revision: `default`)
- Adapter: `/home/rajiv/QwenCoder/QwenCoder/data/training/runs/dpo_20260819_061731_8314/adapter`
- Training run: `dpo_20260819_061731_8314`

## Generation Configuration

- base_seed: `1000`
- do_sample: `True`
- max_new_tokens: `512`
- num_samples: `2`
- repetition_penalty: `1.0`
- temperature: `0.2`
- top_p: `0.95`

## pass@k Results

| k | Base | DPO | 95% CI (Base) | 95% CI (DPO) |
|---|------|-----|----------------|----------------|
| 1 | 50.0% | 50.0% | [0.0%, 100.0%] | [0.0%, 100.0%] |

## Test Pass Rate

- Base mean test pass rate: 71.4%
- DPO mean test pass rate: 71.4%

Test failure distribution (Base): {'0%': 0, '1-20%': 0, '20-40%': 0, '40-60%': 2, '60-80%': 0, '80-99%': 0, '100%': 2}

Test failure distribution (DPO): {'0%': 0, '1-20%': 0, '20-40%': 0, '40-60%': 2, '60-80%': 0, '80-99%': 0, '100%': 2}

## Win/Tie/Loss

- DPO wins: 0
- Ties: 2
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

- Base Qwen: mean 5554 ms, p50 2538 ms, p95 8581 ms, 10.8 tokens/s
- DPO Qwen: mean 9368 ms, p50 4329 ms, p95 14502 ms, 6.4 tokens/s

## Memory

- Base Qwen peak GPU memory: 5.89 GiB
- DPO Qwen peak GPU memory: 7.26 GiB

## Failure Analysis

- Base Qwen: {'generation_error': 0, 'syntax_error': 0, 'import_error': 0, 'runtime_error': 0, 'assertion_failure': 2, 'timeout': 0, 'infrastructure_error': 0}
- DPO Qwen: {'generation_error': 0, 'syntax_error': 0, 'import_error': 0, 'runtime_error': 0, 'assertion_failure': 2, 'timeout': 0, 'infrastructure_error': 0}

## Conclusions

DPO showed no measurable difference in pass@1 (50.0% for both models).

This benchmark contains only 2 problem(s). Per spec section 144, a benchmark this size is suitable for pipeline validation, not for reliable model-performance conclusions.

