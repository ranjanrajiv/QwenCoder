# Error Analysis — analysis_20260819_084641_6e74

## Headline

**Iteration decision: `insufficient_evidence`**

- benchmark has 7 problem(s), below the configured minimum of 30

> This decision gates every other finding below. The analyses still ran and their numbers are reported, but the evidence does not meet the configured minimum, so none of them is offered as a conclusion about model quality.

## Lineage

- `evaluation_run_id`: eval_20260818_155511_1633
- `training_run_id`: dpo_20260818_081231_a91d
- `preference_run_id`: pref_20260818_074347_5eff
- `ranking_run_id`: rank_20260817_161726_a84d
- `candidate_run_id`: 20260817_055411
- `sandbox_evaluation_run_id`: eval_20260817_115154_dcd4

## Outcomes

7 problem(s) analysed — 0 improved, 0 regressed, 7 unchanged.

## Error profiles

| Variant | Samples | Passed | Failure categories |
|---|---|---|---|
| base | 70 | 54 | {'assertion_failure': 16} |
| dpo | 70 | 55 | {'assertion_failure': 15} |

## Failure subcategories

- **base** — likely failure: `AssertionError` (48 occurrence(s))
- **dpo** — likely failure: `AssertionError` (49 occurrence(s))

## Output diversity

- base: 22/70 unique (0.314)
- dpo: 20/70 unique (0.286)
- relative change: -9.1%
- mode-collapse warning: `false`

> Low absolute diversity is expected at low sampling temperature and is not evidence of collapse on its own; only the base-to-DPO change speaks to that.

## Test-level failures

| Problem | Test | Base | DPO | Hard | DPO-specific | Base-specific |
|---|---|---|---|---|---|---|
| p002 | p002_t002 | 100% | 100% | yes |  |  |
| p002 | p002_t005 | 100% | 100% | yes |  |  |
| p002 | p002_t006 | 100% | 100% | yes |  |  |
| p002 | p002_t007 | 100% | 100% | yes |  |  |
| p004 | p004_t002 | 50% | 50% | yes |  |  |
| p004 | p004_t008 | 30% | 40% |  |  |  |
| p005 | p005_t005 | 10% | 0% |  |  |  |

## Coverage gaps

| Category | Training share | Benchmark share | Ratio | Verdict |
|---|---|---|---|---|
| async | 0% | 0% | n/a | `absent_from_both` |
| dictionaries | 0% | 14% | 0.00 | `underrepresented` |
| edge_cases | 50% | 0% | n/a | `not_in_benchmark` |
| exceptions | 50% | 0% | n/a | `not_in_benchmark` |
| generators | 0% | 14% | 0.00 | `underrepresented` |
| lists | 0% | 14% | 0.00 | `underrepresented` |
| recursion | 0% | 14% | 0.00 | `underrepresented` |
| sets | 0% | 14% | 0.00 | `underrepresented` |
| sorting | 0% | 14% | 0.00 | `underrepresented` |
| strings | 0% | 14% | 0.00 | `underrepresented` |

| Difficulty | Training share | Benchmark share | Ratio | Verdict |
|---|---|---|---|---|
| easy | 100% | 43% | 2.33 | `overrepresented` |
| hard | 0% | 0% | n/a | `absent_from_both` |
| medium | 0% | 57% | 0.00 | `underrepresented` |

> These shares are arithmetic over a catalog carrying roughly one problem per category. The arithmetic is correct; at this sample size the distribution it describes is close to noise, and the table should be read as a structural observation rather than a measurement.

## Training curve

- verdict: `insufficient_data`
- 1 distinct training step(s) logged; over- and undertraining are trend properties and need at least two
- preference overfitting: `not_applicable`
  - Stage 10 evaluates held-out problems only and does not measure training-set performance, so no train-vs-held-out gap can be computed (spec section 69)

## Preference coverage

- pairs in the run: 22
- pairs that reached training: 12
- problems with pairs: p004, p007, p008, p010
- problems without pairs: p001, p002, p003, p005, p006, p009

## Recommendations

### 1. `expand_benchmark` (score 0.670)

**Hypothesis.** Extending the benchmark to cover edge_cases, exceptions would make the categories DPO actually trained on measurable; at present nothing that was trained is evaluated

- confidence: high
- evidence: `{'trained_categories_absent_from_benchmark': ['edge_cases', 'exceptions'], 'training_shares': {'edge_cases': 0.5, 'exceptions': 0.5}}`

### 2. `refine_preference_pairs` (score 0.640)

**Hypothesis.** Only 12 pair(s) reached training; increasing the pair count should give the optimiser enough signal to move the policy at all

- confidence: high
- evidence: `{'trained_pairs': 12, 'total_pairs': 22, 'problems_without_pairs': ['p001', 'p002', 'p003', 'p005', 'p006', 'p009']}`

### 3. `add_data` (score 0.490)

**Hypothesis.** Adding preference pairs for dictionaries should raise held-out pass@1 in those categories, which the current training split does not cover

- confidence: medium
- evidence: `{'underrepresented_categories': ['dictionaries', 'generators', 'lists', 'recursion', 'sets', 'sorting', 'strings'], 'benchmark_categories_unsolved_by_dpo': ['dictionaries'], 'coverage_ratios': {'dictionaries': 0.0, 'generators': 0.0, 'lists': 0.0, 'recursion': 0.0, 'sets': 0.0, 'sorting': 0.0, 'strings': 0.0}}`

### 4. `increase_problem_difficulty` (score 0.440)

**Hypothesis.** The training split is skewed toward easy problems; adding medium and hard problems should produce preference pairs that discriminate on cases the benchmark actually tests

- confidence: medium
- evidence: `{'difficulty_shares': {'easy': {'training': 1.0, 'benchmark': 0.42857142857142855}, 'hard': {'training': 0.0, 'benchmark': 0.0}, 'medium': {'training': 0.0, 'benchmark': 0.5714285714285714}}}`

## What this analysis does not establish

Coverage gaps reported above are associations between the training split's composition and observed failures. They are potential data gaps. This analysis does not establish that any gap produced any failure, and with roughly one problem per category it is not capable of establishing that.

