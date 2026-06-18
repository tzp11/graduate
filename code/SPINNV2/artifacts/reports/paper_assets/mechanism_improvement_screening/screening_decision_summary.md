# 保护机制改进筛选结论

## Guard-Gated DMR

- decision: `keep_as_ablation`
- reason: partial gain but below the main-method threshold

- nodes_evaluated: `20`
- latency_reduction: `0.7699572552477518`
- recovery_retention: `0.6215760908874921`
- always_dmr_peak_extra_memory_bytes: `6422528`
- gated_dmr_peak_extra_memory_bytes: `6422528`

## Adaptive Quantile Range Guard

- decision: `reject`
- reason: no verified coverage gain under the false-positive constraint

- policies_evaluated: `4`
- best_policy: `minmax`
- best_critical_coverage: `0.0`
- best_false_positive_rate: `0.0`
- best_coverage_gain_vs_minmax: `0.0`

## Task-Utility Consistency Guard

- decision: `adopt_as_main`
- reason: task utility guard improves reduction or reduces rerun triggers under false-positive constraint

- baseline_task_output_guard_reduction: `0.5052264808362369`
- task_utility_guard_reduction: `0.9721254355400697`
- reduction_gain_vs_current: `0.4668989547038328`
- utility_guard_trigger_rate: `0.028`
- utility_clean_false_positive_rate: `0.0`

## Confidence-Robust Bounded-Loss Memory ILP

- decision: `adopt_as_main`
- reason: risk retention and memory saving pass the robust low-memory optimization thresholds

- risk_loss_tolerance: `0.05`
- best_latency_budget_ms: `5.0`
- best_peak_memory_budget_bytes: `1048576`
- best_risk_retention_vs_ilp: `0.9738805502401912`
- best_memory_saving_ratio_vs_ilp: `1.0`
