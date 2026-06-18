from research.reliability.mechanism_screening import (
    ADOPT_AS_MAIN,
    KEEP_AS_ABLATION,
    REJECT,
    decide_adaptive_range,
    decide_guard_gated_dmr,
    decide_robust_memory_ilp,
    decide_task_utility_guard,
    wilson_lower_bound,
)


def test_wilson_lower_bound_is_conservative():
    lower = wilson_lower_bound(80, 100)
    assert 0.70 < lower < 0.80
    assert wilson_lower_bound(0, 0) == 0.0


def test_guard_gated_dmr_decision_thresholds():
    assert decide_guard_gated_dmr(latency_reduction=0.31, recovery_retention=0.81).decision == ADOPT_AS_MAIN
    assert decide_guard_gated_dmr(latency_reduction=0.20, recovery_retention=0.70).decision == KEEP_AS_ABLATION
    assert decide_guard_gated_dmr(latency_reduction=0.10, recovery_retention=0.90).decision == REJECT


def test_adaptive_range_decision_thresholds():
    assert decide_adaptive_range(false_positive_rate=0.005, coverage_gain=0.11).decision == ADOPT_AS_MAIN
    assert decide_adaptive_range(false_positive_rate=0.015, coverage_gain=0.01).decision == KEEP_AS_ABLATION
    assert decide_adaptive_range(false_positive_rate=0.02, coverage_gain=0.00).decision == REJECT


def test_task_utility_guard_decision_thresholds():
    assert (
        decide_task_utility_guard(false_positive_rate=0.001, reduction_gain=0.11, trigger_reduction=0.0).decision
        == ADOPT_AS_MAIN
    )
    assert (
        decide_task_utility_guard(false_positive_rate=0.001, reduction_gain=0.0, trigger_reduction=0.21).decision
        == ADOPT_AS_MAIN
    )
    assert (
        decide_task_utility_guard(false_positive_rate=0.009, reduction_gain=0.01, trigger_reduction=0.0).decision
        == KEEP_AS_ABLATION
    )
    assert (
        decide_task_utility_guard(false_positive_rate=0.02, reduction_gain=0.50, trigger_reduction=0.50).decision
        == REJECT
    )


def test_robust_memory_ilp_decision_thresholds():
    assert decide_robust_memory_ilp(risk_retention=0.96, memory_saving_ratio=0.25).decision == ADOPT_AS_MAIN
    assert decide_robust_memory_ilp(risk_retention=0.91, memory_saving_ratio=0.10).decision == KEEP_AS_ABLATION
    assert decide_robust_memory_ilp(risk_retention=0.80, memory_saving_ratio=0.50).decision == REJECT
