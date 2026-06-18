from research.reliability.optimizer.plan_optimizer import (
    CandidateMode,
    optimize_bounded_loss_memory_ilp,
    optimize_greedy,
    optimize_ilp,
    optimize_random_dmr,
    optimize_topk_single_mode,
)
from research.reliability.profiling.risk_profile import InjectionObservation, build_risk_profile


def test_risk_profile_orders_critical_node_first():
    observations = [
        InjectionObservation(0, 1, 100, True, 1.0),
        InjectionObservation(0, 1, 100, True, 0.8),
        InjectionObservation(1, 2, 100, False, 0.0),
        InjectionObservation(1, 2, 100, False, 0.1),
    ]
    profile = build_risk_profile(observations)
    assert profile[0].node_id == 0
    assert profile[0].critical_probability == 1.0
    assert profile[0].ci_low > 0.0


def test_ilp_can_outperform_latency_ratio_greedy():
    candidates = [
        CandidateMode(0, 10, "dmr_compare_rerun", 0.60, 6.0, 10),
        CandidateMode(1, 11, "range_guard_rerun", 0.39, 4.0, 10),
        CandidateMode(2, 12, "range_guard_rerun", 0.39, 4.0, 10),
    ]
    greedy = optimize_greedy(candidates, latency_budget_ms=8.0, memory_budget_bytes=30)
    optimum = optimize_ilp(candidates, latency_budget_ms=8.0, memory_budget_bytes=30)
    assert optimum.total_risk_reduction == 0.78
    assert optimum.total_risk_reduction > greedy.total_risk_reduction
    plan = optimum.to_protection_plan(
        model_id="resnet50_eurosat",
        platform_profile="win_x64_cpu_ref",
        latency_budget_ms=8.0,
        memory_budget_bytes=30,
    )
    assert len(plan["nodes"]) == 2


def test_peak_memory_is_reused_across_multiple_protected_nodes():
    candidates = [
        CandidateMode(0, 10, "dmr_compare_rerun", 0.40, 1.0, 10),
        CandidateMode(1, 11, "dmr_compare_rerun", 0.30, 1.0, 10),
    ]
    result = optimize_ilp(candidates, latency_budget_ms=2.0, memory_budget_bytes=10)
    assert len(result.selected) == 2
    assert result.total_extra_memory_bytes == 10


def test_ilp_handles_small_probability_objective_coefficients():
    candidates = [
        CandidateMode(0, 10, "dmr_compare_rerun", 6e-10, 6.0, 10),
        CandidateMode(1, 11, "dmr_compare_rerun", 3.9e-10, 4.0, 10),
        CandidateMode(2, 12, "dmr_compare_rerun", 3.9e-10, 4.0, 10),
    ]
    result = optimize_ilp(candidates, latency_budget_ms=8.0, memory_budget_bytes=10)
    assert {choice.node_id for choice in result.selected} == {1, 2}


def test_ilp_uses_lower_latency_for_equal_risk_solution():
    candidates = [
        CandidateMode(0, 10, "dmr_compare_rerun", 0.5, 4.0, 10),
        CandidateMode(1, 11, "dmr_compare_rerun", 0.5, 2.0, 10),
    ]
    result = optimize_ilp(candidates, latency_budget_ms=4.0, memory_budget_bytes=10)
    assert [choice.node_id for choice in result.selected] == [1]


def test_single_mode_and_random_baselines_respect_budgets():
    candidates = [
        CandidateMode(0, 10, "dmr_compare_rerun", 0.60, 6.0, 10),
        CandidateMode(1, 11, "dmr_compare_rerun", 0.30, 4.0, 8),
        CandidateMode(1, 11, "range_guard_rerun", 0.20, 1.0, 0),
    ]
    topk = optimize_topk_single_mode(
        candidates, mode="dmr_compare_rerun", latency_budget_ms=4.0, memory_budget_bytes=8
    )
    random_result = optimize_random_dmr(candidates, latency_budget_ms=4.0, memory_budget_bytes=8, seed=2026)
    assert [choice.node_id for choice in topk.selected] == [1]
    assert all(choice.mode == "dmr_compare_rerun" for choice in random_result.selected)
    assert random_result.total_latency_overhead_ms <= 4.0


def test_bounded_loss_memory_ilp_reduces_peak_memory_with_limited_risk_loss():
    candidates = [
        CandidateMode(0, 10, "dmr_compare_rerun", 1.00, 2.0, 100),
        CandidateMode(0, 10, "range_guard_rerun", 0.96, 0.5, 0),
        CandidateMode(1, 11, "dmr_compare_rerun", 0.20, 1.0, 20),
    ]
    optimum = optimize_ilp(candidates, latency_budget_ms=3.0, memory_budget_bytes=100)
    bounded = optimize_bounded_loss_memory_ilp(
        candidates,
        latency_budget_ms=3.0,
        memory_budget_bytes=100,
        risk_loss_tolerance=0.05,
    )
    assert optimum.total_risk_reduction == 1.2
    assert bounded.total_risk_reduction >= optimum.total_risk_reduction * 0.95
    assert bounded.total_extra_memory_bytes < optimum.total_extra_memory_bytes
    assert {choice.mode for choice in bounded.selected} == {"range_guard_rerun", "dmr_compare_rerun"}
