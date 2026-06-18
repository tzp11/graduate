"""Multiple-choice protection-plan optimization under latency and memory budgets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import pulp


@dataclass(frozen=True)
class CandidateMode:
    node_id: int
    tensor_id: int
    mode: str
    risk_reduction: float
    latency_overhead_ms: float
    extra_memory_bytes: int
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True)
class OptimizationResult:
    method: str
    selected: tuple[CandidateMode, ...]
    total_risk_reduction: float
    total_latency_overhead_ms: float
    total_extra_memory_bytes: int

    def to_protection_plan(
        self,
        *,
        model_id: str,
        platform_profile: str,
        latency_budget_ms: float,
        memory_budget_bytes: int,
        fault_prior: str = "activation_bytes_weighted_single_bit",
        workload_scope: str | None = None,
    ) -> dict:
        nodes = []
        for choice in self.selected:
            node = {"node_id": choice.node_id, "tensor_id": choice.tensor_id, "mode": choice.mode}
            if choice.lower_bound is not None:
                node["lower_bound"] = choice.lower_bound
            if choice.upper_bound is not None:
                node["upper_bound"] = choice.upper_bound
            nodes.append(node)
        plan = {
            "version": 1,
            "model_id": model_id,
            "platform_profile": platform_profile,
            "fault_prior": fault_prior,
            "budgets": {
                "latency_overhead_ms": latency_budget_ms,
                "extra_memory_bytes": memory_budget_bytes,
            },
            "optimizer": {
                "method": self.method,
                "predicted_risk_reduction": self.total_risk_reduction,
                "predicted_latency_overhead_ms": self.total_latency_overhead_ms,
                "peak_extra_memory_bytes": self.total_extra_memory_bytes,
            },
            "nodes": nodes,
        }
        if workload_scope is not None:
            plan["workload_scope"] = workload_scope
        return plan


def optimize_ilp(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    if latency_budget_ms < 0 or memory_budget_bytes < 0:
        raise ValueError("budgets must be non-negative")
    problem = pulp.LpProblem("selective_software_protection", pulp.LpMaximize)
    variables = {
        index: problem.add_variable(f"x_{candidate.node_id}_{candidate.mode}_{index}", cat="Binary")
        for index, candidate in enumerate(candidates)
    }
    max_reduction = max((candidate.risk_reduction for candidate in candidates), default=0.0)
    objective_scale = 1.0 / max_reduction if max_reduction > 0.0 else 1.0
    risk_objective = pulp.lpSum(
        variables[i] * candidate.risk_reduction * objective_scale for i, candidate in enumerate(candidates)
    )
    latency_objective = pulp.lpSum(
        variables[i] * candidate.latency_overhead_ms for i, candidate in enumerate(candidates)
    )
    problem += risk_objective
    problem += latency_objective <= latency_budget_ms
    for index, candidate in enumerate(candidates):
        if candidate.extra_memory_bytes > memory_budget_bytes:
            problem += variables[index] == 0
    for node_id in {candidate.node_id for candidate in candidates}:
        problem += pulp.lpSum(variables[i] for i, candidate in enumerate(candidates) if candidate.node_id == node_id) <= 1
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"optimizer failed: {pulp.LpStatus[status]}")
    optimal_risk = float(pulp.value(risk_objective) or 0.0)
    problem += risk_objective >= optimal_risk - 1e-9
    problem.sense = pulp.LpMinimize
    problem.setObjective(latency_objective)
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"optimizer tie-break failed: {pulp.LpStatus[status]}")
    selected = tuple(candidate for i, candidate in enumerate(candidates) if pulp.value(variables[i]) > 0.5)
    return _summarize("ilp", selected)


def optimize_bounded_loss_memory_ilp(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
    risk_loss_tolerance: float = 0.05,
) -> OptimizationResult:
    """Minimize peak protection memory while bounding risk-reduction loss.

    The usual ILP maximizes expected risk reduction and treats memory only as a
    hard feasibility bound. On memory-constrained deployment targets this can
    select a high-memory DMR mode even when a lower-memory mode gives nearly the
    same task-level benefit. This variant first finds the best achievable risk
    reduction, then solves a second ILP that accepts a bounded relative loss and
    minimizes peak extra memory.
    """
    if not 0.0 <= risk_loss_tolerance < 1.0:
        raise ValueError("risk_loss_tolerance must be in [0, 1)")
    optimal = optimize_ilp(
        candidates,
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
    )
    if optimal.total_risk_reduction <= 0.0:
        return _summarize(f"bounded_loss_memory_ilp_tol{risk_loss_tolerance:g}", ())

    min_risk = optimal.total_risk_reduction * (1.0 - risk_loss_tolerance)
    problem = pulp.LpProblem("bounded_loss_memory_protection", pulp.LpMinimize)
    variables = {
        index: problem.add_variable(f"x_{candidate.node_id}_{candidate.mode}_{index}", cat="Binary")
        for index, candidate in enumerate(candidates)
    }
    peak_memory = problem.add_variable("peak_extra_memory_bytes", lowBound=0)
    latency = pulp.lpSum(variables[i] * candidate.latency_overhead_ms for i, candidate in enumerate(candidates))
    risk = pulp.lpSum(variables[i] * candidate.risk_reduction for i, candidate in enumerate(candidates))
    problem += peak_memory
    problem += latency <= latency_budget_ms
    problem += peak_memory <= memory_budget_bytes
    problem += risk >= min_risk
    for index, candidate in enumerate(candidates):
        problem += peak_memory >= variables[index] * candidate.extra_memory_bytes
    for node_id in {candidate.node_id for candidate in candidates}:
        problem += pulp.lpSum(variables[i] for i, candidate in enumerate(candidates) if candidate.node_id == node_id) <= 1
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"bounded-loss memory optimizer failed: {pulp.LpStatus[status]}")

    optimal_peak = float(pulp.value(peak_memory) or 0.0)
    problem += peak_memory <= optimal_peak + 1e-6
    problem.sense = pulp.LpMaximize
    problem.setObjective(risk)
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"bounded-loss memory optimizer tie-break failed: {pulp.LpStatus[status]}")

    best_risk = float(pulp.value(risk) or 0.0)
    problem += risk >= best_risk - 1e-9
    problem.sense = pulp.LpMinimize
    problem.setObjective(latency)
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"bounded-loss memory optimizer latency tie-break failed: {pulp.LpStatus[status]}")

    selected = tuple(candidate for i, candidate in enumerate(candidates) if pulp.value(variables[i]) > 0.5)
    return _summarize(f"bounded_loss_memory_ilp_tol{risk_loss_tolerance:g}", selected)


def optimize_greedy(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    selected: list[CandidateMode] = []
    used_nodes: set[int] = set()
    used_latency = 0.0
    peak_memory = 0
    ordered = sorted(
        candidates,
        key=lambda choice: (
            -(choice.risk_reduction / max(choice.latency_overhead_ms, 1e-12)),
            -choice.risk_reduction,
        ),
    )
    for choice in ordered:
        if choice.node_id in used_nodes:
            continue
        if used_latency + choice.latency_overhead_ms > latency_budget_ms:
            continue
        if max(peak_memory, choice.extra_memory_bytes) > memory_budget_bytes:
            continue
        selected.append(choice)
        used_nodes.add(choice.node_id)
        used_latency += choice.latency_overhead_ms
        peak_memory = max(peak_memory, choice.extra_memory_bytes)
    return _summarize("greedy", tuple(selected))


def optimize_topk_single_mode(
    candidates: list[CandidateMode],
    *,
    mode: str,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """Select highest-benefit objects using one fixed protection primitive."""
    return _pack_ordered(
        [candidate for candidate in candidates if candidate.mode == mode],
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        method=f"topk_{mode}",
        order_key=lambda choice: (-choice.risk_reduction, choice.latency_overhead_ms),
    )


def optimize_random_dmr(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
    seed: int,
) -> OptimizationResult:
    """Random DMR selection baseline with deterministic seed."""
    dmr = [candidate for candidate in candidates if candidate.mode == "dmr_compare_rerun"]
    generator = random.Random(seed)
    generator.shuffle(dmr)
    return _pack_ordered(
        dmr,
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        method="random_dmr",
        order_key=None,
    )


def write_plan(plan: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize(method: str, selected: tuple[CandidateMode, ...]) -> OptimizationResult:
    return OptimizationResult(
        method=method,
        selected=selected,
        total_risk_reduction=sum(choice.risk_reduction for choice in selected),
        total_latency_overhead_ms=sum(choice.latency_overhead_ms for choice in selected),
        total_extra_memory_bytes=max((choice.extra_memory_bytes for choice in selected), default=0),
    )


def _pack_ordered(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
    method: str,
    order_key,
) -> OptimizationResult:
    ordered = sorted(candidates, key=order_key) if order_key is not None else candidates
    selected = []
    used_nodes = set()
    used_latency = 0.0
    peak_memory = 0
    for choice in ordered:
        if choice.node_id in used_nodes:
            continue
        if used_latency + choice.latency_overhead_ms > latency_budget_ms:
            continue
        if max(peak_memory, choice.extra_memory_bytes) > memory_budget_bytes:
            continue
        selected.append(choice)
        used_nodes.add(choice.node_id)
        used_latency += choice.latency_overhead_ms
        peak_memory = max(peak_memory, choice.extra_memory_bytes)
    return _summarize(method, tuple(selected))
