"""CLI for producing a protected-execution plan from measured candidates."""

from __future__ import annotations

import argparse
import json

from research.reliability.optimizer.plan_optimizer import (
    CandidateMode,
    optimize_bounded_loss_memory_ilp,
    optimize_greedy,
    optimize_ilp,
    write_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize a SPINNV2 reliability ProtectionPlan.")
    parser.add_argument("candidates", help="JSON list of measured candidate protection modes.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--platform-profile", default="win_x64_cpu_ref")
    parser.add_argument("--fault-prior", default="activation_bytes_weighted_single_bit")
    parser.add_argument("--workload-scope")
    parser.add_argument("--latency-budget-ms", required=True, type=float)
    parser.add_argument("--memory-budget-bytes", required=True, type=int)
    parser.add_argument("--method", choices=("ilp", "greedy", "bounded-loss-memory"), default="ilp")
    parser.add_argument("--bounded-loss-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    raw = json.loads(open(args.candidates, encoding="utf-8").read())
    candidates = [CandidateMode(**item) for item in raw]
    if args.method == "ilp":
        result = optimize_ilp(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "greedy":
        result = optimize_greedy(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    else:
        result = optimize_bounded_loss_memory_ilp(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
            risk_loss_tolerance=args.bounded_loss_tolerance,
        )
    plan = result.to_protection_plan(
        model_id=args.model_id,
        platform_profile=args.platform_profile,
        latency_budget_ms=args.latency_budget_ms,
        memory_budget_bytes=args.memory_budget_bytes,
        fault_prior=args.fault_prior,
        workload_scope=args.workload_scope,
    )
    write_plan(plan, args.output)
    print(json.dumps({"method": result.method, "selected": len(result.selected), "risk_reduction": result.total_risk_reduction}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
