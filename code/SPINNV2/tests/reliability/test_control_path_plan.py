import json
from pathlib import Path

from research.reliability.build_control_path_plan import build_plan


def test_control_path_plan_selects_observed_errors_with_peak_memory_budget(tmp_path: Path):
    injections = tmp_path / "faults.jsonl"
    injections.write_text(
        "\n".join(
            [
                json.dumps({"node_id": 2, "failure_mode": "controlled_execution_error"}),
                json.dumps({"node_id": 2, "failure_mode": "controlled_execution_error"}),
                json.dumps({"node_id": 1, "failure_mode": "controlled_execution_error"}),
                json.dumps({"node_id": 0, "failure_mode": "task_output_failure"}),
            ]
        ),
        encoding="utf-8",
    )
    debug = tmp_path / "model.spk.json"
    debug.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": 1, "outputs": [11]},
                    {"id": 2, "outputs": [12]},
                ],
                "tensors": [
                    {"id": 11, "size_bytes": 16},
                    {"id": 12, "size_bytes": 128},
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = build_plan(
        injections,
        debug,
        model_id="detector",
        platform_profile="windows",
        memory_budget_bytes=64,
        fault_prior="stratified_runtime_objects",
        workload_scope="formal detection workload",
    )
    assert plan["nodes"] == [{"node_id": 1, "tensor_id": 11, "mode": "dmr_compare_rerun"}]
    assert plan["fault_prior"] == "stratified_runtime_objects"
    assert plan["workload_scope"] == "formal detection workload"
    assert plan["optimizer"]["observed_execution_errors_covered"] == 1
    assert plan["optimizer"]["observed_execution_errors_skipped"] == 2
