from pathlib import Path

from compiler.ir import types
from compiler.ir.graph import Graph, Node, Tensor
from compiler.packager.spk_writer import write_spk
from compiler.planner.kernel_spec import select_kernel_specs
from compiler.planner.memory_plan import plan_memory
from compiler.reliability.protection_plan import ProtectionEntry, ProtectionPlan
from compiler.target.profile import load_target_profile


def test_dmr_plan_accounts_for_two_output_copies(tmp_path: Path):
    graph = Graph(model_name="protected")
    graph.add_tensor(Tensor(0, "input", types.DTYPE_FP32, [1, 4], types.ROLE_INPUT))
    graph.add_tensor(Tensor(1, "output", types.DTYPE_FP32, [1, 4], types.ROLE_OUTPUT))
    graph.inputs = [0]
    graph.outputs = [1]
    graph.add_node(Node(0, "Relu", [0], [1]))
    profile = load_target_profile("cpu_ref")
    plan = ProtectionPlan(
        version=1,
        model_id="protected",
        nodes=(ProtectionEntry(0, 1, "dmr_compare_rerun"),),
    )
    path = tmp_path / "protected.spk"
    write_spk(
        graph,
        path,
        profile,
        memory_plan=plan_memory(graph),
        kernel_plan=select_kernel_specs(graph, profile),
        protection_plan=plan,
    )
    debug = path.with_suffix(".spk.json").read_text(encoding="utf-8")
    assert '"protection_scratch_bytes": 32' in debug
    assert '"dmr_compare_rerun"' in debug
