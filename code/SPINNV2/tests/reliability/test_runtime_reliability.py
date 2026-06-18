import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper
import onnx
import pytest

from research.reliability.injection.bitflip import FaultEvent
from research.reliability.runtime_driver import RuntimeDriver


def _cmake() -> str:
    discovered = shutil.which("cmake")
    if discovered:
        return discovered
    return str(Path(sys.executable).parent / "Scripts" / "cmake.exe")


def test_runtime_dmr_detects_and_recovers_single_fault(tmp_path: Path):
    model_path = tmp_path / "relu.onnx"
    spk_path = tmp_path / "relu_protected.spk"
    plan_path = tmp_path / "plan.json"
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"])],
        "relu",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, model_path)
    plan_path.write_text(
        json.dumps({"version": 1, "model_id": "relu", "nodes": [{"node_id": 0, "tensor_id": 1, "mode": "dmr_compare_rerun"}]}),
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-m", "spinnv2.compiler", "compile", str(model_path), "-o", str(spk_path), "--protection-plan", str(plan_path)],
        check=True,
    )
    subprocess.run([_cmake(), "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run(
        [_cmake(), "--build", "build/runtime", "--config", "Debug", "--target", "spkv2_reliability_test", "spkv2_runtime_shared"],
        check=True,
    )
    runner = Path("build/runtime/Debug/spkv2_reliability_test.exe")
    if not runner.exists():
        runner = Path("build/runtime/spkv2_reliability_test")
    subprocess.run([str(runner), str(spk_path)], check=True)
    library = Path("build/runtime/Debug/spkv2_runtime_shared.dll")
    if library.exists():
        event = FaultEvent("relu", "sample", 0, 1, element_index=1, bit_index=31)
        with RuntimeDriver(library, spk_path) as runtime:
            output, stats = runtime.run(np.array([[-1.0, 2.0, 3.0, 4.0]], dtype=np.float32), event)
        np.testing.assert_array_equal(output, np.array([0.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert stats["injected_faults"] == 1
        assert stats["detected_faults"] == 1
        assert stats["recovered_faults"] == 1


def test_runtime_range_guard_exports_clean_runtime_observations(tmp_path: Path):
    model_path = tmp_path / "relu.onnx"
    spk_path = tmp_path / "relu_range_guard.spk"
    plan_path = tmp_path / "plan.json"
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"])],
        "relu",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, model_path)
    plan_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": "relu",
                "nodes": [
                    {
                        "node_id": 0,
                        "tensor_id": 1,
                        "mode": "range_guard_rerun",
                        "lower_bound": -0.1,
                        "upper_bound": 4.1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-m", "spinnv2.compiler", "compile", str(model_path), "-o", str(spk_path), "--protection-plan", str(plan_path)],
        check=True,
    )
    subprocess.run([_cmake(), "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run([_cmake(), "--build", "build/runtime", "--config", "Debug", "--target", "spkv2_runtime_shared"], check=True)
    library = Path("build/runtime/Debug/spkv2_runtime_shared.dll")
    if library.exists():
        with RuntimeDriver(library, spk_path) as runtime:
            output, stats = runtime.run(np.array([[-1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
            observations = runtime.range_observations([0])
        np.testing.assert_array_equal(output, np.array([0.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert stats["detected_faults"] == 0
        assert observations == [
            {"node_id": 0, "tensor_id": 1, "observed_min": 0.0, "observed_max": 4.0, "observations": 1}
        ]


def test_faulted_gather_index_returns_controlled_error_instead_of_crashing(tmp_path: Path):
    model_path = tmp_path / "faulted_indices.onnx"
    spk_path = tmp_path / "faulted_indices.spk"
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["indices"], ["cast_indices"], to=TensorProto.INT64),
            helper.make_node("GatherElements", ["data", "cast_indices"], ["output"], axis=1),
        ],
        "faulted_indices",
        [helper.make_tensor_value_info("indices", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(np.array([[10.0, 20.0]], dtype=np.float32), "data")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, model_path)
    subprocess.run([sys.executable, "-m", "spinnv2.compiler", "compile", str(model_path), "-o", str(spk_path)], check=True)
    subprocess.run([_cmake(), "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run([_cmake(), "--build", "build/runtime", "--config", "Debug", "--target", "spkv2_runtime_shared"], check=True)
    debug = json.loads(spk_path.with_suffix(".spk.json").read_text(encoding="utf-8"))
    cast_node = next(node for node in debug["nodes"] if node["op_type"] == "Cast")
    event = FaultEvent("faulted_indices", "sample", cast_node["id"], cast_node["outputs"][0], 0, 30)
    library = Path("build/runtime/Debug/spkv2_runtime_shared.dll")
    if library.exists():
        with RuntimeDriver(library, spk_path) as runtime:
            with pytest.raises(RuntimeError, match="runtime run failed"):
                runtime.run(np.array([[0.0, 1.0]], dtype=np.float32), event)
