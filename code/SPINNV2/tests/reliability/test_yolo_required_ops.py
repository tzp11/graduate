from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def _cmake() -> str:
    discovered = shutil.which("cmake")
    if discovered:
        return discovered
    return str(Path(sys.executable).parent / "Scripts" / "cmake.exe")


def test_gather_and_slice_compile_and_execute_on_windows_runtime(tmp_path: Path):
    model_path = tmp_path / "gather_slice.onnx"
    spk_path = tmp_path / "gather_slice.spk"
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    nodes = [
        helper.make_node("Gather", ["input", "indices"], ["selected"], axis=0),
        helper.make_node("Slice", ["selected", "starts", "ends", "axes"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "gather_slice",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 1])],
        [
            numpy_helper.from_array(np.array([2, 0], dtype=np.int64), "indices"),
            numpy_helper.from_array(np.array([0], dtype=np.int64), "starts"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "ends"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, model_path)
    subprocess.run([sys.executable, "-m", "spinnv2.compiler", "compile", str(model_path), "-o", str(spk_path)], check=True)
    subprocess.run([_cmake(), "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run([_cmake(), "--build", "build/runtime", "--config", "Debug", "--target", "spkv2_run"], check=True)
    values = np.arange(8, dtype=np.float32).reshape(4, 2)
    values.tofile(input_path)
    runner = Path("build/runtime/Debug/spkv2_run.exe")
    if not runner.exists():
        runner = Path("build/runtime/spkv2_run")
    subprocess.run([str(runner), str(spk_path), str(input_path), str(output_path)], check=True)
    expected = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]).run(None, {"input": values})[0]
    actual = np.fromfile(output_path, dtype=np.float32).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
