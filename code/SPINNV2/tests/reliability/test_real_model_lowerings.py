from pathlib import Path

from onnx import TensorProto, helper
import onnx

from compiler.frontend.onnx_importer import import_onnx
from research.reliability.audit_spinn_compat import audit_model


def test_global_average_pool_is_lowered_to_supported_reduce_mean(tmp_path: Path):
    path = tmp_path / "global_avg.onnx"
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["input"], ["output"])],
        "global_avg",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, path)
    imported = import_onnx(path)
    assert imported.nodes[0].op_type == "ReduceMean"
    assert imported.nodes[0].attrs == {"axes": [2, 3], "keepdims": 1}
    report = audit_model(path)
    assert report["import_compatible"]
    assert report["lowered_import_ops"] == ["GlobalAveragePool"]
