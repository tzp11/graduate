import json

import onnx
from onnx import TensorProto, helper

from research.reliability.map_runtime_candidates import map_candidates


def test_mapping_marks_eliminated_and_retained_outputs(tmp_path):
    model_path = tmp_path / "model.onnx"
    debug_path = tmp_path / "model.spk.json"
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "weight"], ["conv_out"], name="/conv/Conv"),
            helper.make_node("Relu", ["conv_out"], ["relu_out"], name="/relu/Relu"),
        ],
        "mapping",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("relu_out", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor("weight", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])],
    )
    onnx.save(helper.make_model(graph), model_path)
    debug_path.write_text(
        json.dumps(
            {
                "tensors": [{"id": 0, "name": "relu_out"}],
                "nodes": [{"id": 0, "op_type": "Conv", "outputs": [0], "attrs": {"fused_activation": "Relu"}}],
            }
        ),
        encoding="utf-8",
    )
    report = map_candidates(
        [
            {"node_id": 0, "module_name": "conv", "invocation_index": 1},
            {"node_id": 1, "module_name": "relu", "invocation_index": 1},
        ],
        model_path,
        debug_path,
    )
    assert not report["mappings"][0]["retained_after_passes"]
    assert report["mappings"][1]["runtime_node_id"] == 0
    assert report["retained_count"] == 1
