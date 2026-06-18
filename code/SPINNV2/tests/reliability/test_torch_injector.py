import torch

from research.reliability.injection.bitflip import FaultEvent
from research.reliability.injection.torch_injector import (
    capture_candidate_output,
    candidate_module_names,
    discover_candidate_points,
    infer_with_fault,
    infer_with_fault_and_capture,
)


def test_torch_semantic_injector_flips_selected_module_output():
    model = torch.nn.Sequential(torch.nn.ReLU())
    event = FaultEvent("m", "s", 0, 0, element_index=1, bit_index=31)
    output = infer_with_fault(model, torch.tensor([[-1.0, 2.0]], dtype=torch.float32), module_name="0", event=event)
    assert output.tolist() == [[0.0, -2.0]]
    assert candidate_module_names(model) == ["0"]


def test_reused_module_is_discovered_as_separate_runtime_outputs():
    class SharedRelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, inputs):
            self.relu(inputs)
            return self.relu(torch.cat((inputs, inputs), dim=1))

    model = SharedRelu()
    inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    points = discover_candidate_points(model, inputs)
    assert [(point.module_name, point.invocation_index, point.element_count) for point in points] == [
        ("relu", 1, 2),
        ("relu", 2, 4),
    ]
    event = FaultEvent("m", "s", 1, 1, element_index=3, bit_index=31, invocation_index=2)
    output = infer_with_fault(model, inputs, module_name="relu", event=event)
    assert output.tolist() == [[1.0, 2.0, 1.0, -2.0]]


def test_captured_activation_matches_selected_clean_and_faulted_invocation():
    model = torch.nn.Sequential(torch.nn.ReLU())
    inputs = torch.tensor([[-1.0, 2.0]], dtype=torch.float32)
    clean = capture_candidate_output(model, inputs, module_name="0", invocation_index=1)
    event = FaultEvent("m", "s", 0, 0, element_index=1, bit_index=31)
    output, faulted = infer_with_fault_and_capture(model, inputs, module_name="0", event=event)
    assert clean.tolist() == [[0.0, 2.0]]
    assert faulted.tolist() == [[0.0, -2.0]]
    assert output.tolist() == faulted.tolist()
