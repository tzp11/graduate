import numpy as np

from research.reliability.injection.bitflip import FaultEvent, flip_fp32_bit, sample_fault_event
from research.reliability.metrics.task_metrics import (
    classification_failure,
    detection_failure,
    target_aware_detection_failure,
)


def test_fp32_bit_flip_is_deterministic_and_round_trips():
    values = np.array([1.0, 2.0], dtype=np.float32)
    changed = flip_fp32_bit(values, 1, 31)
    assert changed.tolist() == [1.0, -2.0]
    assert flip_fp32_bit(changed, 1, 31).tolist() == values.tolist()
    first = sample_fault_event(model_id="m", sample_id="s", node_id=1, tensor_id=2, element_count=32, seed=2026)
    second = FaultEvent.from_json(first.to_json())
    assert first == second


def test_classification_metric_marks_correct_to_wrong_as_critical():
    result = classification_failure(
        np.array([4.0, 0.0], dtype=np.float32),
        np.array([0.0, 4.0], dtype=np.float32),
        target_class=0,
    )
    assert result.critical_failure
    assert result.baseline_class == 0
    assert result.faulted_class == 1


def test_classification_metric_marks_nonfinite_fault_output_as_critical():
    result = classification_failure(
        np.array([4.0, 0.0], dtype=np.float32),
        np.array([np.inf, 0.0], dtype=np.float32),
        target_class=0,
    )
    assert result.critical_failure
    assert result.severity == 1.0
    assert result.faulted_class == -1


def test_detection_metric_marks_missed_target():
    baseline = np.array([[0, 0, 10, 10, 0.9, 1]], dtype=np.float32)
    faulted = np.empty((0, 6), dtype=np.float32)
    result = detection_failure(baseline, faulted)
    assert result.critical_failure
    assert result.missed_targets == 1


def test_target_aware_detection_ignores_existing_false_positive_loss():
    baseline = np.array([[0, 0, 10, 10, 0.9, 1]], dtype=np.float32)
    faulted = np.empty((0, 6), dtype=np.float32)
    targets = np.array([[20, 20, 30, 30, 1]], dtype=np.float32)
    result = target_aware_detection_failure(baseline, faulted, targets)
    assert not result.critical_failure
    assert result.baseline_true_positives == 0


def test_target_aware_detection_marks_loss_of_baseline_true_positive():
    baseline = np.array([[0, 0, 10, 10, 0.9, 1]], dtype=np.float32)
    faulted = np.empty((0, 6), dtype=np.float32)
    targets = np.array([[0, 0, 10, 10, 1]], dtype=np.float32)
    result = target_aware_detection_failure(baseline, faulted, targets)
    assert result.critical_failure
    assert result.missed_targets == 1
    assert result.baseline_true_positives == 1
    assert result.severity == 1.0


def test_target_aware_detection_severity_is_bounded_for_many_new_false_positives():
    baseline = np.array([[0, 0, 10, 10, 0.9, 1]], dtype=np.float32)
    faulted = np.array([[20 + i, 20, 30 + i, 30, 0.9, 1] for i in range(20)], dtype=np.float32)
    targets = np.array([[0, 0, 10, 10, 1]], dtype=np.float32)
    result = target_aware_detection_failure(baseline, faulted, targets)
    assert result.critical_failure
    assert result.severity == 1.0
