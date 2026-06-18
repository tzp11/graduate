from research.reliability.models.train_resnet50 import _limit_indices, _resolve_device


def test_limit_indices_keeps_all_or_returns_requested_prefix() -> None:
    indices = [1, 2, 3, 4]
    assert _limit_indices(indices, 0) == indices
    assert _limit_indices(indices, 2) == [1, 2]


def test_explicit_cpu_device_is_supported_for_smoke_runs() -> None:
    assert str(_resolve_device("cpu")) == "cpu"
