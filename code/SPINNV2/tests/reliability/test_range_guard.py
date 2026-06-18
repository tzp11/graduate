import numpy as np

from research.reliability.profiling.range_guard import calibrate_scalar_bounds, outside_scalar_bounds


def test_scalar_range_guard_uses_margin_and_detects_nonfinite_faults():
    bounds = calibrate_scalar_bounds([-1.0, -0.5], [1.0, 0.75], margin_ratio=0.10)
    assert bounds.lower_bound < -1.0
    assert bounds.upper_bound > 1.0
    assert not outside_scalar_bounds(np.array([0.0, 1.05], dtype=np.float32), bounds)
    assert outside_scalar_bounds(np.array([float("inf")], dtype=np.float32), bounds)
    assert outside_scalar_bounds(np.array([1.5], dtype=np.float32), bounds)
