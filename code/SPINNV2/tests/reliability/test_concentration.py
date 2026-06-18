import pandas as pd
import pytest

from research.reliability.profiling.concentration import gini, summarize_concentration


def test_concentration_measures_task_risk_ranking_against_exposure_proxy():
    frame = pd.DataFrame(
        [
            {"candidate": "a", "critical_failures": 8, "risk": 0.8, "activation_bytes": 1},
            {"candidate": "b", "critical_failures": 2, "risk": 0.2, "activation_bytes": 1},
            {"candidate": "c", "critical_failures": 0, "risk": 0.1, "activation_bytes": 100},
            {"candidate": "d", "critical_failures": 0, "risk": 0.0, "activation_bytes": 50},
        ]
    )
    summary, curves = summarize_concentration(frame, scope="runtime_protectable")
    assert summary.top_25_percent_coverage == pytest.approx(0.8)
    assert summary.fraction_for_80_percent_coverage == pytest.approx(0.25)
    assert summary.task_risk_auc > summary.exposure_proxy_auc
    assert set(curves["ranking"]) == {"task_risk", "activation_bytes"}


def test_gini_returns_zero_without_observed_failures():
    assert gini(pd.Series([0, 0, 0]).to_numpy()) == 0.0
