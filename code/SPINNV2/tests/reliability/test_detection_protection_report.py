import json

from research.reliability.export_detection_protection_report import _paired_reduction_ci


def test_paired_reduction_ci_is_one_for_complete_recovery(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    protected = tmp_path / "protected.jsonl"
    baseline.write_text(
        "\n".join(json.dumps({"critical_failure": value}) for value in [True, False, True, False]),
        encoding="utf-8",
    )
    protected.write_text(
        "\n".join(json.dumps({"critical_failure": False}) for _ in range(4)),
        encoding="utf-8",
    )
    low, high = _paired_reduction_ci(str(baseline), str(protected), iterations=100, seed=1)
    assert low == 1.0
    assert high == 1.0
