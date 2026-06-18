from research.reliability.profile_detection_protection_costs import build_candidates


def test_detection_candidate_uses_activation_prior_task_risk_and_dmr_cost():
    candidates, details, excluded = build_candidates(
        [
            {
                "node_id": 2,
                "tensor_id": 12,
                "activation_bytes": 100,
                "critical_failures": 2,
                "critical_probability": 0.25,
                "exposure_ratio": 0.4,
            },
            {
                "node_id": 3,
                "tensor_id": 13,
                "activation_bytes": 20,
                "critical_failures": 0,
                "critical_probability": 0.0,
                "exposure_ratio": 0.1,
            },
        ],
        {2: {"avg_ms": 1.5}},
        [
            {"tensor_bytes": 40, "dmr_buffer_ms": 0.1, "range_scan_ms": 0.1},
            {"tensor_bytes": 100, "dmr_buffer_ms": 0.2, "range_scan_ms": 0.2},
        ],
        protectable_node_ids={2},
    )
    assert len(candidates) == 1
    assert candidates[0]["risk_reduction"] == 0.1
    assert candidates[0]["latency_overhead_ms"] == 1.7
    assert candidates[0]["extra_memory_bytes"] == 200
    assert details[0]["critical_failures"] == 2
    assert excluded == []


def test_detection_candidate_excludes_unprotectable_multi_output_node():
    candidates, details, excluded = build_candidates(
        [
            {
                "node_id": 2,
                "tensor_id": 12,
                "activation_bytes": 100,
                "critical_failures": 2,
                "critical_probability": 0.25,
                "exposure_ratio": 0.4,
            }
        ],
        {},
        [],
        protectable_node_ids=set(),
    )
    assert candidates == []
    assert details == []
    assert excluded[0]["node_id"] == 2
