from scripts.collect_engine_table import row_text


def test_row_text_formats_ratios_and_marks_missing_values():
    row = {
        "clients": 16,
        "batched": {"tokens_per_second": 275.0, "latency_p50": 0.869, "kv_reserved_gb": 0.805},
        "paged": {
            "tokens_per_second": 260.6,
            "latency_p50": 0.92,
            "kv_reserved_gb": 0.403,
            "kv_peak_used_gb": 0.213,
        },
        "speedup_median": 0.94,
        "speedup_all": [1.01, 0.94, 0.96, 0.93, 0.94],
    }
    text = row_text("Paged KV cache", "GPU", "batched", "paged", row)
    assert "0.94x (0.93-1.01)" in text
    assert "275 -> 261" in text
    assert "0.81 -> 0.40" in text
    assert text.count(" - ") >= 2  # first-token speedup, peak used and hit rate are missing
