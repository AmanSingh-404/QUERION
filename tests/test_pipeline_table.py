from scripts.collect_pipeline_table import row_text


def test_row_text_formats_accuracy_rates_and_a_missing_gpu_figure():
    summary = {
        "items": 1034,
        "clients": 16,
        "samples": 8,
        "exec_accuracy": 0.3559,
        "matches": 368,
        "questions_per_second": 2.28,
        "latency_p50": 5.94,
        "latency_p95": 18.37,
        "prefix_hit_rate": 0.974,
    }
    text = row_text("full_vote", summary)
    assert "35.6% (368)" in text and "2.28" in text and "97%" in text
    assert text.endswith("| - |")
