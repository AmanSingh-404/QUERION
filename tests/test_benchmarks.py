from scripts.collect_benchmarks import summarize


def rec(status, match=False, error=None, gold_empty=False):
    return {"status": status, "match": match, "error": error, "gold_empty": gold_empty}


def test_summarize_counts_and_error_types():
    records = [
        rec("match", True),
        rec("match", True),
        rec("mismatch"),
        rec("pred_error", error="OperationalError: no such column: x"),
        rec("pred_error", error="OperationalError: no such table: y"),
        rec("pred_error", error='OperationalError: near "AS": syntax error'),
        rec("gold_error"),
        rec("match", True, gold_empty=True),
    ]
    s = summarize(records)
    assert s["items"] == 8 and s["matches"] == 3
    assert s["exec_accuracy"] == 0.375
    assert s["status"]["pred_error"] == 3
    assert s["error_types"] == {"unknown column": 1, "unknown table": 1, "syntax error": 1}
    assert s["exec_accuracy_nonempty_gold"] == round(2 / 6, 4)
