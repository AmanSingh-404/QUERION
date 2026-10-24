import re
from collections import Counter

from eval.executor import execute


def norm_cell(v):
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 6)
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    return v


def has_order_by(sql):
    return re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE) is not None


def same_result(pred_rows, gold_rows, ordered):
    """Execution match: same cells, ignoring column order. Row order only matters if ordered."""
    if len(pred_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True
    pred_cols = list(zip(*[[norm_cell(v) for v in r] for r in pred_rows], strict=True))
    gold_cols = list(zip(*[[norm_cell(v) for v in r] for r in gold_rows], strict=True))
    if len(pred_cols) != len(gold_cols):
        return False
    if not ordered:
        pred_cols = [tuple(sorted(c, key=repr)) for c in pred_cols]
        gold_cols = [tuple(sorted(c, key=repr)) for c in gold_cols]
    return Counter(pred_cols) == Counter(gold_cols)


def exec_match(db_path, pred_sql, gold_sql, timeout=5.0):
    gold_rows, gold_err = execute(db_path, gold_sql, timeout)
    if gold_err:
        return {"status": "gold_error", "match": False, "error": gold_err}
    pred_rows, pred_err = execute(db_path, pred_sql, timeout)
    if pred_err:
        return {"status": "pred_error", "match": False, "error": pred_err}
    ok = same_result(pred_rows, gold_rows, ordered=has_order_by(gold_sql))
    return {"status": "match" if ok else "mismatch", "match": ok, "error": None}
