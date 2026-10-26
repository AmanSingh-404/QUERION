import sqlite3
from collections import Counter
from pathlib import Path

from eval.compare import exec_match
from tokenizer.normalize import normalize


def db_file(root, db_id):
    return Path(root) / "database" / db_id / f"{db_id}.sqlite"


def schema_statements(db_path):
    """CREATE TABLE statements of a SQLite file, normalized like the training prompts."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        query = (
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        )
        return [normalize(row[0]) for row in conn.execute(query)]
    finally:
        conn.close()


def classify_error(message):
    text = (message or "").lower()
    for key, label in (
        ("no such column", "unknown column"),
        ("no such table", "unknown table"),
        ("ambiguous column", "ambiguous column"),
        ("syntax error", "syntax error"),
        ("incomplete input", "syntax error"),
    ):
        if key in text:
            return label
    return "other error"


def evaluate(items, root, predict, timeout=5.0):
    """predict(statements, question) -> SQL string, or None if the prompt was too long."""
    records, status, errors = [], Counter(), Counter()
    for item in items:
        db = db_file(root, item["db_id"])
        statements = schema_statements(db)
        pred = predict(statements, item["question"])
        if pred is None:
            result = {"status": "prompt_too_long", "match": False, "error": None}
        else:
            result = exec_match(db, pred, item["sql"], timeout)
        status[result["status"]] += 1
        if result["status"] == "pred_error":
            errors[classify_error(result["error"])] += 1
        records.append({**item, "pred": pred, **result})
        total = len(items)
    informative = [r for r in records if r["status"] != "gold_error" and not r.get("gold_empty")]
    summary = {
        "items": total,
        "exec_accuracy": round(status["match"] / total, 4) if total else 0.0,
        "exec_accuracy_nonempty_gold": round(
            sum(r["match"] for r in informative) / len(informative), 4
        )
        if informative
        else 0.0,
        "nonempty_gold_items": len(informative),
        "status": dict(status),
        "error_types": dict(errors),
    }
    return summary, records
