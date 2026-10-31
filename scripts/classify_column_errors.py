import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from eval.spider import db_file

REF = re.compile(r"\b(?:from|join)\s+(\w+)", re.IGNORECASE)
MISSING = re.compile(r"no such column: (?:\w+\.)?(\w+)", re.IGNORECASE)


def table_columns(db):
    uri = f"{Path(db).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        query = "SELECT name FROM sqlite_master WHERE type = 'table'"
        out = {}
        for (t,) in conn.execute(query).fetchall():
            out[t.lower()] = {r[1].lower() for r in conn.execute(f'PRAGMA table_info("{t}")')}
        return out
    finally:
        conn.close()


def classify(r, root):
    col = MISSING.search(r["error"]).group(1).lower()
    cols = table_columns(db_file(root, r["db_id"]))
    used = {t.lower() for t in REF.findall(r["pred"])}
    if not any(col in c for c in cols.values()):
        return "column exists in no table of the database"
    if any(col in cols.get(t, set()) for t in used):
        return "column belongs to another table of the query (wrong alias)"
    return "column belongs only to tables the query does not use"


def main():
    path = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "C:/querion-data/spider_data"
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    counts, examples = Counter(), {}
    for r in records:
        if r["status"] == "pred_error" and "no such column" in (r["error"] or ""):
            kind = classify(r, root)
            counts[kind] += 1
            examples.setdefault(kind, []).append(r)
    print(json.dumps(dict(counts), indent=2))
    for kind, rows in examples.items():
        print(f"\n== {kind}")
        for r in rows[:4]:
            print(f"Q: {r['question']}\n  gold: {r['sql']}\n  pred: {r['pred']}\n  {r['error']}")


if __name__ == "__main__":
    main()
