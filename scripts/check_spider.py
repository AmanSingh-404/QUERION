import json
import sys
from pathlib import Path

from eval.executor import execute

EVAL = Path(__file__).resolve().parents[1] / "eval" / "spider_dev.jsonl"


def check(rows, root, official=None):
    root = Path(root)
    report = {"items": len(rows), "missing_databases": [], "gold_errors": 0, "gold_empty": 0}
    for r in rows:
        db = root / "database" / r["db_id"] / f"{r['db_id']}.sqlite"
        if not db.exists():
            if r["db_id"] not in report["missing_databases"]:
                report["missing_databases"].append(r["db_id"])
            continue
        out, err = execute(db, r["sql"])
        if err:
            report["gold_errors"] += 1
        elif not out:
            report["gold_empty"] += 1
    if official is not None:
        if len(official) != len(rows):
            report["dev_json_mismatches"] = f"different length ({len(official)} vs {len(rows)})"
        else:
            pairs = zip(rows, official, strict=True)
            bad = sum((a["db_id"], a["question"]) != (b["db_id"], b["question"]) for a, b in pairs)
            report["dev_json_mismatches"] = bad
    return report


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/querion-data/spider")
    with open(EVAL, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    dev = root / "dev.json"
    official = json.loads(dev.read_text(encoding="utf-8")) if dev.exists() else None
    print(json.dumps(check(rows, root, official), indent=2))


if __name__ == "__main__":
    main()
