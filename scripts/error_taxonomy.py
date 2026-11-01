import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eval.spider import classify_error
from eval.taxonomy import classify_failure
from scripts.classify_column_errors import classify

EVAL_DIR = Path("C:/querion-runs/eval")
OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "error_taxonomy.json"
COLUMN_KIND = {
    "column exists in no table of the database": "invented name",
    "column belongs to another table of the query (wrong alias)": "wrong alias",
    "column belongs only to tables the query does not use": "table not in the query",
}


def label(record, root):
    column_kind = None
    if record["status"] == "pred_error" and classify_error(record["error"]) == "unknown column":
        try:
            column_kind = COLUMN_KIND[classify(record, root)]
        except (AttributeError, KeyError):
            column_kind = "other"
    return classify_failure(record, column_kind)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "augfulls8"
    root = sys.argv[2] if len(sys.argv) > 2 else "C:/querion-data/spider_data"
    with open(EVAL_DIR / f"spider_dev_{name}.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    failures = [r for r in records if r["status"] not in ("match", "gold_error")]
    labeled = [(label(r, root), r) for r in failures]
    counts = Counter(lab for lab, _ in labeled)
    result = {
        "run": name,
        "items": len(records),
        "failures": len(failures),
        "labels": dict(counts.most_common()),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    rng = random.Random(0)
    by_label = defaultdict(list)
    for lab, r in labeled:
        by_label[lab].append(r)
    with open(EVAL_DIR / "error_examples.md", "w", encoding="utf-8") as f:
        for lab, rows in by_label.items():
            f.write(f"## {lab} ({len(rows)})\n\n")
            for r in rng.sample(rows, min(3, len(rows))):
                f.write(f"Q: {r['question']}\ngold: {r['sql']}\npred: {r['pred']}\n\n")
    with open(EVAL_DIR / "hand_check.md", "w", encoding="utf-8") as f:
        for i, (lab, r) in enumerate(rng.sample(labeled, min(40, len(labeled))), 1):
            f.write(f"{i}. [{lab}]\nQ: {r['question']}\ngold: {r['sql']}\npred: {r['pred']}\n\n")
    print(f"{len(failures)} failures out of {len(records)} questions")
    for lab, n in counts.most_common():
        print(f"{n:4d}  {n / len(failures):5.1%}  {lab}")


if __name__ == "__main__":
    main()
