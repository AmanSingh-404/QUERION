import json
import sys

from eval.compare import exec_match
from eval.spider import db_file
from eval.tidy import tidy_literals


def main():
    path = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "C:/querion-data/spider_data"
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    before = after = changed = 0
    for r in records:
        before += r["match"]
        ok = r["match"]
        tidy = tidy_literals(r["pred"]) if r["pred"] else r["pred"]
        if tidy != r["pred"]:
            changed += 1
            ok = exec_match(db_file(root, r["db_id"]), tidy, r["sql"])["match"]
        after += ok
    summary = {
        "items": len(records),
        "predictions_changed": changed,
        "matches_before": before,
        "matches_after": after,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
