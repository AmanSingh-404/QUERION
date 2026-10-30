import json
import sys

from eval.compare import exec_match
from eval.repair import repair
from eval.spider import db_file


def main():
    path = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "C:/querion-data/spider_data"
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    before = after = fixed = 0
    for r in records:
        before += r["match"]
        ok = False
        if r["pred"]:
            db = db_file(root, r["db_id"])
            ok = exec_match(db, repair(db, r["pred"]), r["sql"])["match"]
        after += ok
        fixed += ok and not r["match"]
    n = len(records)
    summary = {"items": n, "before": before / n, "after": after / n, "newly_fixed": fixed}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
