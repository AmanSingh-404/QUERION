import json
import os
import sqlite3
import time
from collections import Counter

RAW = "/content/drive/MyDrive/querion/data/raw/gretel"
OUT = "/content/drive/MyDrive/querion/data/verified"
TIMEOUT = 2.0


def is_select(sql):
    return sql.lstrip().lower().startswith(("select", "with"))


def runs_ok(context, sql, timeout=TIMEOUT):
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(context)
        deadline = time.time() + timeout
        conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100000)
        conn.execute(sql).fetchall()
        return "ok"
    except sqlite3.Error as e:
        return f"error: {type(e).__name__}"
    finally:
        conn.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for split in ("train", "test"):
        stats = Counter()
        with open(f"{RAW}/{split}.jsonl", encoding="utf-8") as fin:
            with open(f"{OUT}/gretel_{split}.jsonl", "w", encoding="utf-8") as fout:
                for line in fin:
                    rec = json.loads(line)
                    stats["total"] += 1
                    if not is_select(rec["sql"]):
                        stats["drop_not_select"] += 1
                        continue
                    result = runs_ok(rec["sql_context"], rec["sql"])
                    if result != "ok":
                        stats["drop_" + result.replace("error: ", "").lower()] += 1
                        continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["kept"] += 1
        stats["kept_pct"] = round(100 * stats["kept"] / stats["total"], 1)
        report[split] = dict(stats)
        print(split, dict(stats), flush=True)
    with open(f"{OUT}/gretel_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
