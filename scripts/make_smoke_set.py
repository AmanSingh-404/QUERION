import faulthandler
import json
import random
import sqlite3
from pathlib import Path

from data.synth import KINDS, make_instance, strict_subset_ok, verify

OUT = Path(__file__).resolve().parents[1] / "tests" / "smoke"


def build(instances=10, per_instance=3, seed=987654):
    rng = random.Random(seed)
    (OUT / "dbs").mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(instances):
        conn, tables, _ = make_instance(rng)
        seen, made = set(), 0
        for _ in range(per_instance * 20):
            if made >= per_instance:
                break
            res = rng.choice(KINDS)(rng, conn, tables)
            if res is None:
                continue
            kind, question, sql = res
            try:
                rows = conn.execute(sql).fetchall()
            except sqlite3.Error:
                continue
            if not rows or not verify(kind, rows, tables):
                continue
            if not strict_subset_ok(kind, sql, rows, conn, tables):
                continue
            if (question, sql) in seen:
                continue
            seen.add((question, sql))
            items.append(
                {
                    "db": f"dbs/smoke{i}.sqlite",
                    "question": question,
                    "sql": sql,
                    "kind": kind.split(":")[0],
                }
            )
            made += 1
        path = OUT / "dbs" / f"smoke{i}.sqlite"
        path.unlink(missing_ok=True)
        disk = sqlite3.connect(path)
        disk.executescript("\n".join(conn.iterdump()))
        disk.close()
        conn.close()
        print("instance", i, "done,", len(items), "items so far", flush=True)
    with open(OUT / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return items


if __name__ == "__main__":
    faulthandler.dump_traceback_later(60, exit=True)
    items = build()
    faulthandler.cancel_dump_traceback_later()
    print(len(items), "items written to", OUT)
