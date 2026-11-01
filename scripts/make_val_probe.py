import json
from pathlib import Path

from train.sft_data import is_val_db

ROOT = Path("C:/querion-data/spider_data")
OUT = Path("C:/querion-runs/eval/spider_valprobe.jsonl")


def main():
    with open(ROOT / "train_spider.json", encoding="utf-8") as f:
        rows = [r for r in json.load(f) if is_val_db(r["db_id"])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            item = {"id": i, "db_id": r["db_id"], "question": r["question"], "sql": r["query"]}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(len(rows), "items from", len({r["db_id"] for r in rows}), "databases")


if __name__ == "__main__":
    main()
