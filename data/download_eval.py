import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

EVAL = Path(__file__).resolve().parents[1] / "eval"
BASE = "https://huggingface.co/datasets/xlangai/spider/resolve/main/spider"
URL = f"{BASE}/validation-00000-of-00001.parquet"


def main():
    EVAL.mkdir(exist_ok=True)
    tmp = EVAL / "_spider_dev.parquet"
    urllib.request.urlretrieve(URL, tmp)
    df = pd.read_parquet(tmp)
    tmp.unlink()
    path = EVAL / "spider_dev.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(df.to_dict("records")):
            row = {"id": i, "db_id": r["db_id"], "question": r["question"], "sql": r["query"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "spider_dev.jsonl": {
            "items": len(df),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": "xlangai/spider, validation split",
            "license": "CC BY-SA 4.0",
        }
    }
    (EVAL / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
