import json
import os
import sys

from datasets import load_dataset

RAW = "/content/drive/MyDrive/querion/data/raw"
STACK_MAX_BYTES = 1_500_000_000
SHARD_BYTES = 100_000_000
MIN_CHARS = 100
MAX_CHARS = 200_000


def download_gretel():
    out = f"{RAW}/gretel"
    os.makedirs(out, exist_ok=True)
    ds = load_dataset("gretelai/synthetic_text_to_sql")
    for split in ds:
        ds[split].to_json(f"{out}/{split}.jsonl", force_ascii=False)
        print("gretel", split, len(ds[split]))


def download_stack_sql():
    out = f"{RAW}/stack_sql"
    done = f"{out}/DONE"
    if os.path.exists(done):
        with open(done) as g:
            print("stack_sql already downloaded:", g.read())
        return
    os.makedirs(out, exist_ok=True)
    ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/sql", split="train", streaming=True)
    shard, files, total, shard_bytes = 0, 0, 0, 0
    f = open(f"{out}/part-{shard:05d}.jsonl", "w", encoding="utf-8")
    for row in ds:
        text = row["content"]
        if not MIN_CHARS <= len(text) <= MAX_CHARS:
            continue
        rec = {
            "text": text,
            "repo": row.get("max_stars_repo_name"),
            "path": row.get("max_stars_repo_path"),
            "licenses": row.get("max_stars_repo_licenses"),
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        n = len(line.encode("utf-8"))
        if shard_bytes + n > SHARD_BYTES:
            f.close()
            shard += 1
            shard_bytes = 0
            f = open(f"{out}/part-{shard:05d}.jsonl", "w", encoding="utf-8")
        f.write(line)
        shard_bytes += n
        total += n
        files += 1
        if files % 20000 == 0:
            print("files", files, "MB", total // 1_000_000, flush=True)
        if total >= STACK_MAX_BYTES:
            break
    f.close()
    msg = f"files={files} bytes={total} shards={shard + 1}"
    with open(done, "w") as g:
        g.write(msg)
    print("stack_sql done:", msg)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("gretel", "all"):
        download_gretel()
    if what in ("stack", "all"):
        download_stack_sql()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
