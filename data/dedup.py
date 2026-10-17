import glob
import hashlib
import json
import os
import re

SRC = "/content/drive/MyDrive/querion/data/clean/stack_sql"
OUT = "/content/drive/MyDrive/querion/data/dedup/stack_sql"
NUM_PERM = 64
THRESHOLD = 0.85
SHINGLE = 5
MAX_WORDS = 5000
WORD = re.compile(r"\w+")


def shingles(text):
    words = WORD.findall(text.lower())[:MAX_WORDS]
    if len(words) < SHINGLE:
        return {" ".join(words)}
    return {" ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def main():
    from datasketch import MinHash, MinHashLSH

    os.makedirs(OUT, exist_ok=True)
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    seen = set()
    stats = {"files_in": 0, "dup_exact": 0, "dup_near": 0, "files_kept": 0}
    for path in sorted(glob.glob(f"{SRC}/part-*.jsonl")):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as fin:
            with open(f"{OUT}/{name}", "w", encoding="utf-8") as fout:
                for line in fin:
                    stats["files_in"] += 1
                    text = json.loads(line)["text"]
                    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if h in seen:
                        stats["dup_exact"] += 1
                        continue
                    seen.add(h)
                    m = MinHash(num_perm=NUM_PERM)
                    m.update_batch([s.encode("utf-8") for s in shingles(text)])
                    if lsh.query(m):
                        stats["dup_near"] += 1
                        continue
                    lsh.insert(str(stats["files_in"]), m)
                    fout.write(line)
                    stats["files_kept"] += 1
        print(name, stats, flush=True)
    stats["dup_pct"] = round(100 * (1 - stats["files_kept"] / stats["files_in"]), 1)
    with open(f"{OUT}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
