import glob
import json
import os
from collections import Counter

from tokenizer.normalize import normalize

RAW = "/content/drive/MyDrive/querion/data/raw/stack_sql"
OUT = "/content/drive/MyDrive/querion/data/clean/stack_sql"
MIN_CHARS = 200
MAX_LINE = 1000


def is_data_line(s):
    s = s.strip().lower()
    if s.startswith(("insert ", "insert\t", "values")):
        return True
    return s.startswith("(") and s.endswith(("),", ");")) and len(s) > 20


def reject_reason(text):
    if "\x00" in text:
        return "nul_byte"
    lines = [x for x in text.split("\n") if x.strip()]
    if not lines:
        return "empty"
    if max(len(x) for x in lines) > MAX_LINE:
        return "long_line"
    if sum(is_data_line(x) for x in lines) / len(lines) > 0.5:
        return "data_dump"
    if sum(c.isalnum() for c in text) / len(text) < 0.25:
        return "low_alnum"
    if sum(ord(c) > 127 for c in text) / len(text) > 0.2:
        return "non_english"
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = Counter()
    for path in sorted(glob.glob(f"{RAW}/part-*.jsonl")):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as fin:
            with open(f"{OUT}/{name}", "w", encoding="utf-8") as fout:
                for line in fin:
                    rec = json.loads(line)
                    stats["files_in"] += 1
                    stats["chars_in"] += len(rec["text"])
                    reason = reject_reason(rec["text"])
                    if reason is None:
                        rec["text"] = normalize(rec["text"])
                        if len(rec["text"]) < MIN_CHARS:
                            reason = "too_short"
                    if reason:
                        stats["drop_" + reason] += 1
                        continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["files_kept"] += 1
                    stats["chars_kept"] += len(rec["text"])
        print(name, stats["files_kept"], "/", stats["files_in"], flush=True)
    result = dict(stats)
    result["kept_pct"] = round(100 * stats["files_kept"] / stats["files_in"], 1)
    with open(f"{OUT}/stats.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
