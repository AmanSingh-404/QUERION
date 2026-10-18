import glob
import itertools
import json
import os
import re
from pathlib import Path

ROOT = "/content/drive/MyDrive/querion/data"
EVAL = Path(__file__).resolve().parents[1] / "eval"
MIN_QUESTION = 25
MIN_SQL = 40


def norm(s):
    s = re.sub(r"\s+", " ", s.lower()).strip().rstrip(";")
    return s.strip()


def build_needles(rows):
    """Map normalized eval question/SQL text to its kind. Very short strings are ignored."""
    needles = {}
    for r in rows:
        q, s = norm(r["question"]), norm(r["sql"])
        if len(q) >= MIN_QUESTION:
            needles.setdefault(q, "question")
        if len(s) >= MIN_SQL:
            needles.setdefault(s, "sql")
    return needles


def doc_hits(text, needles):
    """Eval strings that appear in a document as a whole line or a whole ';'-statement."""
    hits = set()
    for piece in itertools.chain(text.split("\n"), text.split(";")):
        if len(piece.strip()) < MIN_QUESTION:
            continue
        n = norm(piece)
        if n in needles:
            hits.add(n)
    return hits


def scan_pairs(path, q_key, s_key, needles):
    out = {"pairs": 0, "question_matches": 0, "sql_matches": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out["pairs"] += 1
            if needles.get(norm(r[q_key])) == "question":
                out["question_matches"] += 1
            if needles.get(norm(r[s_key])) == "sql":
                out["sql_matches"] += 1
    return out


def main():
    rows = []
    for path in sorted(EVAL.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f]
    needles = build_needles(rows)
    report = {"eval_items": len(rows), "needles": len(needles)}

    blocked, idx = [], 0
    for path in sorted(glob.glob(f"{ROOT}/dedup/stack_sql/part-*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if doc_hits(json.loads(line)["text"], needles):
                    blocked.append(idx)
                idx += 1
    report["stack_docs_scanned"] = idx
    report["stack_docs_contaminated"] = len(blocked)
    report["gretel_train"] = scan_pairs(
        f"{ROOT}/verified/gretel_train.jsonl", "sql_prompt", "sql", needles
    )
    report["synthetic"] = scan_pairs(f"{ROOT}/synth.jsonl", "question", "sql", needles)
    report["fineweb"] = "not scanned (general web text)"

    os.makedirs(f"{ROOT}/blocklist", exist_ok=True)
    with open(f"{ROOT}/blocklist/stack_sql.json", "w") as f:
        json.dump(blocked, f)
    with open(f"{ROOT}/leakage_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
