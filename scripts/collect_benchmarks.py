import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from eval.spider import classify_error

EVAL_DIR = Path("C:/querion-runs/eval")
OUT_DIR = Path(__file__).resolve().parents[1] / "benchmarks"

RUNS = [
    ("pretrained", "Pretrained, zero-shot, greedy", "pretrained"),
    ("sftfull", "+ SFT", "sft"),
    ("augfull", "+ SFT with identifier augmentation (v0.6.0 weights)", "sft-aug"),
    ("augfullk", "+ schema-constrained decoding", "sft-aug"),
    ("augfulls8", "+ 8 samples with execution-guided vote", "sft-aug"),
]


def summarize(records):
    n = len(records)
    status = Counter(r["status"] for r in records)
    errors = Counter(classify_error(r["error"]) for r in records if r["status"] == "pred_error")
    informative = [r for r in records if r["status"] != "gold_error" and not r.get("gold_empty")]
    matches = status["match"]
    p = matches / n
    nonempty = sum(r["match"] for r in informative) / len(informative) if informative else 0.0
    return {
        "items": n,
        "matches": matches,
        "exec_accuracy": round(p, 4),
        "std_error": round(math.sqrt(p * (1 - p) / n), 4),
        "exec_accuracy_nonempty_gold": round(nonempty, 4),
        "status": dict(status),
        "error_types": dict(errors),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    rows = []
    for name, label, model in RUNS:
        path = EVAL_DIR / f"spider_dev_{name}.jsonl"
        if not path.exists():
            print("missing:", path)
            continue
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        rows.append({"run": name, "label": label, "model": model, **summarize(records)})
    result = {
        "benchmark": "Spider dev, plain execution accuracy on the original dev databases",
        "collected_at_commit": git.stdout.strip(),
        "runs": rows,
    }
    (OUT_DIR / "accuracy.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["| Stage | Execution accuracy | Matches |", "|---|---|---|"]
    for r in rows:
        pct = r["exec_accuracy"] * 100
        lines.append(f"| {r['label']} | {pct:.1f}% | {r['matches']} of {r['items']} |")
    (OUT_DIR / "accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
