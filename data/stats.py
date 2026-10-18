import json
import sys
from pathlib import Path

import numpy as np

BOS = 1
EOS = 2
VOCAB = 24000
MAX_LEN = 2048
BINS = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
PLAN = {"fineweb": 1_180_000_000, "stack": 800_000_000, "pairs": 20_000_000}


def doc_lengths(arr):
    """Lengths (in tokens, including BOS and EOS) of the documents in one token stream."""
    ends = np.flatnonzero(arr == EOS)
    starts = np.concatenate([[0], ends[:-1] + 1]) if len(ends) else np.array([], dtype=int)
    return ends - starts + 1


def analyze_source(src_dir):
    src = Path(src_dir)
    train_files = sorted(src.glob("train-*.bin"))
    if not train_files:
        raise FileNotFoundError(f"no train-*.bin files found in {src}")
    lengths, tokens, problems = [], 0, []
    for path in train_files:
        arr = np.fromfile(path, dtype=np.uint16)
        tokens += len(arr)
        if arr[0] != BOS or arr[-1] != EOS:
            problems.append(f"{path.name}: does not start with BOS / end with EOS")
        if int(arr.max()) >= VOCAB:
            problems.append(f"{path.name}: token id outside the vocabulary")
        lengths.append(doc_lengths(arr))
    lengths = np.concatenate(lengths)
    val_path = src / "val.bin"
    val_tokens = val_path.stat().st_size // 2 if val_path.exists() else 0
    summary = json.loads((src / "summary.json").read_text())
    if tokens != summary["train_tokens"]:
        problems.append(f"train tokens {tokens} != summary {summary['train_tokens']}")
    if val_tokens != summary["val_tokens"]:
        problems.append(f"val tokens {val_tokens} != summary {summary['val_tokens']}")
    hist = np.histogram(lengths, bins=BINS + [np.inf])[0]
    long_docs = lengths > MAX_LEN
    return {
        "docs": int(len(lengths)),
        "train_tokens": int(tokens),
        "val_tokens": int(val_tokens),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "docs_over_ctx_pct": float(100 * long_docs.mean()),
        "tokens_over_ctx_pct": float(100 * lengths[long_docs].sum() / lengths.sum()),
        "hist": [int(x) for x in hist],
        "problems": problems,
    }


def render(results):
    total = sum(r["train_tokens"] for r in results.values())
    planned = sum(PLAN[k] for k in results)
    out = ["# Data statistics", "", "All numbers are measured from the shard files.", ""]
    out += ["## Token counts and training mix", ""]
    out += ["| Source | Docs | Train tokens | Val tokens | Planned tokens | Epochs | Mix |"]
    out += ["|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        epochs = PLAN[name] / r["train_tokens"]
        share = 100 * PLAN[name] / planned
        out.append(
            f"| {name} | {r['docs']:,} | {r['train_tokens']:,} | {r['val_tokens']:,} "
            f"| {PLAN[name]:,} | {epochs:.2f} | {share:.1f}% |"
        )
    out += ["", f"Unique train tokens: {total:,}. Planned training tokens: {planned:,}.", ""]
    out += ["## Document length (tokens, including BOS and EOS)", ""]
    out += ["| Source | Mean | Median | P90 | P99 | Max | Docs over 2048 | Tokens in those |"]
    out += ["|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        out.append(
            f"| {name} | {r['mean']:.0f} | {r['median']:.0f} | {r['p90']:.0f} | {r['p99']:.0f} "
            f"| {r['max']:,} | {r['docs_over_ctx_pct']:.1f}% | {r['tokens_over_ctx_pct']:.1f}% |"
        )
    labels = [f"{a}-{b}" for a, b in zip(BINS[:-1], BINS[1:], strict=True)] + [f"{BINS[-1]}+"]
    out += ["", "## Length histogram (number of documents)", ""]
    out += ["| Source | " + " | ".join(labels) + " |"]
    out += ["|---|" + "---|" * len(labels)]
    for name, r in results.items():
        out.append(f"| {name} | " + " | ".join(f"{x:,}" for x in r["hist"]) + " |")
    problems = [f"{n}: {p}" for n, r in results.items() for p in r["problems"]]
    out += ["", "## Integrity checks", ""]
    out += ["All checks passed." if not problems else "\n".join(f"- {p}" for p in problems)]
    return "\n".join(out) + "\n"


def main():
    root = Path(sys.argv[1])
    results = {name: analyze_source(root / name) for name in PLAN}
    text = render(results)
    out_path = Path(__file__).resolve().parents[1] / "docs" / "data_stats.md"
    out_path.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
