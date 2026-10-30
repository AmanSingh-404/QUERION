import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from data.leakage import build_needles
from train.sft_data import (
    drop_contaminated,
    encode_example,
    gretel_examples,
    spider_examples,
    split_examples,
    synth_examples,
)
from train.train import TOK_PATH

DEV = Path(__file__).resolve().parents[1] / "eval" / "spider_dev.jsonl"


def main():
    spider_root = sys.argv[1] if len(sys.argv) > 1 else "C:/querion-data/spider_data"
    gretel_path = sys.argv[2] if len(sys.argv) > 2 else "C:/querion-data/gretel_train.jsonl"
    out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else "C:/querion-data/sft")
    out_dir.mkdir(parents=True, exist_ok=True)

    spider, skipped = spider_examples(spider_root)
    examples = spider + gretel_examples(gretel_path) + synth_examples()
    with open(DEV, encoding="utf-8") as f:
        needles = build_needles([json.loads(line) for line in f])
    examples, dropped = drop_contaminated(examples, needles)

    tok = Tokenizer.from_file(str(TOK_PATH))
    kept, too_long = [], Counter()
    for ex in examples:
        enc = encode_example(tok, ex)
        if enc is None:
            too_long[ex["source"]] += 1
        else:
            ex["tokens"] = len(enc[0]) + 1
            kept.append(ex)
    train, val = split_examples(kept)

    stats = {"spider_skipped": skipped, "contaminated_dropped": dropped, "too_long": dict(too_long)}
    for name, rows in (("train", train), ("val", val)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        per_source = {}
        for src in ("spider", "gretel", "synth"):
            lengths = [e["tokens"] for e in rows if e["source"] == src]
            if lengths:
                per_source[src] = {
                    "examples": len(lengths),
                    "tokens": sum(lengths),
                    "median_length": statistics.median(lengths),
                    "max_length": max(lengths),
                }
        stats[name] = per_source
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
