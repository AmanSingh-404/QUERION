import argparse
import faulthandler
import json
import time
from pathlib import Path

from data.prompt import format_prompt
from eval.compare import exec_match
from eval.spider import schema_statements
from infer.constrain import SqlConstraint, token_texts
from infer.engine import generate_cached
from scripts.ask import load
from tokenizer.special_tokens import EOS_ID

SMOKE = Path(__file__).resolve().parents[1] / "tests" / "smoke"


def main():
    p = argparse.ArgumentParser(description="Fast end-to-end check of weights and engine.")
    p.add_argument("--weights", default="SinghBro/querion-100m-sql")
    p.add_argument("--min-accuracy", type=float, default=0.0)
    args = p.parse_args()
    faulthandler.dump_traceback_later(600, exit=True)

    model, tok = load(args.weights, "cpu")
    texts = token_texts(tok)
    with open(SMOKE / "items.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    start = time.time()
    matches, failures = 0, []
    for n, it in enumerate(items, 1):
        db = SMOKE / it["db"]
        statements = schema_statements(db)
        ids = tok.encode(format_prompt(statements, it["question"])).ids
        constraint = SqlConstraint("\n".join(statements), texts, EOS_ID)
        out = generate_cached(model, ids, 160, EOS_ID, "cpu", constraint=constraint)[len(ids) :]
        if out and out[-1] == EOS_ID:
            out = out[:-1]
        pred = tok.decode(out, skip_special_tokens=False).strip().rstrip(";")
        print(n, "of", len(items), round(time.time() - start), "s", flush=True)
        if exec_match(db, pred, it["sql"])["match"]:
            matches += 1
        else:
            failures.append({"question": it["question"], "gold": it["sql"], "pred": pred})
    faulthandler.cancel_dump_traceback_later()
    accuracy = matches / len(items)
    summary = {
        "items": len(items),
        "matches": matches,
        "accuracy": round(accuracy, 4),
        "seconds": round(time.time() - start),
    }
    print(json.dumps(summary))
    for failure in failures[:3]:
        print(json.dumps(failure))
    if accuracy < args.min_accuracy:
        raise SystemExit(f"smoke accuracy {accuracy:.1%} is below {args.min_accuracy:.1%}")


if __name__ == "__main__":
    main()
