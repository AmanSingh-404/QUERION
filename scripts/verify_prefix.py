import argparse
import json
from pathlib import Path

import torch

from data.prompt import format_prompt
from engine.baseline import BaselineServer
from engine.bench import Request
from engine.prefix import PrefixServer
from eval.spider import db_file, schema_statements
from scripts.ask import load
from tokenizer.special_tokens import EOS_ID

DEV = Path(__file__).resolve().parents[1] / "eval" / "spider_dev.jsonl"


def clustered_requests(tok, root, n, max_new):
    """The first n dev questions: consecutive ones share a database, hence a schema prefix."""
    with open(DEV, encoding="utf-8") as f:
        items = [json.loads(line) for line in f][:n]
    requests = []
    for it in items:
        statements = schema_statements(db_file(root, it["db_id"]))
        ids = tok.encode(format_prompt(statements, it["question"])).ids
        if len(ids) <= 1800:
            requests.append(Request(ids=ids, max_new=max_new))
    return requests


def main():
    p = argparse.ArgumentParser(description="Check the prefix cache against the baseline.")
    p.add_argument("--weights", default="SinghBro/querion-100m-sql")
    p.add_argument("--root", default="C:/querion-data/spider_data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--blocks", type=int, default=1024)
    args = p.parse_args()

    model, tok = load(args.weights, args.device)
    requests = clustered_requests(tok, args.root, args.requests, args.max_new)
    base = BaselineServer(model, EOS_ID, args.device)
    ref = [base.submit(r).result()["ids"] for r in requests]
    base.close()
    del base

    server = PrefixServer(model, EOS_ID, args.device, num_blocks=args.blocks, max_seqs=args.slots)
    futures = [server.submit(r) for r in requests]
    got = [f.result()["ids"] for f in futures]
    server.close()
    differ = sum(a != b for a, b in zip(ref, got, strict=True))
    print(f"prefix: {differ} of {len(requests)} outputs differ from the baseline")
    print(
        f"prompt tokens {server.prompt_tokens}, reused from cache {server.cached_tokens} "
        f"({server.hit_rate:.1%})"
    )
    print(f"{server.alloc.free_blocks} of {server.alloc.num_blocks} blocks free after the run")


if __name__ == "__main__":
    main()
