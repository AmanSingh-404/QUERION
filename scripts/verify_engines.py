import argparse

import torch

from scripts.ask import load
from scripts.bench_compare import make_server
from scripts.bench_engine import build_requests


def outputs(kind, model, requests, device, slots, blocks, sequential):
    server = make_server(kind, model, device, slots, blocks)
    if sequential:
        got = [server.submit(r).result()["ids"] for r in requests]
    else:
        futures = [server.submit(r) for r in requests]
        got = [f.result()["ids"] for f in futures]
    server.close()
    return got, server


def main():
    p = argparse.ArgumentParser(description="Compare engine outputs with the baseline.")
    p.add_argument("--weights", default="SinghBro/querion-100m-sql")
    p.add_argument("--root", default="C:/querion-data/spider_data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--blocks", type=int, default=512)
    args = p.parse_args()

    model, tok = load(args.weights, args.device)
    requests = build_requests(tok, args.root, args.requests, args.max_new)
    ref, _ = outputs("baseline", model, requests, args.device, args.slots, args.blocks, True)
    for kind in ("batched", "paged"):
        got, server = outputs(kind, model, requests, args.device, args.slots, args.blocks, False)
        differ = sum(a != b for a, b in zip(ref, got, strict=True))
        print(f"{kind}: {differ} of {len(requests)} outputs differ from the baseline")
        if kind == "paged":
            free = server.alloc.free_blocks
            print(f"paged: {free} of {server.alloc.num_blocks} blocks free after the run")


if __name__ == "__main__":
    main()
