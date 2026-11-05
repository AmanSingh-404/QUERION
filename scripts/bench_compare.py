import argparse
import faulthandler
import gc
import json
import statistics
import subprocess
from pathlib import Path

import torch

from engine.baseline import BaselineServer
from engine.batched import BatchedServer
from engine.bench import run_closed_loop, summarize
from engine.paged import PagedServer
from scripts.ask import load
from scripts.bench_engine import WARMUP, build_requests
from tokenizer.special_tokens import EOS_ID

OUT = Path(__file__).resolve().parents[1] / "benchmarks"
KINDS = ["baseline", "batched", "paged"]


def make_server(kind, model, device, slots, blocks):
    if kind == "paged":
        return PagedServer(model, EOS_ID, device, num_blocks=blocks, max_seqs=slots)
    if kind == "batched":
        return BatchedServer(model, EOS_ID, device, slots=slots)
    return BaselineServer(model, EOS_ID, device)


def kv_stats(server):
    pool = getattr(server, "pool", None)
    if pool is None:
        return {}
    reserved = sum(t.numel() * t.element_size() for t in pool.k + pool.v) / 1e9
    stats = {"kv_reserved_gb": round(reserved, 3)}
    if hasattr(server, "kv_peak_gb"):
        stats["kv_peak_used_gb"] = round(server.kv_peak_gb, 3)
        stats["fragmentation"] = round(server.fragmentation, 3)
    return stats


def median_run(runs):
    """The run with the median tokens per second."""
    return sorted(runs, key=lambda r: r["tokens_per_second"])[len(runs) // 2]


def run_once(kind, model, requests, clients, device, slots, blocks):
    gc.collect()
    server = make_server(kind, model, device, slots, blocks)
    results, wall = run_closed_loop(server, requests, clients)
    server.close()
    row = {**summarize(results, wall), **kv_stats(server)}
    del server
    return row


def main():
    p = argparse.ArgumentParser(description="Interleaved A/B benchmark of two engines.")
    p.add_argument("--a", choices=KINDS, default="baseline")
    p.add_argument("--b", choices=KINDS, default="batched")
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--blocks", type=int, default=1024)
    p.add_argument("--weights", default="SinghBro/querion-100m-sql")
    p.add_argument("--root", default="C:/querion-data/spider_data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--clients", default="1,4,16")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    faulthandler.dump_traceback_later(3600, exit=True)

    model, tok = load(args.weights, args.device)
    requests = build_requests(tok, args.root, args.requests, args.max_new)
    for kind in (args.a, args.b):
        warm = make_server(kind, model, args.device, args.slots, args.blocks)
        for request in requests[:WARMUP]:
            warm.submit(request).result()
        warm.close()
        del warm

    rows = []
    for clients in [int(c) for c in args.clients.split(",")]:
        runs = {args.a: [], args.b: []}
        ratios = []
        for i in range(args.repeats):
            order = (args.a, args.b) if i % 2 == 0 else (args.b, args.a)
            done = {}
            for kind in order:
                done[kind] = run_once(
                    kind, model, requests, clients, args.device, args.slots, args.blocks
                )
                runs[kind].append(done[kind])
            ratios.append(done[args.b]["tokens_per_second"] / done[args.a]["tokens_per_second"])
        row = {
            "clients": clients,
            args.a: median_run(runs[args.a]),
            args.b: median_run(runs[args.b]),
            "speedup_median": round(statistics.median(ratios), 2),
            "speedup_all": [round(r, 2) for r in ratios],
        }
        print(json.dumps(row), flush=True)
        rows.append(row)

    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    result = {
        "method": (
            f"interleaved: {args.repeats} repeats, each runs both engines back to back with "
            f"alternating order; speedup is {args.b} tokens/s over {args.a} tokens/s of the "
            "same repeat; rows show the median run of each engine"
        ),
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "workload": f"{len(requests)} Spider dev prompts, greedy, max {args.max_new} new tokens",
        "slots": args.slots,
        "blocks": args.blocks if "paged" in (args.a, args.b) else None,
        "collected_at_commit": git.stdout.strip(),
        "rows": rows,
    }
    OUT.mkdir(exist_ok=True)
    name = args.name or f"compare_{args.a}_vs_{args.b}"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
