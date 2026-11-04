import argparse
import faulthandler
import gc
import json
import subprocess
from pathlib import Path

import torch

from data.prompt import format_prompt
from engine.baseline import BaselineServer
from engine.batched import BatchedServer
from engine.bench import Request, run_closed_loop, summarize
from eval.spider import db_file, schema_statements
from scripts.ask import load
from tokenizer.special_tokens import EOS_ID

DEV = Path(__file__).resolve().parents[1] / "eval" / "spider_dev.jsonl"
OUT = Path(__file__).resolve().parents[1] / "benchmarks"
ENGINES = {
    "baseline": "baseline: one request at a time, simple KV cache, greedy, no constraint",
    "batched": "continuous batching over a slot pool, greedy, no constraint",
}
WARMUP = 16


def build_requests(tok, root, n, max_new):
    with open(DEV, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    items = items[:: max(1, len(items) // n)][:n]
    requests = []
    for it in items:
        statements = schema_statements(db_file(root, it["db_id"]))
        ids = tok.encode(format_prompt(statements, it["question"])).ids
        if len(ids) <= 1800:
            requests.append(Request(ids=ids, max_new=max_new))
    return requests


def make_server(kind, model, device, slots):
    if kind == "batched":
        return BatchedServer(model, EOS_ID, device, slots=slots)
    return BaselineServer(model, EOS_ID, device)


def kv_reserved_gb(server):
    pool = getattr(server, "pool", None)
    if pool is None:
        return None
    return round(sum(t.numel() * t.element_size() for t in pool.k + pool.v) / 1e9, 2)


def measure(kind, model, requests, clients, repeats, device, slots):
    """Median run of `repeats` closed-loop runs, plus every run's wall time."""
    runs, kv = [], None
    for _ in range(repeats):
        server = make_server(kind, model, device, slots)
        results, wall = run_closed_loop(server, requests, clients)
        server.close()
        kv = kv_reserved_gb(server)
        del server
        runs.append(summarize(results, wall))
    runs.sort(key=lambda r: r["wall_seconds"])
    median = runs[len(runs) // 2]
    row = {"clients": clients, **median, "wall_seconds_all": [r["wall_seconds"] for r in runs]}
    if kv is not None:
        row["kv_reserved_gb"] = kv
    return row


def verify(model, requests, device, slots):
    """How many real prompts give different tokens on the batched and the baseline engine."""
    base = BaselineServer(model, EOS_ID, device)
    ref = [base.submit(r).result()["ids"] for r in requests]
    base.close()
    del base
    server = BatchedServer(model, EOS_ID, device, slots=slots)
    futures = [server.submit(r) for r in requests]
    got = [f.result()["ids"] for f in futures]
    server.close()
    del server
    return sum(a != b for a, b in zip(ref, got, strict=True))


def main():
    p = argparse.ArgumentParser(description="Closed-loop benchmark of a serving engine.")
    p.add_argument("--engine", choices=list(ENGINES), default="baseline")
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--weights", default="SinghBro/querion-100m-sql")
    p.add_argument("--root", default="C:/querion-data/spider_data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--clients", default="1,4,8,16")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--name", default=None)
    args = p.parse_args()
    faulthandler.dump_traceback_later(3600, exit=True)

    model, tok = load(args.weights, args.device)
    requests = build_requests(tok, args.root, args.requests, args.max_new)
    warm = make_server(args.engine, model, args.device, args.slots)
    for request in requests[:WARMUP]:
        warm.submit(request).result()
    warm.close()
    del warm

    rows = []
    for clients in [int(c) for c in args.clients.split(",")]:
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        row = measure(args.engine, model, requests, clients, args.repeats, args.device, args.slots)
        if args.device == "cuda":
            row["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print(json.dumps(row), flush=True)
        rows.append(row)

    differing = None
    if args.verify and args.engine == "batched":
        gc.collect()
        differing = verify(model, requests, args.device, args.slots)
        print(json.dumps({"outputs_differing_from_baseline": differing, "of": len(requests)}))

    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    result = {
        "engine": ENGINES[args.engine],
        "slots": args.slots if args.engine == "batched" else None,
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "workload": f"{len(requests)} Spider dev prompts, max {args.max_new} new tokens",
        "method": f"median of {args.repeats} runs after a {WARMUP}-request warm-up",
        "outputs_differing_from_baseline": differing,
        "collected_at_commit": git.stdout.strip(),
        "rows": rows,
    }
    OUT.mkdir(exist_ok=True)
    name = args.name or f"engine_{args.engine}"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
