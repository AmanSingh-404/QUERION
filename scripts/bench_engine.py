import argparse
import faulthandler
import json
import subprocess
from pathlib import Path

import torch

from data.prompt import format_prompt
from engine.baseline import BaselineServer
from engine.bench import Request, run_closed_loop, summarize
from eval.spider import db_file, schema_statements
from scripts.ask import load
from tokenizer.special_tokens import EOS_ID

DEV = Path(__file__).resolve().parents[1] / "eval" / "spider_dev.jsonl"
OUT = Path(__file__).resolve().parents[1] / "benchmarks"


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


def measure(model, requests, clients, repeats, device):
    """Median run of `repeats` closed-loop runs, plus every run's wall time."""
    runs = []
    for _ in range(repeats):
        server = BaselineServer(model, EOS_ID, device)
        results, wall = run_closed_loop(server, requests, clients)
        server.close()
        runs.append(summarize(results, wall))
    runs.sort(key=lambda r: r["wall_seconds"])
    median = runs[len(runs) // 2]
    return {"clients": clients, **median, "wall_seconds_all": [r["wall_seconds"] for r in runs]}


def main():
    p = argparse.ArgumentParser(description="Closed-loop benchmark of the baseline engine.")
    p.add_argument("--weights", default="SinghBro/querion-100m-sql")
    p.add_argument("--root", default="C:/querion-data/spider_data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--clients", default="1,4,8,16")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--name", default="engine_baseline")
    args = p.parse_args()
    faulthandler.dump_traceback_later(3600, exit=True)

    model, tok = load(args.weights, args.device)
    requests = build_requests(tok, args.root, args.requests, args.max_new)
    warm = BaselineServer(model, EOS_ID, args.device)
    for request in requests[:8]:
        warm.submit(request).result()
    warm.close()

    rows = []
    for clients in [int(c) for c in args.clients.split(",")]:
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        row = measure(model, requests, clients, args.repeats, args.device)
        if args.device == "cuda":
            row["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print(json.dumps(row), flush=True)
        rows.append(row)

    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    result = {
        "engine": "baseline: one request at a time, simple KV cache, greedy, no constraint",
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "workload": f"{len(requests)} Spider dev prompts, max {args.max_new} new tokens",
        "method": f"median of {args.repeats} runs after an 8-request warm-up",
        "collected_at_commit": git.stdout.strip(),
        "rows": rows,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
