import argparse
import faulthandler
import gc
import json
import statistics
import subprocess
from pathlib import Path

import torch

from engine.bench import run_closed_loop, summarize
from scripts.ask import load
from scripts.bench_engine import WARMUP, build_requests, make_server

OUT = Path(__file__).resolve().parents[1] / "benchmarks"


def median_run(runs):
    """The run with the median tokens per second."""
    return sorted(runs, key=lambda r: r["tokens_per_second"])[len(runs) // 2]


def run_once(kind, model, requests, clients, device, slots):
    gc.collect()
    server = make_server(kind, model, device, slots)
    results, wall = run_closed_loop(server, requests, clients)
    server.close()
    del server
    return summarize(results, wall)


def main():
    p = argparse.ArgumentParser(description="Interleaved A/B benchmark: baseline vs batched.")
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--weights", default="SinghBro/querion-100m-sql")
    p.add_argument("--root", default="C:/querion-data/spider_data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--clients", default="1,4,16")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--name", default="engine_compare")
    args = p.parse_args()
    faulthandler.dump_traceback_later(3600, exit=True)

    model, tok = load(args.weights, args.device)
    requests = build_requests(tok, args.root, args.requests, args.max_new)
    for kind in ("baseline", "batched"):
        warm = make_server(kind, model, args.device, args.slots)
        for request in requests[:WARMUP]:
            warm.submit(request).result()
        warm.close()
        del warm

    rows = []
    for clients in [int(c) for c in args.clients.split(",")]:
        runs = {"baseline": [], "batched": []}
        ratios = []
        for i in range(args.repeats):
            order = ("baseline", "batched") if i % 2 == 0 else ("batched", "baseline")
            done = {}
            for kind in order:
                done[kind] = run_once(kind, model, requests, clients, args.device, args.slots)
                runs[kind].append(done[kind])
            ratios.append(
                done["batched"]["tokens_per_second"] / done["baseline"]["tokens_per_second"]
            )
        row = {
            "clients": clients,
            "baseline": median_run(runs["baseline"]),
            "batched": median_run(runs["batched"]),
            "speedup_median": round(statistics.median(ratios), 2),
            "speedup_all": [round(r, 2) for r in ratios],
        }
        print(json.dumps(row), flush=True)
        rows.append(row)

    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    result = {
        "method": (
            f"interleaved: {args.repeats} repeats, each runs both engines back to back with "
            "alternating order; speedup is batched tokens/s over baseline tokens/s of the "
            "same repeat; rows show the median run of each engine"
        ),
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "workload": f"{len(requests)} Spider dev prompts, greedy, max {args.max_new} new tokens",
        "slots": args.slots,
        "collected_at_commit": git.stdout.strip(),
        "rows": rows,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
