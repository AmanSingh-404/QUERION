import argparse
import json
import time

import torch

from model.config import QUERION_100M, TINY
from model.transformer import Transformer


def run_one(cfg, batch, seq, checkpointing, device, steps=5):
    model = Transformer(cfg).to(device)
    model.grad_checkpointing = checkpointing
    x = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    amp = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")

    def step():
        with amp:
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    return {
        "tokens_per_sec": round(batch * seq * steps / elapsed),
        "peak_mem_gb": None if peak is None else round(peak, 2),
        "status": "ok",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["100m", "tiny"], default="100m")
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--batches", default="1,2,4,8")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = QUERION_100M if args.config == "100m" else TINY
    seq = min(args.seq, cfg.max_seq_len)
    results = []
    for checkpointing in (False, True):
        for batch in [int(b) for b in args.batches.split(",")]:
            row = {"checkpointing": checkpointing, "micro_batch": batch, "seq": seq}
            try:
                row.update(run_one(cfg, batch, seq, checkpointing, device))
            except torch.cuda.OutOfMemoryError:
                row.update({"tokens_per_sec": None, "peak_mem_gb": None, "status": "oom"})
            torch.cuda.empty_cache() if device == "cuda" else None
            results.append(row)
            print(json.dumps(row), flush=True)
    ok = [r for r in results if r["status"] == "ok"]
    out = {
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "config": args.config,
        "results": results,
    }
    if ok:
        best = max(ok, key=lambda r: r["tokens_per_sec"])
        out["best"] = best
        out["hours_for_2b_tokens"] = round(2e9 / best["tokens_per_sec"] / 3600, 1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
