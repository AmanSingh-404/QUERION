import argparse
import faulthandler
import json
import threading
import time
from pathlib import Path

import torch

from engine.bench import percentile
from engine.pipeline import Question, SqlPipeline
from engine.prefix import PrefixServer
from eval.compare import exec_match
from eval.spider import db_file, schema_statements
from infer.constrain import token_texts
from infer.quantize import quantize_int8
from scripts.ask import load
from scripts.collect_benchmarks import summarize
from tokenizer.special_tokens import EOS_ID

DEV = Path(__file__).resolve().parents[1] / "eval" / "spider_dev.jsonl"
EVAL_DIR = Path("C:/querion-runs/eval")
BENCH = Path(__file__).resolve().parents[1] / "benchmarks"


def answer(pipeline, item, root):
    db = db_file(root, item["db_id"])
    started = time.perf_counter()
    try:
        job = Question(schema_statements(db), item["question"], db)
        pred = pipeline.submit(job).result()["sql"]
        result = exec_match(db, pred, item["sql"])
    except ValueError:  # the prompt does not fit the context window
        pred, result = None, {"status": "prompt_too_long", "match": False, "error": None}
    except Exception as e:
        pred, result = None, {"status": "engine_error", "match": False, "error": type(e).__name__}
    return {**item, "pred": pred, **result}, time.perf_counter() - started


def main():
    p = argparse.ArgumentParser(description="Spider dev through the serving engine.")
    p.add_argument("--weights", default="SinghBro/querion-100m-sql")
    p.add_argument("--root", default="C:/querion-data/spider_data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--requests", type=int, default=0, help="0 means the whole dev set")
    p.add_argument("--clients", type=int, default=16)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--slots", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2048)
    p.add_argument("--compare", default=None, help="saved run to count differing predictions")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--name", default="engine_vote")
    args = p.parse_args()
    faulthandler.dump_traceback_later(7200, exit=True)

    model, tok = load(args.weights, args.device)
    if args.int8:
        model = quantize_int8(model)
    texts = token_texts(tok)
    server = PrefixServer(model, EOS_ID, args.device, num_blocks=args.blocks, max_seqs=args.slots)
    pipeline = SqlPipeline(
        server,
        tok,
        texts,
        EOS_ID,
        samples=args.samples,
        temperature=args.temperature,
        workers=args.clients,
    )
    with open(DEV, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if args.requests:
        items = items[:: max(1, len(items) // args.requests)][: args.requests]

    records, latencies = [None] * len(items), []
    lock = threading.Lock()
    cursor = iter(range(len(items)))

    def client():
        while True:
            with lock:
                i = next(cursor, None)
            if i is None:
                return
            record, seconds = answer(pipeline, items[i], args.root)
            with lock:
                records[i] = record
                latencies.append(seconds)

    threads = [threading.Thread(target=client) for _ in range(args.clients)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - start
    pipeline.close()
    server.close()

    summary = summarize(records)
    summary.update(
        {
            "clients": args.clients,
            "int8": args.int8,
            "samples": args.samples,
            "wall_seconds": round(wall),
            "questions_per_second": round(len(items) / wall, 2),
            "latency_p50": round(percentile(latencies, 50), 2),
            "latency_p95": round(percentile(latencies, 95), 2),
            "prefix_hit_rate": round(server.hit_rate, 3),
            "preemptions": server.preemptions,
            "kv_peak_used_gb": round(server.kv_peak_gb, 3),
        }
    )
    if args.device == "cuda":
        summary["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if args.compare:
        with open(EVAL_DIR / f"spider_dev_{args.compare}.jsonl", encoding="utf-8") as f:
            other = {r["id"]: r["pred"] for r in map(json.loads, f)}
        summary[f"pred_differs_from_{args.compare}"] = sum(
            r["pred"] != other.get(r["id"]) for r in records
        )
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / f"spider_dev_{args.name}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    BENCH.mkdir(exist_ok=True)
    path = BENCH / f"pipeline_{args.name}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
