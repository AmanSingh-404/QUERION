import time
from concurrent.futures import Future

import torch

from engine.baseline import BaselineServer
from engine.bench import Request, percentile, run_closed_loop, summarize
from infer.engine import generate_cached
from model.config import TINY
from model.transformer import Transformer


class InstantServer:
    def submit(self, request):
        future = Future()
        time.sleep(0.01)
        future.set_result({"tokens": 5, "first_token_at": time.perf_counter(), "ids": [1]})
        return future


def test_percentile_and_summary():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 95) == 0.0
    results = [
        {"latency": 1.0, "ttft": 0.5, "tokens": 10},
        {"latency": 3.0, "ttft": 1.5, "tokens": 30},
    ]
    s = summarize(results, 4.0)
    assert s["requests"] == 2 and s["tokens_per_second"] == 10.0
    assert s["latency_p50"] == 2.0


def test_closed_loop_runs_every_request_once():
    requests = [Request(ids=[1, 2, 3]) for _ in range(12)]
    results, wall = run_closed_loop(InstantServer(), requests, clients=4)
    assert len(results) == 12 and wall > 0
    assert all(r["latency"] >= 0.01 for r in results)


def test_baseline_server_matches_greedy_decoding_and_handles_concurrency():
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    ids = torch.randint(2, TINY.vocab_size, (6,)).tolist()
    server = BaselineServer(model, 1, "cpu")
    futures = [server.submit(Request(ids=ids, max_new=10)) for _ in range(3)]
    results = [f.result() for f in futures]
    server.close()
    ref = generate_cached(model, ids, 10, 1, "cpu")[len(ids) :]
    assert all(r["ids"] == ref for r in results)
