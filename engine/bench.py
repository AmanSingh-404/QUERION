import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Request:
    ids: list
    max_new: int = 128
    timeout: float | None = None
    constraint: object = None
    temperature: float = 0.0


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def run_closed_loop(server, requests, clients):
    """N client threads; each sends its next request as soon as its previous one finished.

    A request that fails is recorded as an error entry instead of stopping its client."""
    results, lock = [], threading.Lock()
    cursor = iter(range(len(requests)))

    def client():
        while True:
            with lock:
                i = next(cursor, None)
            if i is None:
                return
            submitted = time.perf_counter()
            try:
                res = server.submit(requests[i]).result()
            except Exception as e:
                with lock:
                    results.append({"error": type(e).__name__})
                continue
            done = time.perf_counter()
            with lock:
                results.append(
                    {
                        "latency": done - submitted,
                        "ttft": res["first_token_at"] - submitted,
                        "tokens": res["tokens"],
                    }
                )

    threads = [threading.Thread(target=client) for _ in range(clients)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.perf_counter() - start


def summarize(results, wall):
    ok = [r for r in results if "error" not in r]
    latency = [r["latency"] for r in ok]
    ttft = [r["ttft"] for r in ok]
    tokens = sum(r["tokens"] for r in ok)
    return {
        "requests": len(results),
        "failed": len(results) - len(ok),
        "wall_seconds": round(wall, 2),
        "tokens_per_second": round(tokens / wall, 1),
        "requests_per_second": round(len(ok) / wall, 2),
        "latency_p50": round(percentile(latency, 50), 3),
        "latency_p95": round(percentile(latency, 95), 3),
        "ttft_p50": round(percentile(ttft, 50), 3),
        "ttft_p95": round(percentile(ttft, 95), 3),
    }
