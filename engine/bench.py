import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Request:
    ids: list
    max_new: int = 128


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def run_closed_loop(server, requests, clients):
    """N client threads; each sends its next request as soon as its previous one finished."""
    results, lock = [], threading.Lock()
    cursor = iter(range(len(requests)))

    def client():
        while True:
            with lock:
                i = next(cursor, None)
            if i is None:
                return
            submitted = time.perf_counter()
            res = server.submit(requests[i]).result()
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
    latency = [r["latency"] for r in results]
    ttft = [r["ttft"] for r in results]
    tokens = sum(r["tokens"] for r in results)
    return {
        "requests": len(results),
        "wall_seconds": round(wall, 2),
        "tokens_per_second": round(tokens / wall, 1),
        "requests_per_second": round(len(results) / wall, 2),
        "latency_p50": round(percentile(latency, 50), 3),
        "latency_p95": round(percentile(latency, 95), 3),
        "ttft_p50": round(percentile(ttft, 50), 3),
        "ttft_p95": round(percentile(ttft, 95), 3),
    }
