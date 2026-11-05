import time
from concurrent.futures import Future

from engine.bench import Request, run_closed_loop, summarize


class FlakyServer:
    def __init__(self):
        self.calls = 0

    def submit(self, request):
        self.calls += 1
        future = Future()
        if self.calls % 2 == 0:
            future.set_exception(RuntimeError("boom"))
        else:
            future.set_result({"tokens": 4, "first_token_at": time.perf_counter(), "ids": [1]})
        return future


def test_failed_requests_are_counted_and_do_not_stop_the_run():
    requests = [Request(ids=[1]) for _ in range(6)]
    results, wall = run_closed_loop(FlakyServer(), requests, clients=1)
    s = summarize(results, wall)
    assert s["requests"] == 6 and s["failed"] == 3
