import random

import pytest
import torch

from engine.baseline import BaselineServer
from engine.bench import Request
from engine.paged import PagedServer
from engine.prefix import PrefixServer
from model.config import TINY
from model.transformer import Transformer

NEVER = -1  # an end token that is never produced, so every request runs to its length limit


def make():
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    with torch.no_grad():
        model.tok_emb.weight.mul_(20)  # widen logit gaps so greedy ties cannot flip
    return model


def server_for(model, blocks=12, seqs=2, cls=PagedServer):
    return cls(model, NEVER, "cpu", num_blocks=blocks, block_size=4, max_seqs=seqs, max_len=32)


def test_cancelling_a_request_does_not_stop_the_server_or_leak_blocks():
    server = server_for(make())
    first = server.submit(Request(ids=[2, 3, 4, 5, 6], max_new=12))
    doomed = server.submit(Request(ids=[7, 8, 9], max_new=12))
    assert doomed.cancel()
    later = server.submit(Request(ids=[10, 11, 12], max_new=6))
    assert first.result(timeout=30)["tokens"] == 12
    assert later.result(timeout=30)["tokens"] == 6
    server.close()
    assert doomed.cancelled()
    assert server.alloc.free_blocks == 12


def test_a_request_that_runs_out_of_time_fails_and_frees_its_blocks():
    server = server_for(make())
    slow = server.submit(Request(ids=[2, 3, 4], max_new=20, timeout=0.0))
    fine = server.submit(Request(ids=[5, 6, 7], max_new=4))
    with pytest.raises(TimeoutError):
        slow.result(timeout=30)
    assert fine.result(timeout=30)["tokens"] == 4
    server.close()
    assert server.timeouts == 1
    assert server.alloc.free_blocks == 12


def test_out_of_blocks_preempts_the_youngest_request_and_outputs_stay_correct():
    model = make()
    prompts = [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19]]
    base = BaselineServer(model, NEVER, "cpu")
    ref = [base.submit(Request(ids=p, max_new=14)).result()["ids"] for p in prompts]
    base.close()
    server = server_for(model, blocks=8, seqs=3)
    futures = [server.submit(Request(ids=p, max_new=14)) for p in prompts]
    got = [f.result(timeout=60)["ids"] for f in futures]
    server.close()
    assert got == ref
    assert server.preemptions > 0
    assert server.alloc.free_blocks == 8


def test_an_internal_error_fails_every_request_instead_of_hanging():
    server = server_for(make())

    def broken():
        raise RuntimeError("boom")

    server._step = broken
    failing = server.submit(Request(ids=[2, 3, 4], max_new=5))
    with pytest.raises(RuntimeError):
        failing.result(timeout=30)
    server.close()
    late = server.submit(Request(ids=[2], max_new=2))
    with pytest.raises(RuntimeError):
        late.result(timeout=5)


@pytest.mark.parametrize("cls", [PagedServer, PrefixServer])
def test_stress_mix_of_requests_never_leaks_blocks(cls):
    model = make()
    rng = random.Random(7)
    server = server_for(model, blocks=10, seqs=4, cls=cls)
    shared = [rng.randrange(2, 100) for _ in range(8)]
    futures = []
    for _ in range(60):
        roll = rng.random()
        if roll < 0.1:
            ids = [rng.randrange(2, 100) for _ in range(40)]  # longer than max_len: refused
        elif roll < 0.6:
            ids = shared + [rng.randrange(2, 100) for _ in range(rng.randrange(1, 6))]
        else:
            ids = [rng.randrange(2, 100) for _ in range(rng.randrange(1, 12))]
        timeout = 0.0 if rng.random() < 0.1 else None
        future = server.submit(Request(ids=ids, max_new=rng.randrange(1, 10), timeout=timeout))
        if rng.random() < 0.15:
            future.cancel()
        futures.append(future)
    server.close()
    assert all(f.done() for f in futures)
    finished = [f for f in futures if not f.cancelled() and f.exception() is None]
    assert len(finished) > 0
    assert server.alloc.free_blocks == 10
    assert all(r == 0 for r in server.alloc.refs)
