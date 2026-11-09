import torch

from engine.bench import Request
from engine.paged import PagedServer
from infer.engine import generate_cached
from model.config import TINY
from model.transformer import Transformer

NEVER = -1  # an end token that is never produced, so every request runs to its length limit


def make(scale=20):
    """scale 20 widens the logit gaps so greedy ties cannot flip; scale 1 keeps logits flat."""
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    with torch.no_grad():
        model.tok_emb.weight.mul_(scale)
    return model


class EvenOnly:
    eos_id = NEVER

    def reset(self):
        pass

    def accepts(self, cid):
        return cid % 2 == 0

    def advance(self, cid):
        pass


class Forced:
    """Accepts exactly one token at each step, so a missed reset shows up in the output."""

    eos_id = NEVER

    def __init__(self):
        self.history = []

    def reset(self):
        self.history = []

    def accepts(self, cid):
        return cid == (len(self.history) * 7 + 2) % 100

    def advance(self, cid):
        self.history.append(cid)


def test_greedy_constrained_request_matches_offline_decoding():
    model = make()
    ids = torch.randint(2, TINY.vocab_size, (6,)).tolist()
    ref = generate_cached(model, ids, 10, NEVER, "cpu", constraint=EvenOnly())[len(ids) :]
    server = PagedServer(model, NEVER, "cpu", num_blocks=12, block_size=4, max_len=32)
    request = Request(ids=ids, max_new=10, constraint=EvenOnly())
    got = server.submit(request).result(timeout=30)["ids"]
    server.close()
    assert got == ref


def test_sampled_requests_obey_their_constraint_and_differ():
    model = make(scale=1)
    ids = torch.randint(2, TINY.vocab_size, (6,)).tolist()
    server = PagedServer(
        model, NEVER, "cpu", num_blocks=30, block_size=4, max_seqs=6, max_len=32, seed=3
    )
    futures = [
        server.submit(Request(ids=ids, max_new=10, constraint=EvenOnly(), temperature=1.5))
        for _ in range(6)
    ]
    outs = [f.result(timeout=30)["ids"] for f in futures]
    server.close()
    assert all(t % 2 == 0 for out in outs for t in out)
    assert len({tuple(o) for o in outs}) > 1


def test_a_preempted_request_restarts_its_constraint():
    model = make()
    prompts = [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19]]
    server = PagedServer(model, NEVER, "cpu", num_blocks=8, block_size=4, max_seqs=3, max_len=32)
    futures = [server.submit(Request(ids=p, max_new=14, constraint=Forced())) for p in prompts]
    got = [f.result(timeout=60)["ids"] for f in futures]
    server.close()
    expected = [(i * 7 + 2) % 100 for i in range(14)]
    assert got == [expected] * 3
    assert server.preemptions > 0
