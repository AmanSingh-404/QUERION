from dataclasses import replace

import pytest
import torch

from engine.baseline import BaselineServer
from engine.batched import BatchedServer, SlotPool, decode_step
from engine.bench import Request
from infer.engine import KVCache
from model.config import TINY
from model.transformer import Transformer

CONFIGS = [TINY, replace(TINY, pos_emb="learned"), replace(TINY, n_kv_heads=4)]


def make(cfg):
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    with torch.no_grad():
        model.tok_emb.weight.mul_(20)  # widen logit gaps so greedy ties cannot flip
    return model


def prefill_into(model, pool, slot, ids):
    cache = KVCache(model.cfg, 1, len(ids), "cpu", torch.float32)
    model(torch.tensor([ids]), cache=cache, pos=0)
    for layer in range(model.cfg.n_layers):
        pool.k[layer][slot, :, : len(ids)] = cache.k[layer][0]
        pool.v[layer][slot, :, : len(ids)] = cache.v[layer][0]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_decode_step_matches_full_forward_for_mixed_lengths(cfg):
    model = make(cfg)
    pool = SlotPool(cfg, 3, 16, "cpu", torch.float32)
    a = torch.randint(2, cfg.vocab_size, (5,)).tolist()
    b = torch.randint(2, cfg.vocab_size, (9,)).tolist()
    with torch.no_grad():
        prefill_into(model, pool, 2, a)
        prefill_into(model, pool, 0, b)
        tokens = torch.tensor([[7], [11]])
        out = decode_step(model, pool, tokens, torch.tensor([2, 0]), torch.tensor([5, 9]))
        for row, ids, tok in ((0, a, 7), (1, b, 11)):
            ref = model(torch.tensor([ids + [tok]]))[0][0, -1]
            assert torch.allclose(out[row], ref, atol=1e-4)


def test_batched_server_matches_baseline_with_joins_and_leaves():
    model = make(TINY)
    torch.manual_seed(1)
    lengths = (3, 8, 5, 4, 7, 6)
    prompts = [torch.randint(2, TINY.vocab_size, (n,)).tolist() for n in lengths]
    base = BaselineServer(model, 1, "cpu")
    ref = [base.submit(Request(ids=p, max_new=8)).result()["ids"] for p in prompts]
    base.close()
    server = BatchedServer(model, 1, "cpu", slots=3, max_len=32)
    futures = [server.submit(Request(ids=p, max_new=8)) for p in prompts]
    got = [f.result()["ids"] for f in futures]
    server.close()
    assert got == ref


def test_oversized_prompt_fails_without_stopping_the_server():
    model = make(TINY)
    server = BatchedServer(model, 1, "cpu", slots=2, max_len=16)
    bad = server.submit(Request(ids=list(range(2, 22)), max_new=4))
    good = server.submit(Request(ids=[2, 3, 4], max_new=4))
    with pytest.raises(ValueError):
        bad.result()
    assert good.result()["tokens"] >= 1
    server.close()
