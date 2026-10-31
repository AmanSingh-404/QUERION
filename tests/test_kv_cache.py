from dataclasses import replace

import pytest
import torch

from infer.engine import KVCache, generate_cached
from model.config import TINY
from model.transformer import Transformer
from train.sample import generate

CONFIGS = [
    TINY,
    replace(TINY, pos_emb="learned"),
    replace(TINY, n_kv_heads=4),
]


def make(cfg):
    torch.manual_seed(0)
    return Transformer(cfg).eval()


@pytest.mark.parametrize("cfg", CONFIGS)
def test_cached_logits_match_full_forward(cfg):
    model = make(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full = model(ids)[0]
        cache = KVCache(cfg, 1, 12, "cpu", torch.float32)
        last = model(ids[:, :5], cache=cache, pos=0)[0]
        assert torch.allclose(last[:, 0], full[:, 4], atol=1e-4)
        for pos in range(5, 12):
            step = model(ids[:, pos : pos + 1], cache=cache, pos=pos)[0]
            assert torch.allclose(step[:, 0], full[:, pos], atol=1e-4)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_cached_greedy_matches_uncached(cfg):
    model = make(cfg)
    ids = torch.randint(0, cfg.vocab_size, (7,)).tolist()
    slow = generate(model, ids, 14, -1, "cpu")
    fast = generate_cached(model, ids, 14, -1, "cpu")
    assert slow == fast


def test_full_cache_stops_instead_of_crashing():
    model = make(TINY)
    ids = torch.randint(0, TINY.vocab_size, (30,)).tolist()
    assert len(generate_cached(model, ids, 10, -1, "cpu")) == 33
