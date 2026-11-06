from dataclasses import replace

import pytest
import torch

from engine.baseline import BaselineServer
from engine.bench import Request
from engine.paged import BlockPool, OutOfBlocks
from engine.prefix import CachingAllocator, PrefixServer, prefill_paged
from model.config import TINY
from model.transformer import Transformer

CONFIGS = [TINY, replace(TINY, pos_emb="learned"), replace(TINY, n_kv_heads=4)]


def make(cfg):
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    with torch.no_grad():
        model.tok_emb.weight.mul_(20)  # widen logit gaps so greedy ties cannot flip
    return model


@pytest.mark.parametrize("cfg", CONFIGS)
def test_prefill_paged_matches_full_forward_and_reuses_cached_blocks(cfg):
    model = make(cfg)
    pool = BlockPool(cfg, 12, 4, "cpu", torch.float32)
    first = torch.randint(2, cfg.vocab_size, (11,)).tolist()
    second = first[:8] + torch.randint(2, cfg.vocab_size, (4,)).tolist()
    with torch.no_grad():
        got = prefill_paged(model, pool, first, [5, 1, 9], 0)
        assert torch.allclose(got[0], model(torch.tensor([first]))[0][0, -1], atol=1e-4)
        # the second prompt shares its first two blocks (8 tokens) with the first one
        got = prefill_paged(model, pool, second, [5, 1, 3], 8)
        assert torch.allclose(got[0], model(torch.tensor([second]))[0][0, -1], atol=1e-4)


def test_cached_blocks_are_found_by_prefix_and_survive_being_freed():
    alloc = CachingAllocator(6)
    ids = list(range(12))
    blocks = [alloc.allocate() for _ in range(3)]
    alloc.register(ids, 4, blocks)
    for b in blocks:
        alloc.free(b)
    assert alloc.free_blocks == 6  # cached blocks still count as reusable
    assert alloc.match(ids, 4, 3) == blocks
    assert alloc.free_blocks == 3
    for b in blocks:
        alloc.free(b)
    # the same second chunk after a different first chunk must not match: hashes chain
    other = [50, 51, 52, 53] + ids[4:8]
    assert alloc.match(other, 4, 2) == []
    # a prompt that diverges in its second block shares only the first block
    diverging = ids[:4] + [90, 91, 92, 93]
    assert alloc.match(diverging, 4, 2) == blocks[:1]


def test_least_recently_freed_block_is_evicted_first():
    alloc = CachingAllocator(2)
    ids = list(range(8))
    blocks = [alloc.allocate(), alloc.allocate()]
    with pytest.raises(OutOfBlocks):
        alloc.allocate()
    alloc.register(ids, 4, blocks)
    for b in blocks:
        alloc.free(b)
    assert alloc.allocate() == blocks[0]  # the oldest cached block is taken
    assert alloc.match(ids, 4, 2) == []  # its chunk is gone, so nothing matches any more


def test_prefix_server_matches_baseline_reuses_blocks_and_returns_all():
    model = make(TINY)
    torch.manual_seed(1)
    prefix = torch.randint(2, TINY.vocab_size, (8,)).tolist()
    suffixes = (3, 5, 2, 6, 4, 1)
    prompts = [prefix + torch.randint(2, TINY.vocab_size, (k,)).tolist() for k in suffixes]
    base = BaselineServer(model, 1, "cpu")
    ref = [base.submit(Request(ids=p, max_new=8)).result()["ids"] for p in prompts]
    base.close()
    server = PrefixServer(model, 1, "cpu", num_blocks=16, block_size=4, max_seqs=3, max_len=32)
    futures = [server.submit(Request(ids=p, max_new=8)) for p in prompts]
    got = [f.result()["ids"] for f in futures]
    server.close()
    assert got == ref
    assert server.cached_tokens > 0
    assert server.alloc.free_blocks == 16
