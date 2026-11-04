from dataclasses import replace

import pytest
import torch

from engine.baseline import BaselineServer
from engine.bench import Request
from engine.paged import (
    BlockAllocator,
    BlockPool,
    OutOfBlocks,
    PagedServer,
    decode_step_paged,
    write_prompt,
)
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


def prefill_into_blocks(model, pool, blocks, ids):
    cache = KVCache(model.cfg, 1, len(ids), "cpu", torch.float32)
    with torch.no_grad():
        model(torch.tensor([ids]), cache=cache, pos=0)
    write_prompt(pool, cache, len(ids), blocks)


def test_allocator_hands_out_every_block_once_and_takes_them_back():
    alloc = BlockAllocator(4)
    blocks = [alloc.allocate() for _ in range(4)]
    assert sorted(blocks) == [0, 1, 2, 3] and alloc.free_blocks == 0
    with pytest.raises(OutOfBlocks):
        alloc.allocate()
    for b in blocks:
        alloc.free(b)
    assert alloc.free_blocks == 4
    with pytest.raises(ValueError):
        alloc.free(blocks[0])


@pytest.mark.parametrize("cfg", CONFIGS)
def test_paged_decode_matches_full_forward_with_scattered_blocks(cfg):
    model = make(cfg)
    pool = BlockPool(cfg, 12, 4, "cpu", torch.float32)
    a = torch.randint(2, cfg.vocab_size, (5,)).tolist()
    b = torch.randint(2, cfg.vocab_size, (9,)).tolist()
    with torch.no_grad():
        prefill_into_blocks(model, pool, [7, 2], a)
        prefill_into_blocks(model, pool, [10, 0, 5], b)
        # the padding entry of the first table points at block 0, which belongs to the second
        tables = torch.tensor([[7, 2, 0], [10, 0, 5]])
        tokens = torch.tensor([[7], [11]])
        positions = torch.tensor([5, 9])
        out = decode_step_paged(model, pool, tokens, tables, positions)
        for row, ids, tok in ((0, a, 7), (1, b, 11)):
            ref = model(torch.tensor([ids + [tok]]))[0][0, -1]
            assert torch.allclose(out[row], ref, atol=1e-4)


def test_paged_server_matches_baseline_and_returns_every_block():
    model = make(TINY)
    torch.manual_seed(1)
    lengths = (3, 8, 5, 4, 7, 6)
    prompts = [torch.randint(2, TINY.vocab_size, (n,)).tolist() for n in lengths]
    base = BaselineServer(model, 1, "cpu")
    ref = [base.submit(Request(ids=p, max_new=8)).result()["ids"] for p in prompts]
    base.close()
    server = PagedServer(model, 1, "cpu", num_blocks=14, block_size=4, max_seqs=3, max_len=32)
    futures = [server.submit(Request(ids=p, max_new=8)) for p in prompts]
    got = [f.result()["ids"] for f in futures]
    server.close()
    assert got == ref
    assert server.alloc.free_blocks == 14
    assert 0 < server.peak_blocks <= 14


def test_request_larger_than_the_pool_fails_without_stopping_the_server():
    model = make(TINY)
    server = PagedServer(model, 1, "cpu", num_blocks=4, block_size=4, max_len=32)
    bad = server.submit(Request(ids=list(range(2, 22)), max_new=4))
    good = server.submit(Request(ids=[2, 3, 4], max_new=4))
    with pytest.raises(OutOfBlocks):
        bad.result()
    assert good.result()["tokens"] >= 1
    server.close()
    assert server.alloc.free_blocks == 4
