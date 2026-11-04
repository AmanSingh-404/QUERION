import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from infer.engine import KVCache
from model.layers import apply_rope


class OutOfBlocks(Exception):
    """The KV block pool has no room for this request."""


class BlockAllocator:
    """Hands out block ids from a free list and takes them back."""

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.free_list = list(range(num_blocks - 1, -1, -1))
        self.refs = [0] * num_blocks

    @property
    def free_blocks(self):
        return len(self.free_list)

    def allocate(self):
        if not self.free_list:
            raise OutOfBlocks("no free KV blocks")
        block = self.free_list.pop()
        self.refs[block] = 1
        return block

    def free(self, block):
        if self.refs[block] <= 0:
            raise ValueError(f"block {block} is already free")
        self.refs[block] -= 1
        if self.refs[block] == 0:
            self.free_list.append(block)


class BlockPool:
    """KV storage in fixed-size blocks that sequences borrow as they grow."""

    def __init__(self, cfg, num_blocks, block_size, device, dtype):
        shape = (num_blocks, cfg.n_kv_heads, block_size, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.block_size = block_size
        per_layer = cfg.n_kv_heads * block_size * cfg.head_dim * self.k[0].element_size()
        self.block_bytes = 2 * cfg.n_layers * per_layer


@torch.no_grad()
def write_prompt(pool, cache, n, blocks):
    """Copy the first n cached positions of a batch-1 KVCache into the given blocks."""
    bs, nb = pool.block_size, len(blocks)
    idx = torch.tensor(blocks, device=pool.k[0].device)
    for layer in range(len(pool.k)):
        for src, dst in ((cache.k[layer], pool.k[layer]), (cache.v[layer], pool.v[layer])):
            n_kv, hd = src.shape[1], src.shape[3]
            padded = torch.zeros(n_kv, nb * bs, hd, device=src.device, dtype=src.dtype)
            padded[:, :n] = src[0, :, :n]
            dst[idx] = padded.view(n_kv, nb, bs, hd).permute(1, 0, 2, 3)


@torch.no_grad()
def decode_step_paged(model, pool, tokens, tables, positions):
    """One new token for each of B sequences, reading K and V through block tables.

    tokens: (B, 1). tables: (B, W) block ids per sequence, padded with any valid id.
    positions: (B,) index where each token goes. Returns logits of shape (B, vocab)."""
    b, bs = tokens.shape[0], pool.block_size
    x = model.tok_emb(tokens)
    if model.pos_emb is not None:
        x = x + model.pos_emb(positions)[:, None, :]
        cos = sin = None
    else:
        cos = model.cos[positions][:, None, None, :]
        sin = model.sin[positions][:, None, None, :]
    rows = torch.arange(b, device=tokens.device)
    write_blocks = tables[rows, positions // bs]
    write_offsets = positions % bs
    nblk = int(positions.max()) // bs + 1
    read_tables = tables[:, :nblk]
    length = nblk * bs
    valid = torch.arange(length, device=tokens.device)[None, :] <= positions[:, None]
    mask = valid[:, None, None, :]

    def gather(t):
        return t[read_tables].permute(0, 2, 1, 3, 4).reshape(b, t.shape[1], length, t.shape[3])

    for i, block in enumerate(model.blocks):
        attn = block.attn
        h = block.attn_norm(x)
        q = attn.wq(h).view(b, 1, attn.n_heads, attn.hd).transpose(1, 2)
        k = attn.wk(h).view(b, 1, attn.n_kv, attn.hd).transpose(1, 2)
        v = attn.wv(h).view(b, 1, attn.n_kv, attn.hd).transpose(1, 2)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        pool.k[i][write_blocks, :, write_offsets] = k[:, :, 0]
        pool.v[i][write_blocks, :, write_offsets] = v[:, :, 0]
        q = q.reshape(b, attn.n_kv, attn.rep, attn.hd)
        y = F.scaled_dot_product_attention(q, gather(pool.k[i]), gather(pool.v[i]), attn_mask=mask)
        x = x + attn.wo(y.reshape(b, 1, -1))
        x = x + block.mlp(block.mlp_norm(x))
    return model.lm_head(model.norm(x))[:, 0]


@dataclass
class PagedSequence:
    future: Future
    pos: int
    total: int
    max_new: int
    out: list
    first_at: float
    next_token: int
    blocks: list


class PagedServer:
    """Continuous batching over a paged KV cache: blocks are borrowed as sequences grow."""

    def __init__(
        self,
        model,
        eos_id,
        device,
        num_blocks=1024,
        block_size=16,
        max_seqs=16,
        max_len=None,
        max_admit=4,
    ):
        self.model, self.eos_id, self.device = model.eval(), eos_id, device
        self.max_len = max_len or model.cfg.max_seq_len
        dtype = next(model.parameters()).dtype
        self.pool = BlockPool(model.cfg, num_blocks, block_size, device, dtype)
        self.alloc = BlockAllocator(num_blocks)
        self.max_seqs, self.max_admit = max_seqs, max_admit
        self.active, self.waiting = [], deque()
        self.closing = False
        self.peak_blocks = 0
        self.peak_cap = self.peak_used = 0
        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, request):
        future = Future()
        self.queue.put((request, future))
        return future

    def close(self):
        self.queue.put(None)
        self.worker.join()

    @property
    def kv_reserved_gb(self):
        return self.alloc.num_blocks * self.pool.block_bytes / 1e9

    @property
    def kv_peak_gb(self):
        return self.peak_blocks * self.pool.block_bytes / 1e9

    @property
    def fragmentation(self):
        """Share of allocated block space not holding tokens, at the moment of peak use."""
        return 1 - self.peak_used / self.peak_cap if self.peak_cap else 0.0

    def _loop(self):
        while True:
            if not self.active and not self.waiting and not self.closing:
                item = self.queue.get()  # idle: wait for work
                if item is None:
                    return
                self.waiting.append(item)
            while True:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self.closing = True
                else:
                    self.waiting.append(item)
            admitted = 0
            while self.waiting and len(self.active) < self.max_seqs and admitted < self.max_admit:
                if not self._admit(self.waiting[0]):
                    break
                self.waiting.popleft()
                admitted += 1
            if self.active:
                self._step()
            elif self.closing and not self.waiting:
                return

    def _admit(self, item):
        """Start a request. Returns False if it has to wait for blocks to be freed."""
        request, future = item
        ids, n = request.ids, len(request.ids)
        bs = self.pool.block_size
        need = -(-n // bs)
        if n + 1 > self.max_len:
            future.set_exception(ValueError(f"prompt of {n} tokens does not fit in {self.max_len}"))
            return True
        if need > self.alloc.num_blocks:
            future.set_exception(OutOfBlocks(f"prompt needs {need} blocks, pool has fewer"))
            return True
        if self.alloc.free_blocks < need + len(self.active):
            return False
        blocks = []
        try:
            total = min(n + request.max_new, self.max_len)
            one = KVCache(self.model.cfg, 1, n, self.device, self.pool.k[0].dtype)
            with torch.no_grad():
                logits, _ = self.model(torch.tensor([ids], device=self.device), cache=one, pos=0)
            token = int(logits[0, -1].argmax())
            first = time.perf_counter()
            if token == self.eos_id or request.max_new <= 1 or n >= total:
                future.set_result({"tokens": 1, "first_token_at": first, "ids": [token]})
                return True
            blocks = [self.alloc.allocate() for _ in range(need)]
            write_prompt(self.pool, one, n, blocks)
            seq = PagedSequence(future, n, total, request.max_new, [token], first, token, blocks)
            self.active.append(seq)
        except Exception as e:  # report the failure to the caller, keep the server alive
            for b in blocks:
                self.alloc.free(b)
            future.set_exception(e)
        return True

    def _release(self, seq):
        for b in seq.blocks:
            self.alloc.free(b)
        seq.blocks = []

    def _step(self):
        bs = self.pool.block_size
        ready = []
        for s in self.active:
            if s.pos // bs == len(s.blocks):  # the next token starts a new block
                try:
                    s.blocks.append(self.alloc.allocate())
                except OutOfBlocks as e:
                    s.future.set_exception(e)
                    self._release(s)
                    continue
            ready.append(s)
        self.active = ready
        if not ready:
            return
        in_use = self.alloc.num_blocks - self.alloc.free_blocks
        self.peak_blocks = max(self.peak_blocks, in_use)
        cap = sum(len(s.blocks) for s in ready) * bs
        if cap > self.peak_cap:
            self.peak_cap, self.peak_used = cap, sum(s.pos for s in ready)
        width = max(len(s.blocks) for s in ready)
        tables = torch.tensor(
            [s.blocks + [0] * (width - len(s.blocks)) for s in ready], device=self.device
        )
        tokens = torch.tensor([[s.next_token] for s in ready], device=self.device)
        positions = torch.tensor([s.pos for s in ready], device=self.device)
        try:
            logits = decode_step_paged(self.model, self.pool, tokens, tables, positions)
            chosen = logits.argmax(-1).tolist()
        except Exception as e:
            for s in ready:
                s.future.set_exception(e)
                self._release(s)
            self.active = []
            return
        still = []
        for s, token in zip(ready, chosen, strict=True):
            s.pos += 1
            s.out.append(token)
            if token == self.eos_id or s.pos >= s.total or len(s.out) >= s.max_new:
                result = {"tokens": len(s.out), "first_token_at": s.first_at, "ids": s.out}
                s.future.set_result(result)
                self._release(s)
            else:
                s.next_token = token
                still.append(s)
        self.active = still
