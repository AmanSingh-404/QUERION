import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from infer.engine import KVCache
from model.layers import apply_rope


class SlotPool:
    """Contiguous KV storage with one fixed-size slot per running sequence."""

    def __init__(self, cfg, slots, max_len, device, dtype):
        shape = (slots, cfg.n_kv_heads, max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]


@torch.no_grad()
def decode_step(model, pool, tokens, slots, positions):
    """One new token for each of B sequences, each at its own position.

    tokens: (B, 1) ids to feed. slots: (B,) pool slot of each sequence.
    positions: (B,) index where each token goes. Returns logits of shape (B, vocab)."""
    b = tokens.shape[0]
    x = model.tok_emb(tokens)
    if model.pos_emb is not None:
        x = x + model.pos_emb(positions)[:, None, :]
        cos = sin = None
    else:
        cos = model.cos[positions][:, None, None, :]
        sin = model.sin[positions][:, None, None, :]
    length = int(positions.max()) + 1
    valid = torch.arange(length, device=tokens.device)[None, :] <= positions[:, None]
    mask = valid[:, None, None, :]
    for i, block in enumerate(model.blocks):
        attn = block.attn
        h = block.attn_norm(x)
        q = attn.wq(h).view(b, 1, attn.n_heads, attn.hd).transpose(1, 2)
        k = attn.wk(h).view(b, 1, attn.n_kv, attn.hd).transpose(1, 2)
        v = attn.wv(h).view(b, 1, attn.n_kv, attn.hd).transpose(1, 2)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        pool.k[i][slots, :, positions] = k[:, :, 0]
        pool.v[i][slots, :, positions] = v[:, :, 0]
        keys = pool.k[i][slots, :, :length]
        values = pool.v[i][slots, :, :length]
        q = q.reshape(b, attn.n_kv, attn.rep, attn.hd)
        y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        x = x + attn.wo(y.reshape(b, 1, -1))
        x = x + block.mlp(block.mlp_norm(x))
    return model.lm_head(model.norm(x))[:, 0]


@dataclass
class Sequence:
    future: Future
    slot: int
    pos: int
    total: int
    max_new: int
    out: list
    first_at: float
    next_token: int


class BatchedServer:
    """Continuous batching: a request takes a free slot, decodes together with the other
    running requests one token per step, and gives its slot back when it finishes."""

    def __init__(self, model, eos_id, device, slots=16, max_len=None, max_admit=4):
        self.model, self.eos_id, self.device = model.eval(), eos_id, device
        self.max_len = max_len or model.cfg.max_seq_len
        dtype = next(model.parameters()).dtype
        self.pool = SlotPool(model.cfg, slots, self.max_len, device, dtype)
        self.free = list(range(slots))
        self.active = []
        self.max_admit = max_admit
        self.closing = False
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

    def _loop(self):
        while True:
            if not self.active and not self.closing:
                item = self.queue.get()  # idle: wait for work
                if item is None:
                    return
                self._admit(item)
            admitted = 0
            while self.free and admitted < self.max_admit:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self.closing = True
                    break
                self._admit(item)
                admitted += 1
            if self.active:
                self._step()
            elif self.closing:
                return

    def _admit(self, item):
        request, future = item
        try:
            ids, n = request.ids, len(request.ids)
            if n + 1 > self.max_len:
                raise ValueError(f"prompt of {n} tokens does not fit in {self.max_len}")
            total = min(n + request.max_new, self.max_len)
            one = KVCache(self.model.cfg, 1, n, self.device, self.pool.k[0].dtype)
            with torch.no_grad():
                logits, _ = self.model(torch.tensor([ids], device=self.device), cache=one, pos=0)
            token = int(logits[0, -1].argmax())
            first = time.perf_counter()
            if token == self.eos_id or request.max_new <= 1 or n >= total:
                future.set_result({"tokens": 1, "first_token_at": first, "ids": [token]})
                return
            slot = self.free.pop()
            for layer in range(self.model.cfg.n_layers):
                self.pool.k[layer][slot, :, :n] = one.k[layer][0]
                self.pool.v[layer][slot, :, :n] = one.v[layer][0]
            seq = Sequence(future, slot, n, total, request.max_new, [token], first, token)
            self.active.append(seq)
        except Exception as e:  # report the failure to the caller, keep the server alive
            future.set_exception(e)

    def _step(self):
        seqs = self.active
        tokens = torch.tensor([[s.next_token] for s in seqs], device=self.device)
        slots = torch.tensor([s.slot for s in seqs], device=self.device)
        positions = torch.tensor([s.pos for s in seqs], device=self.device)
        try:
            logits = decode_step(self.model, self.pool, tokens, slots, positions)
            chosen = logits.argmax(-1).tolist()
        except Exception as e:
            for s in seqs:
                s.future.set_exception(e)
                self.free.append(s.slot)
            self.active = []
            return
        still = []
        for s, token in zip(seqs, chosen, strict=True):
            s.pos += 1
            s.out.append(token)
            if token == self.eos_id or s.pos >= s.total or len(s.out) >= s.max_new:
                result = {"tokens": len(s.out), "first_token_at": s.first_at, "ids": s.out}
                s.future.set_result(result)
                self.free.append(s.slot)
            else:
                s.next_token = token
                still.append(s)
        self.active = still
