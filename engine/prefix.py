import time
from collections import OrderedDict

import torch
import torch.nn.functional as F

from engine.paged import BlockAllocator, OutOfBlocks, PagedSequence, PagedServer
from model.layers import apply_rope


class CachingAllocator(BlockAllocator):
    """Block allocator that remembers the contents of full prompt blocks.

    A block nobody uses stays cached, counts as free, and is evicted oldest first."""

    def __init__(self, num_blocks):
        super().__init__(num_blocks)
        self.cached = {}  # prefix hash -> block id
        self.key = {}  # block id -> (hash, parent hash, tokens)
        self.lru = OrderedDict()  # cached blocks nobody uses, oldest first

    @property
    def free_blocks(self):
        return len(self.free_list) + len(self.lru)

    def allocate(self):
        if self.free_list:
            block = self.free_list.pop()
        elif self.lru:
            block, _ = self.lru.popitem(last=False)
            h, _, _ = self.key.pop(block)
            del self.cached[h]
        else:
            raise OutOfBlocks("no free KV blocks")
        self.refs[block] = 1
        return block

    def free(self, block):
        if self.refs[block] <= 0:
            raise ValueError(f"block {block} is already free")
        self.refs[block] -= 1
        if self.refs[block] == 0:
            if block in self.key:
                self.lru[block] = None
            else:
                self.free_list.append(block)

    def match(self, ids, bs, limit):
        """Take a reference to the cached blocks that hold the start of `ids`."""
        found, parent = [], 0
        for i in range(limit):
            chunk = tuple(ids[i * bs : (i + 1) * bs])
            h = hash((parent, chunk))
            block = self.cached.get(h)
            if block is None or self.key[block][1:] != (parent, chunk):
                break
            if self.refs[block] == 0:
                del self.lru[block]
            self.refs[block] += 1
            found.append(block)
            parent = h
        return found

    def register(self, ids, bs, blocks):
        """Make the full blocks of a computed prompt available to later requests."""
        parent = 0
        for i in range(len(ids) // bs):
            chunk = tuple(ids[i * bs : (i + 1) * bs])
            h = hash((parent, chunk))
            if h not in self.cached and blocks[i] not in self.key:
                self.cached[h] = blocks[i]
                self.key[blocks[i]] = (h, parent, chunk)
            parent = h


@torch.no_grad()
def prefill_paged(model, pool, ids, blocks, start):
    """Process ids[start:], whose predecessors are already in the pool, and return the logits
    of the last prompt token, shape (1, vocab). `blocks` holds every block of the prompt."""
    n, bs = len(ids), pool.block_size
    t = n - start
    device = pool.k[0].device
    tokens = torch.tensor([ids[start:]], device=device)
    positions = torch.arange(start, n, device=device)
    x = model.tok_emb(tokens)
    if model.pos_emb is not None:
        x = x + model.pos_emb(positions)[None]
        cos = sin = None
    else:
        cos, sin = model.cos[positions], model.sin[positions]
    table = torch.tensor(blocks, device=device)
    write_blocks, write_offsets = table[positions // bs], positions % bs
    nblk = (n - 1) // bs + 1
    read, length = table[:nblk], nblk * bs
    mask = (torch.arange(length, device=device)[None, :] <= positions[:, None])[None, None]
    for i, block in enumerate(model.blocks):
        attn = block.attn
        h = block.attn_norm(x)
        q = attn.wq(h).view(1, t, attn.n_heads, attn.hd).transpose(1, 2)
        k = attn.wk(h).view(1, t, attn.n_kv, attn.hd).transpose(1, 2)
        v = attn.wv(h).view(1, t, attn.n_kv, attn.hd).transpose(1, 2)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        pool.k[i][write_blocks, :, write_offsets] = k[0].transpose(0, 1)
        pool.v[i][write_blocks, :, write_offsets] = v[0].transpose(0, 1)
        keys = pool.k[i][read].permute(1, 0, 2, 3).reshape(1, attn.n_kv, length, attn.hd)
        values = pool.v[i][read].permute(1, 0, 2, 3).reshape(1, attn.n_kv, length, attn.hd)
        if attn.rep > 1:
            keys = keys.repeat_interleave(attn.rep, dim=1)
            values = values.repeat_interleave(attn.rep, dim=1)
        y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        x = x + attn.wo(y.transpose(1, 2).reshape(1, t, -1))
        x = x + block.mlp(block.mlp_norm(x))
    return model.lm_head(model.norm(x[:, -1]))


class PrefixServer(PagedServer):
    """Paged continuous batching that shares prompt blocks between requests with a common prefix."""

    allocator_class = CachingAllocator

    def __init__(self, *args, **kwargs):
        self.prompt_tokens = 0
        self.cached_tokens = 0
        super().__init__(*args, **kwargs)

    @property
    def hit_rate(self):
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def _admit(self, item):
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
        matched = self.alloc.match(ids, bs, (n - 1) // bs)
        if self.alloc.free_blocks < need - len(matched) + len(self.active):
            for b in matched:
                self.alloc.free(b)
            return False
        blocks = list(matched)
        try:
            total = min(n + request.max_new, self.max_len)
            for _ in range(need - len(matched)):
                blocks.append(self.alloc.allocate())
            logits = prefill_paged(self.model, self.pool, ids, blocks, len(matched) * bs)
            token = int(logits[0].argmax())
            first = time.perf_counter()
            self.alloc.register(ids, bs, blocks)
            self.prompt_tokens += n
            self.cached_tokens += len(matched) * bs
            if token == self.eos_id or request.max_new <= 1 or n >= total:
                for b in blocks:
                    self.alloc.free(b)
                future.set_result({"tokens": 1, "first_token_at": first, "ids": [token]})
                return True
            seq = PagedSequence(future, n, total, request.max_new, [token], first, token, blocks)
            self.active.append(seq)
        except Exception as e:  # report the failure to the caller, keep the server alive
            for b in blocks:
                self.alloc.free(b)
            future.set_exception(e)
        return True
