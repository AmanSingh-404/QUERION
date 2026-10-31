import torch


class KVCache:
    """Per-layer key/value buffers, filled in place as tokens are processed."""

    def __init__(self, cfg, batch, max_len, device, dtype):
        shape = (batch, cfg.n_kv_heads, max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]

    def __getitem__(self, layer):
        return self.k[layer], self.v[layer]


def pick(row, constraint, top_k=64):
    """Highest-scoring token the constraint accepts (plain argmax without a constraint)."""
    if constraint is None:
        return int(row.argmax())
    for cid in torch.topk(row, min(top_k, row.numel())).indices.tolist():
        if constraint.accepts(cid):
            return cid
    for cid in torch.argsort(row, descending=True).tolist():
        if constraint.accepts(cid):
            return cid
    return constraint.eos_id


@torch.no_grad()
def generate_cached(model, ids, max_new, eos_id, device, constraint=None):
    """Greedy decoding with a KV cache. Returns the prompt ids plus the new ids."""
    model.eval()
    cfg = model.cfg
    total = min(len(ids) + max_new, cfg.max_seq_len)
    if len(ids) > total:
        raise ValueError("prompt does not fit in the context window")
    if constraint is not None:
        constraint.reset()
    dtype = next(model.parameters()).dtype
    cache = KVCache(cfg, 1, total, device, dtype)
    logits, _ = model(torch.tensor([ids], device=device), cache=cache, pos=0)
    out, pos = list(ids), len(ids)
    for _ in range(max_new):
        nxt = pick(logits[0, -1], constraint)
        out.append(nxt)
        if nxt == eos_id or pos >= total:
            break
        if constraint is not None:
            constraint.advance(nxt)
        logits, _ = model(torch.tensor([[nxt]], device=device), cache=cache, pos=pos)
        pos += 1
    return out


@torch.no_grad()
def generate_batch(
    model, ids, k, max_new, eos_id, device, constraints=None, temperature=0.7, seed=0, top_k=64
):
    """k continuations of one prompt in one batch. Row 0 is greedy, the others sample.

    Returns k lists of new token ids (without the end token). eos_id must be a real token id."""
    model.eval()
    cfg = model.cfg
    total = min(len(ids) + max_new, cfg.max_seq_len)
    if len(ids) > total:
        raise ValueError("prompt does not fit in the context window")
    dtype = next(model.parameters()).dtype
    cache = KVCache(cfg, k, total, device, dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for c in constraints or []:
        c.reset()
    logits, _ = model(torch.tensor([ids] * k, device=device), cache=cache, pos=0)
    outs = [[] for _ in range(k)]
    done = [False] * k
    pos = len(ids)
    for _ in range(max_new):
        last = logits[:, -1].float()
        width = min(top_k, last.shape[-1])
        vals, idx = torch.topk(last, width, dim=-1)
        weights = torch.softmax(vals / temperature, dim=-1) + 1e-9
        order = torch.multinomial(weights, width, generator=gen)
        order[0] = torch.arange(width, device=device)
        tries = torch.gather(idx, 1, order).tolist()
        nxt = []
        for r in range(k):
            if done[r]:
                nxt.append(eos_id)
                continue
            con = constraints[r] if constraints else None
            cid = next((c for c in tries[r] if con is None or con.accepts(c)), None)
            if cid is None:
                cid = pick(last[r], con)
            nxt.append(cid)
            if cid == eos_id:
                done[r] = True
            else:
                outs[r].append(cid)
                if con is not None:
                    con.advance(cid)
        if all(done) or pos >= total:
            break
        logits, _ = model(torch.tensor([[t] for t in nxt], device=device), cache=cache, pos=pos)
        pos += 1
    return outs
