import torch


class KVCache:
    """Per-layer key/value buffers, filled in place as tokens are processed."""

    def __init__(self, cfg, batch, max_len, device, dtype):
        shape = (batch, cfg.n_kv_heads, max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]

    def __getitem__(self, layer):
        return self.k[layer], self.v[layer]


@torch.no_grad()
def generate_cached(model, ids, max_new, eos_id, device):
    """Greedy decoding with a KV cache. Returns the prompt ids plus the new ids."""
    model.eval()
    cfg = model.cfg
    total = min(len(ids) + max_new, cfg.max_seq_len)
    if len(ids) > total:
        raise ValueError("prompt does not fit in the context window")
    dtype = next(model.parameters()).dtype
    cache = KVCache(cfg, 1, total, device, dtype)
    logits, _ = model(torch.tensor([ids], device=device), cache=cache, pos=0)
    out, pos = list(ids), len(ids)
    for _ in range(max_new):
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if nxt == eos_id or pos >= total:
            break
        logits, _ = model(torch.tensor([[nxt]], device=device), cache=cache, pos=pos)
        pos += 1
    return out
