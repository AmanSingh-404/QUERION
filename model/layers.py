import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normed.to(x.dtype)


def precompute_rope(head_dim, max_seq_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    """x: (batch, heads, seq, head_dim); cos, sin: (seq, head_dim)."""
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


class Attention(nn.Module):
    """Causal grouped-query attention: n_heads query heads share n_kv_heads key/value heads."""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.rep = self.n_heads // self.n_kv
        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wo = nn.Linear(self.n_heads * self.hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None, pos=0):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        if cos is not None:
            q = apply_rope(q, cos[pos : pos + t], sin[pos : pos + t])
            k = apply_rope(k, cos[pos : pos + t], sin[pos : pos + t])
        if cache is not None:
            ck, cv = cache
            ck[:, :, pos : pos + t] = k
            cv[:, :, pos : pos + t] = v
            if t == 1 and pos > 0:
                return self._decode(q, ck[:, :, : pos + 1], cv[:, :, : pos + 1])
            if pos != 0:
                raise ValueError("after the prefill, a cached pass takes one token at a time")
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(b, t, -1))

    def _decode(self, q, k, v):
        """Single-token step: group the rep query heads of each KV head, no K/V repeat."""
        b = q.shape[0]
        q = q.reshape(b, self.n_kv, self.rep, self.hd)
        y = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype))
        return self.wo(y.reshape(b, 1, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, pos=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, pos)
        return x + self.mlp(self.mlp_norm(x))
