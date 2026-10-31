import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from model.config import ModelConfig
from model.layers import Block, RMSNorm, precompute_rope


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpointing = False
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.pos_emb = None
        if cfg.pos_emb == "learned":
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            cos = sin = None
        else:
            cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init_weights)
        # scale the output projection of every residual branch by 1/sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("attn.wo.weight", "mlp.w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, cache=None, pos=0):
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(pos, pos + idx.size(1), device=idx.device))
        for i, block in enumerate(self.blocks):
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = block(x, self.cos, self.sin, cache[i] if cache is not None else None, pos)
        if cache is not None and targets is None:
            x = x[:, -1:]
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-100
            )
        return logits, loss
