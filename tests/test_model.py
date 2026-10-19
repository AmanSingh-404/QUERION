import math

import pytest
import torch
import torch.nn.functional as F

from model.config import QUERION_100M, TINY, ModelConfig
from model.layers import Attention, RMSNorm, apply_rope, precompute_rope
from model.transformer import Transformer


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=12)
    with pytest.raises(ValueError):
        ModelConfig(n_heads=12, n_kv_heads=5)
    with pytest.raises(ValueError):
        ModelConfig(d_model=36, n_heads=12, n_kv_heads=4)


def test_param_count_100m():
    assert Transformer(QUERION_100M).num_params() == 101_026_560


def test_rmsnorm_output_has_unit_rms():
    y = RMSNorm(64)(torch.randn(4, 10, 64) * 5)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_vector_norm():
    cos, sin = precompute_rope(16, 32)
    x = torch.randn(2, 3, 32, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_rope_scores_depend_only_on_relative_position():
    cos, sin = precompute_rope(16, 64)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m, n):
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum()

    assert torch.allclose(score(10, 3), score(20, 13), atol=1e-4)
    assert not torch.allclose(score(10, 3), score(10, 4), atol=1e-4)


def test_gqa_matches_mha_with_shared_kv_weights():
    kw = {"vocab_size": 10, "n_layers": 1, "d_model": 64, "n_heads": 4, "ffn_hidden": 32}
    gqa = Attention(ModelConfig(n_kv_heads=2, max_seq_len=16, **kw))
    mha = Attention(ModelConfig(n_kv_heads=4, max_seq_len=16, **kw))
    mha.wq.weight.data.copy_(gqa.wq.weight.data)
    mha.wo.weight.data.copy_(gqa.wo.weight.data)
    hd = 16
    for name in ("wk", "wv"):
        w = getattr(gqa, name).weight.data.view(2, hd, 64)
        getattr(mha, name).weight.data.copy_(w.repeat_interleave(2, dim=0).reshape(4 * hd, 64))
    x = torch.randn(2, 8, 64)
    cos, sin = precompute_rope(hd, 16)
    assert torch.allclose(gqa(x, cos, sin), mha(x, cos, sin), atol=1e-5)


def test_causal_masking():
    model = Transformer(TINY).eval()
    a = torch.randint(0, 100, (1, 16))
    b = a.clone()
    b[0, 10] = (b[0, 10] + 1) % 100
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-5)
    assert not torch.allclose(la[:, 10:], lb[:, 10:], atol=1e-5)


def test_forward_shapes_and_initial_loss():
    model = Transformer(TINY)
    x, y = torch.randint(0, 100, (3, 16)), torch.randint(0, 100, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, 100)
    assert abs(loss.item() - math.log(100)) < 0.3


def test_embeddings_are_tied():
    model = Transformer(TINY)
    assert model.lm_head.weight is model.tok_emb.weight


def test_residual_projections_use_scaled_init():
    model = Transformer(TINY)
    block = model.blocks[0]
    assert block.attn.wo.weight.std().item() < 0.7 * block.attn.wq.weight.std().item()


def test_loss_ignores_masked_targets():
    model = Transformer(TINY).eval()
    x, y = torch.randint(0, 100, (2, 16)), torch.randint(0, 100, (2, 16))
    masked = y.clone()
    masked[:, :8] = -100
    logits, loss = model(x, masked)
    expected = F.cross_entropy(logits[:, 8:].reshape(-1, 100), y[:, 8:].reshape(-1))
    assert torch.allclose(loss, expected, atol=1e-5)


def test_all_parameters_receive_gradients():
    model = Transformer(TINY)
    x, y = torch.randint(0, 100, (2, 16)), torch.randint(0, 100, (2, 16))
    model(x, y)[1].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_overfits_a_tiny_batch():
    torch.manual_seed(0)
    model = Transformer(TINY)
    x, y = torch.randint(0, 100, (4, 16)), torch.randint(0, 100, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
