import pytest
import torch

from model.config import TINY
from model.transformer import Transformer

transformers = pytest.importorskip("transformers")


def build_reference(cfg):
    ref_cfg = transformers.LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.ffn_hidden,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
    )
    return transformers.LlamaForCausalLM(ref_cfg).eval()


def copy_weights(ours, ref):
    sd = {"model.embed_tokens.weight": ours.tok_emb.weight, "model.norm.weight": ours.norm.weight}
    names = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn.wq.weight": "self_attn.q_proj.weight",
        "attn.wk.weight": "self_attn.k_proj.weight",
        "attn.wv.weight": "self_attn.v_proj.weight",
        "attn.wo.weight": "self_attn.o_proj.weight",
        "mlp_norm.weight": "post_attention_layernorm.weight",
        "mlp.w1.weight": "mlp.gate_proj.weight",
        "mlp.w3.weight": "mlp.up_proj.weight",
        "mlp.w2.weight": "mlp.down_proj.weight",
    }
    for i, block in enumerate(ours.blocks):
        params = dict(block.named_parameters())
        for ours_name, ref_name in names.items():
            sd[f"model.layers.{i}.{ref_name}"] = params[ours_name]
    ref.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=False)


def test_logits_match_reference_llama():
    torch.manual_seed(0)
    ours = Transformer(TINY).eval()
    ref = build_reference(TINY)
    copy_weights(ours, ref)
    idx = torch.randint(0, TINY.vocab_size, (3, 24))
    with torch.no_grad():
        mine, _ = ours(idx)
        theirs = ref(idx).logits
    assert (mine - theirs).abs().max().item() < 1e-4
