import pytest
import torch
from torch import nn

from infer.quantize import quantize_int8
from model.config import TINY
from model.transformer import Transformer

ENGINES = [e for e in torch.backends.quantized.supported_engines if e != "none"]


@pytest.mark.skipif(not ENGINES, reason="no int8 backend in this PyTorch build")
def test_int8_model_replaces_every_linear_layer_and_stays_close():
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    with torch.no_grad():
        model.tok_emb.weight.mul_(20)
    quantized = quantize_int8(model)
    before = sum(isinstance(m, nn.Linear) for m in model.modules())
    after = sum(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in quantized.modules())
    assert before == after > 0
    ids = torch.randint(2, TINY.vocab_size, (1, 12))
    with torch.no_grad():
        ref, got = model(ids)[0].flatten(), quantized(ids)[0].flatten()
    assert torch.nn.functional.cosine_similarity(ref, got, dim=0) > 0.95
