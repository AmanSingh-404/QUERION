import torch
from torch import nn


def quantize_int8(model):
    """Dynamic int8 quantization of every linear layer (CPU only), with per-channel scales."""
    config = torch.ao.quantization.per_channel_dynamic_qconfig
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear: config}, dtype=torch.qint8)
