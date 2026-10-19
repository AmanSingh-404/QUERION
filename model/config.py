from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 24000
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden: int = 2304
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


QUERION_100M = ModelConfig()

TINY = ModelConfig(
    vocab_size=100,
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    ffn_hidden=128,
    max_seq_len=32,
)
