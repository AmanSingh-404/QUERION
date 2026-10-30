from dataclasses import dataclass


@dataclass
class SFTConfig:
    sft_dir: str = "C:/querion-data/sft"
    init_ckpt: str = "C:/querion-runs/main/latest.pt"
    out_dir: str = "C:/querion-runs/sft-aug"
    model: str = "100m"
    epochs: int = 2
    spider_repeat: int = 1
    aug_copies: int = 3
    aug_other: float = 0.3
    micro_tokens: int = 2048
    accum: int = 16
    total_steps: int = 0
    warmup_steps: int = 30
    lr: float = 1e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337
    eval_every: int = 50
    ckpt_every: int = 100
    stop_at_step: int = 0
