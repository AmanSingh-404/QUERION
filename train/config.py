from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_root: str = "C:/querion-data/shards"
    out_dir: str = "C:/querion-runs/main"
    run_name: str = "main"
    model: str = "100m"
    seq_len: int = 2048
    micro_batch: int = 1
    accum: int = 256
    total_steps: int = 3815
    warmup_steps: int = 200
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    mix: str = "fineweb:0.59,stack:0.40,pairs:0.01"
    seed: int = 1337
    ckpt_every: int = 20
    milestone_every: int = 1000
    eval_every: int = 100
    eval_windows: int = 16
    wandb: bool = False
    samples: bool = True
    stop_at_step: int = 0


def parse_mix(text):
    pairs = [item.split(":") for item in text.split(",")]
    return [(name, float(weight)) for name, weight in pairs]
