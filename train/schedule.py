import math


def lr_at(step, cfg):
    """Linear warmup, then cosine decay to lr * min_lr_ratio. step is 0-indexed."""
    min_lr = cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.total_steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - min_lr)
