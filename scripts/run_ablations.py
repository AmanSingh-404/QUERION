import sys

from train.checkpoint import load_latest
from train.config import TrainConfig
from train.train import run

OUT = "C:/querion-runs/ablations"
BASE = {
    "model": "proxy",
    "micro_batch": 2,
    "accum": 16,
    "total_steps": 1526,
    "warmup_steps": 100,
    "lr": 1e-3,
    "eval_every": 100,
    "eval_windows": 32,
    "ckpt_every": 100,
    "samples": False,
    "wandb": True,
}
RUNS = {
    "base": {},
    "base-seed2": {"seed": 2},
    "mha": {"n_kv_heads": 6},
    "learned-pos": {"pos_emb": "learned"},
    "mix-sql-heavy": {"mix": "fineweb:0.40,stack:0.58,pairs:0.02"},
    "lr-3e-4": {"lr": 3e-4},
    "lr-3e-3": {"lr": 3e-3},
}


def make_config(name, out_root=OUT):
    return TrainConfig(
        **{**BASE, **RUNS[name], "out_dir": f"{out_root}/{name}", "run_name": f"abl-{name}"}
    )


def is_done(out_dir, total_steps):
    state = load_latest(out_dir)
    return bool(state) and state["step"] >= total_steps


def main():
    for name in sys.argv[1:] or list(RUNS):
        cfg = make_config(name)
        if is_done(cfg.out_dir, cfg.total_steps):
            print(f"=== {name}: already done, skipping ===", flush=True)
            continue
        print(f"=== {name} ===", flush=True)
        run(cfg)


if __name__ == "__main__":
    main()
