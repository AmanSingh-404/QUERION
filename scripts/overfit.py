import argparse
import time

import torch

from model.config import QUERION_100M, TINY
from model.transformer import Transformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["100m", "tiny"], default="100m")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = QUERION_100M if args.config == "100m" else TINY
    seq = min(args.seq, cfg.max_seq_len)
    torch.manual_seed(0)
    model = Transformer(cfg).to(device)
    print(f"device={device}  params={model.num_params():,}")
    x = torch.randint(0, cfg.vocab_size, (args.batch, seq), device=device)
    y = torch.randint(0, cfg.vocab_size, (args.batch, seq), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    amp = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")
    start = time.time()
    for step in range(1, args.steps + 1):
        with amp:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 20 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}", flush=True)
    print(f"final loss {loss.item():.4f}  ({time.time() - start:.0f}s)")
    if device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
