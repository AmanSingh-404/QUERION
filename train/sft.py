import argparse
import json
import math
import random
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.config import PROXY_20M, QUERION_100M, TINY
from model.transformer import Transformer
from tokenizer.special_tokens import EOS_ID
from train.checkpoint import load_latest, save_checkpoint
from train.sample import generate
from train.schedule import lr_at
from train.sft_config import SFTConfig
from train.sft_data import collate, encode_example
from train.train import SQL_PROMPT, TOK_PATH, make_optimizer


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode_all(tok, examples):
    pairs, sources = [], []
    for ex in examples:
        p = encode_example(tok, ex)
        if p is not None:
            pairs.append(p)
            sources.append(ex["source"])
    return pairs, sources


def pack(order, lengths, budget):
    """Group length-sorted example ids into micro-batches of at most `budget` padded tokens."""
    out, cur, width = [], [], 0
    for i in order:
        if cur and max(width, lengths[i]) * (len(cur) + 1) > budget:
            out.append(cur)
            cur, width = [], 0
        cur.append(i)
        width = max(width, lengths[i])
    if cur:
        out.append(cur)
    return out


def make_plan(lengths, sources, cfg):
    """All optimizer steps of all epochs, each a list of micro-batches of example ids."""
    plan = []
    for epoch in range(cfg.epochs):
        rng = random.Random(cfg.seed + epoch)
        idx = []
        for i, s in enumerate(sources):
            idx += [i] * (cfg.spider_repeat if s == "spider" else 1)
        rng.shuffle(idx)
        micro = []
        for s in range(0, len(idx), 512):
            chunk = sorted(idx[s : s + 512], key=lambda i: lengths[i])
            micro += pack(chunk, lengths, cfg.micro_tokens)
        rng.shuffle(micro)
        plan += [micro[j : j + cfg.accum] for j in range(0, len(micro), cfg.accum)]
    return plan


@torch.no_grad()
def evaluate(model, val, cfg, device, amp):
    model.eval()
    out = {}
    for name, pairs in val.items():
        lengths = [len(x) for x, _ in pairs]
        order = sorted(range(len(pairs)), key=lambda i: lengths[i])
        total, count = 0.0, 0
        for group in pack(order, lengths, cfg.micro_tokens):
            x, y = collate([pairs[i] for i in group])
            n = int((y != -100).sum())
            with amp:
                loss = model(x.to(device), y.to(device))[1]
            total += loss.item() * n
            count += n
        out[name] = total / count
    model.train()
    return out


def run(cfg):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available: refusing to fine-tune on the CPU")
    device = "cuda"
    tok = Tokenizer.from_file(str(TOK_PATH))
    pairs, sources = encode_all(tok, read_jsonl(Path(cfg.sft_dir) / "train.jsonl"))
    val_pairs, val_sources = encode_all(tok, read_jsonl(Path(cfg.sft_dir) / "val.jsonl"))
    val = {
        s: [p for p, src in zip(val_pairs, val_sources, strict=True) if src == s]
        for s in sorted(set(val_sources))
    }
    lengths = [len(x) for x, _ in pairs]
    plan = make_plan(lengths, sources, cfg)
    cfg = replace(cfg, total_steps=len(plan))
    print(f"train examples {len(pairs)}, steps {cfg.total_steps}, epochs {cfg.epochs}", flush=True)

    mcfg = {"100m": QUERION_100M, "tiny": TINY, "proxy": PROXY_20M}[cfg.model]
    torch.manual_seed(cfg.seed)
    model = Transformer(mcfg).to(device)
    optimizer = make_optimizer(model, cfg, device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    state = load_latest(out_dir)
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        print(f"resumed SFT from step {start_step}", flush=True)
    else:
        init = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(init["model"])
        print(f"loaded pretrained weights from {cfg.init_ckpt} (step {init['step']})", flush=True)
        del init
    del state

    amp = torch.autocast(device_type=device, dtype=torch.bfloat16)

    def snapshot(step):
        return {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_cfg": asdict(cfg),
            "wandb_id": None,
        }

    log_file = open(out_dir / "log.jsonl", "a", encoding="utf-8")
    done, started, bad_steps = start_step, time.time(), 0
    model.train()
    try:
        for step in range(start_step, cfg.total_steps):
            t0 = time.time()
            lr = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batches = [collate([pairs[i] for i in g]) for g in plan[step]]
            n_total = sum(int((y != -100).sum()) for _, y in batches)
            loss_sum = torch.zeros((), device=device)
            for x, y in batches:
                frac = int((y != -100).sum()) / n_total
                with amp:
                    loss = model(x.to(device), y.to(device))[1]
                (loss * frac).backward()
                loss_sum += loss.detach() * frac
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            if not math.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                bad_steps += 1
                print(f"WARNING: non-finite gradient at step {step + 1}, skipped", flush=True)
                if bad_steps >= 5:
                    raise SystemExit("too many non-finite gradients: stopping")
                continue
            bad_steps = 0
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            done = step + 1
            dt = time.time() - t0
            row = {
                "step": done,
                "loss": loss_sum.item(),
                "lr": lr,
                "grad_norm": grad_norm,
                "target_tokens": n_total,
                "step_time": dt,
            }
            if done % cfg.eval_every == 0 or done == cfg.total_steps:
                row.update(
                    {f"val_{k}": v for k, v in evaluate(model, val, cfg, device, amp).items()}
                )
                ids = generate(model, tok.encode(SQL_PROMPT).ids, 64, EOS_ID, device)
                print(f"--- sample ---\n{tok.decode(ids, skip_special_tokens=False)}\n", flush=True)
                model.train()
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            eta_m = (cfg.total_steps - done) * (time.time() - started) / (done - start_step) / 60
            val_txt = "".join(f"  {k} {v:.4f}" for k, v in row.items() if k.startswith("val_"))
            print(
                f"step {done}/{cfg.total_steps}  loss {row['loss']:.4f}  lr {lr:.2e}  "
                f"gnorm {grad_norm:.2f}  {dt:.1f}s/step  eta {eta_m:.0f}m{val_txt}",
                flush=True,
            )
            stop = bool(cfg.stop_at_step) and done >= cfg.stop_at_step
            last = done == cfg.total_steps
            if done % cfg.ckpt_every == 0 or last or stop:
                save_checkpoint(out_dir, snapshot(done), milestone=last)
            if stop:
                break
    except KeyboardInterrupt:
        print(f"interrupted: saving checkpoint at step {done}", flush=True)
        save_checkpoint(out_dir, snapshot(done))
    finally:
        log_file.close()
    return model


def main():
    p = argparse.ArgumentParser()
    defaults = SFTConfig()
    for f in fields(SFTConfig):
        value = getattr(defaults, f.name)
        p.add_argument(
            "--" + f.name.replace("_", "-"), dest=f.name, type=type(value), default=value
        )
    run(SFTConfig(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
