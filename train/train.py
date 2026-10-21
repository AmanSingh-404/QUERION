import argparse
import json
import time
import uuid
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch
from tokenizers import Tokenizer

from data.prompt import format_prompt
from model.config import PROXY_20M, QUERION_100M, TINY
from model.transformer import Transformer
from tokenizer.special_tokens import EOS_ID
from train.checkpoint import load_latest, save_checkpoint
from train.config import TrainConfig, parse_mix
from train.data import MixedLoader, load_val_sets, val_windows
from train.sample import generate
from train.schedule import lr_at

TOK_PATH = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"
SQL_PROMPT = format_prompt(
    ["CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER)"],
    "What is the average salary of employees in each department?",
)
TEXT_PROMPT = "<|bos|>The capital of France is"


def make_optimizer(model, cfg, device):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8, fused=device == "cuda"
    )


@torch.no_grad()
def evaluate(model, val_sets, cfg, device, amp):
    model.eval()
    out = {}
    for name, arr in val_sets.items():
        losses = []
        for w in val_windows(arr, cfg.seq_len + 1, cfg.eval_windows):
            x = torch.from_numpy(w[:-1])[None].to(device)
            y = torch.from_numpy(w[1:])[None].to(device)
            with amp:
                losses.append(model(x, y)[1].item())
        if losses:
            out[name] = sum(losses) / len(losses)
    model.train()
    return out


def make_samples(model, tok, device):
    texts = {}
    for label, prompt in (("sql", SQL_PROMPT), ("text", TEXT_PROMPT)):
        ids = generate(model, tok.encode(prompt).ids, 64, EOS_ID, device)
        texts[label] = tok.decode(ids, skip_special_tokens=False)
    model.train()
    return texts


def run(cfg: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = {"100m": QUERION_100M, "tiny": TINY, "proxy": PROXY_20M}[cfg.model]
    overrides = {}
    if cfg.n_kv_heads:
        overrides["n_kv_heads"] = cfg.n_kv_heads
    if cfg.pos_emb:
        overrides["pos_emb"] = cfg.pos_emb
    if overrides:
        mcfg = replace(mcfg, **overrides)
    torch.manual_seed(cfg.seed)
    model = Transformer(mcfg).to(device)
    optimizer = make_optimizer(model, cfg, device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step, wandb_id = 0, None
    state = load_latest(out_dir)
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, wandb_id = state["step"], state.get("wandb_id")
        print(f"resumed from step {start_step}", flush=True)
    del state

    mix = parse_mix(cfg.mix)
    loader = MixedLoader(cfg.data_root, mix, cfg.seq_len, cfg.seed)
    val_sets = load_val_sets(cfg.data_root, [name for name, _ in mix])
    weights = {name: w / sum(w for _, w in mix) for name, w in mix}
    tok = Tokenizer.from_file(str(TOK_PATH)) if cfg.samples else None
    amp = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")

    wb = None
    if cfg.wandb:
        try:
            import wandb

            wandb_id = wandb_id or uuid.uuid4().hex[:8]
            wb = wandb.init(
                project="querion",
                name=cfg.run_name,
                id=wandb_id,
                resume="allow",
                config=asdict(cfg),
            )
        except Exception as e:
            print(f"W&B disabled: {type(e).__name__}: {e}", flush=True)
            wb = None

    def snapshot(step):
        return {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_cfg": asdict(cfg),
            "wandb_id": wandb_id,
        }

    tokens_per_step = cfg.micro_batch * cfg.accum * cfg.seq_len
    log_file = open(out_dir / "log.jsonl", "a", encoding="utf-8")
    done, started = start_step, time.time()
    model.train()
    try:
        for step in range(start_step, cfg.total_steps):
            t0 = time.time()
            lr = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_sum = torch.zeros((), device=device)
            for micro in range(cfg.accum):
                x, y = loader.batch(step, micro, cfg.micro_batch)
                with amp:
                    loss = model(x.to(device), y.to(device))[1]
                (loss / cfg.accum).backward()
                loss_sum += loss.detach()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            done = step + 1
            dt = time.time() - t0
            row = {
                "step": done,
                "loss": loss_sum.item() / cfg.accum,
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_sec": tokens_per_step / dt,
                "step_time": dt,
            }
            if device == "cuda":
                row["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            if done % cfg.eval_every == 0 or done == cfg.total_steps:
                val = evaluate(model, val_sets, cfg, device, amp)
                row.update({f"val_{k}": v for k, v in val.items()})
                row["val_loss"] = sum(weights[k] * v for k, v in val.items()) / sum(
                    weights[k] for k in val
                )
                if tok is not None:
                    for label, text in make_samples(model, tok, device).items():
                        print(f"--- sample ({label}) ---\n{text}\n", flush=True)
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            if wb is not None:
                try:
                    wb.log(row, step=done)
                except Exception as e:
                    print(f"W&B logging stopped: {type(e).__name__}: {e}", flush=True)
                    wb = None
            eta_h = (cfg.total_steps - done) * (time.time() - started) / (done - start_step) / 3600
            print(
                f"step {done}/{cfg.total_steps}  loss {row['loss']:.4f}  lr {lr:.2e}  "
                f"gnorm {grad_norm:.2f}  {row['tokens_per_sec']:.0f} tok/s  "
                f"{dt:.1f}s/step  eta {eta_h:.1f}h"
                + (f"  val {row['val_loss']:.4f}" if "val_loss" in row else ""),
                flush=True,
            )
            stop = bool(cfg.stop_at_step) and done >= cfg.stop_at_step
            if done % cfg.ckpt_every == 0 or done == cfg.total_steps or stop:
                save_checkpoint(out_dir, snapshot(done), milestone=done % cfg.milestone_every == 0)
            if stop:
                break
    except KeyboardInterrupt:
        print(f"interrupted: saving checkpoint at step {done}", flush=True)
        save_checkpoint(out_dir, snapshot(done))
    finally:
        log_file.close()
        if wb is not None:
            try:
                wb.finish()
            except Exception as e:
                print(f"W&B finish failed: {type(e).__name__}", flush=True)
    return model


def main():
    p = argparse.ArgumentParser()
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        value = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if value is True:
            p.add_argument("--no-" + f.name.replace("_", "-"), dest=f.name, action="store_false")
        elif value is False:
            p.add_argument(flag, dest=f.name, action="store_true")
        else:
            p.add_argument(flag, type=type(value), default=value)
    run(TrainConfig(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
