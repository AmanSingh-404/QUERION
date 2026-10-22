import json
import math
import sys

import numpy as np
import pytest
import torch

from train.checkpoint import load_latest, save_checkpoint
from train.config import TrainConfig, parse_mix
from train.data import MixedLoader, ShardedTokens, val_windows
from train.schedule import lr_at
from train.train import run
from train.watchdog import SlowdownWatch

SOURCES = ("fineweb", "stack", "pairs")


def make_shards(root, values=None, pattern=False, n_train=6000, n_val=2000):
    rng = np.random.default_rng(0)
    for i, name in enumerate(SOURCES):
        d = root / name
        d.mkdir(parents=True)
        for arr, fname in ((n_train, "train-0000.bin"), (n_val, "val.bin")):
            if pattern:
                data = np.arange(arr) % 7
            elif values is None:
                data = rng.integers(0, 100, arr)
            else:
                data = np.full(arr, values[i])
            data.astype(np.uint16).tofile(d / fname)


def tiny_cfg(tmp_path, **kw):
    base = {
        "data_root": str(tmp_path / "shards"),
        "out_dir": str(tmp_path / "run"),
        "model": "tiny",
        "seq_len": 16,
        "micro_batch": 2,
        "accum": 2,
        "total_steps": 6,
        "warmup_steps": 2,
        "lr": 3e-3,
        "ckpt_every": 100,
        "eval_every": 3,
        "eval_windows": 4,
        "samples": False,
    }
    base.update(kw)
    return TrainConfig(**base)


def test_parse_mix():
    assert parse_mix("a:0.5,b:0.5") == [("a", 0.5), ("b", 0.5)]


def test_lr_schedule_shape():
    cfg = TrainConfig(lr=1.0, warmup_steps=10, total_steps=110, min_lr_ratio=0.1)
    warm = [lr_at(s, cfg) for s in range(10)]
    assert warm == sorted(warm) and 0 < warm[0] < warm[-1] == pytest.approx(1.0)
    decay = [lr_at(s, cfg) for s in range(10, 110)]
    assert decay == sorted(decay, reverse=True)
    assert decay[0] == pytest.approx(1.0)
    assert lr_at(109, cfg) == pytest.approx(0.1, abs=1e-3)
    assert lr_at(500, cfg) == pytest.approx(0.1)
    assert lr_at(60, cfg) == pytest.approx(0.1 + 0.5 * 0.9, abs=0.02)


def test_windows_stay_inside_a_shard_and_are_shifted(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    np.arange(1000, dtype=np.uint16).tofile(d / "train-0000.bin")
    np.arange(5000, 5500, dtype=np.uint16).tofile(d / "train-0001.bin")
    src = ShardedTokens(d)
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = np.asarray(src.window(rng, 33)).astype(int)
        assert len(w) == 33
        assert (np.diff(w) == 1).all()


def test_loader_is_deterministic_and_shifted(tmp_path):
    make_shards(tmp_path / "shards")
    mix = parse_mix("fineweb:0.5,stack:0.3,pairs:0.2")
    a = MixedLoader(tmp_path / "shards", mix, 16, seed=7)
    b = MixedLoader(tmp_path / "shards", mix, 16, seed=7)
    x1, y1 = a.batch(5, 0, 4)
    x2, y2 = b.batch(5, 0, 4)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert x1.shape == (4, 16) and x1.dtype == torch.int64
    assert not torch.equal(x1, a.batch(6, 0, 4)[0])
    assert not torch.equal(x1, a.batch(5, 1, 4)[0])


def test_mixture_proportions(tmp_path):
    make_shards(tmp_path / "shards", values=(10, 20, 30))
    mix = parse_mix("fineweb:0.5,stack:0.4,pairs:0.1")
    loader = MixedLoader(tmp_path / "shards", mix, 8, seed=1)
    x, _ = loader.batch(0, 0, 4000)
    first = x[:, 0]
    shares = [(first == v).float().mean().item() for v in (10, 20, 30)]
    assert shares == pytest.approx([0.5, 0.4, 0.1], abs=0.04)


def test_val_windows_are_deterministic(tmp_path):
    arr = np.arange(100)
    a, b = val_windows(arr, 11, 5), val_windows(arr, 11, 5)
    assert all((x == y).all() for x, y in zip(a, b, strict=True))
    assert val_windows(np.arange(5), 11, 3) == []


def test_checkpoint_roundtrip_and_fallback(tmp_path):
    out = tmp_path / "ck"
    save_checkpoint(out, {"step": 1, "w": torch.ones(2)})
    save_checkpoint(out, {"step": 2, "w": torch.zeros(2)})
    assert load_latest(out)["step"] == 2
    assert (out / "prev.pt").exists() and not (out / "latest.pt.tmp").exists()
    (out / "latest.pt").write_bytes(b"corrupted")
    assert load_latest(out)["step"] == 1
    assert load_latest(tmp_path / "nothing") is None


def test_milestone_checkpoint_is_kept(tmp_path):
    save_checkpoint(tmp_path, {"step": 1000}, milestone=True)
    assert (tmp_path / "step-001000.pt").exists()


def test_resume_matches_uninterrupted_run(tmp_path):
    make_shards(tmp_path / "shards")
    straight = run(tiny_cfg(tmp_path, out_dir=str(tmp_path / "a")))
    cfg_b = tiny_cfg(tmp_path, out_dir=str(tmp_path / "b"))
    run(tiny_cfg(tmp_path, out_dir=str(tmp_path / "b"), stop_at_step=3))
    assert load_latest(tmp_path / "b")["step"] == 3
    resumed = run(cfg_b)
    for (name, p1), (_, p2) in zip(
        straight.named_parameters(), resumed.named_parameters(), strict=True
    ):
        assert torch.allclose(p1, p2, atol=1e-5), name
    assert load_latest(tmp_path / "b")["step"] == 6


def test_training_reduces_loss_and_logs(tmp_path):
    make_shards(tmp_path / "shards", pattern=True)
    cfg = tiny_cfg(tmp_path, total_steps=40, accum=1, warmup_steps=5, eval_every=40)
    run(cfg)
    with open(tmp_path / "run" / "log.jsonl") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 40
    assert math.isfinite(rows[-1]["val_loss"])
    assert rows[-1]["loss"] < 0.5 * rows[0]["loss"]


def test_logging_failure_does_not_stop_training(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    make_shards(tmp_path / "shards")
    run(tiny_cfg(tmp_path, wandb=True, total_steps=2))
    assert load_latest(tmp_path / "run")["step"] == 2


def test_model_overrides_are_applied(tmp_path):
    make_shards(tmp_path / "shards")
    model = run(tiny_cfg(tmp_path, total_steps=2, n_kv_heads=4, pos_emb="learned"))
    assert model.cfg.n_kv_heads == 4 and model.cfg.pos_emb == "learned"
    assert model.pos_emb is not None


def test_refuses_to_train_a_real_model_on_the_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        run(tiny_cfg(tmp_path, model="proxy"))


def test_main_config_defaults_are_frozen():
    cfg = TrainConfig()
    assert (cfg.model, cfg.seq_len, cfg.micro_batch, cfg.accum) == ("100m", 2048, 1, 128)
    assert cfg.lr == 1.5e-3 and cfg.warmup_steps == 300 and cfg.min_lr_ratio == 0.1
    assert cfg.total_steps == 7630 and cfg.weight_decay == 0.1 and cfg.grad_clip == 1.0
    assert parse_mix(cfg.mix) == [("fineweb", 0.59), ("stack", 0.40), ("pairs", 0.01)]
    assert cfg.micro_batch * cfg.accum * cfg.seq_len * cfg.total_steps == pytest.approx(
        2.0e9, rel=0.01
    )


def test_slowdown_watch_warns_after_sustained_drop_only():
    watch = SlowdownWatch()
    assert [watch.update(12000.0) for _ in range(12)] == [None] * 12
    assert watch.update(3000.0) is None  # a single slow step is ignored
    assert watch.update(12000.0) is None
    msgs = [watch.update(2600.0) for _ in range(5)]
    assert msgs[:4] == [None] * 4 and "unplugged" in msgs[4]
    assert watch.update(12000.0) is None  # recovery resets the counter


def patch_clip(monkeypatch, bad_calls):
    real = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def fake(params, max_norm):
        calls["n"] += 1
        norm = real(params, max_norm)
        return torch.tensor(float("nan")) if calls["n"] in bad_calls else norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake)


def test_a_non_finite_gradient_skips_the_update(tmp_path, monkeypatch):
    make_shards(tmp_path / "shards")
    patch_clip(monkeypatch, bad_calls={2})
    model = run(tiny_cfg(tmp_path, total_steps=4, accum=1))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with open(tmp_path / "run" / "log.jsonl") as f:
        assert len(f.readlines()) == 3


def test_repeated_non_finite_gradients_stop_training(tmp_path, monkeypatch):
    make_shards(tmp_path / "shards")
    patch_clip(monkeypatch, bad_calls=set(range(1, 100)))
    with pytest.raises(SystemExit):
        run(tiny_cfg(tmp_path, total_steps=10, accum=1))
    assert load_latest(tmp_path / "run") is None
