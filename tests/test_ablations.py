import json

import pytest

from scripts.run_ablations import BASE, RUNS, make_config
from scripts.summarize_ablations import build_table, common_loss, final_metrics, load_runs


def write_run(root, name, val, steps=40):
    d = root / name
    d.mkdir(parents=True)
    rows = []
    for s in range(1, steps + 1):
        row = {"step": s, "loss": 5.0 - val, "tokens_per_sec": 1000.0}
        if s == steps:
            row.update({"val_fineweb": val, "val_stack": val, "val_pairs": val, "val_loss": val})
        rows.append(row)
    d.joinpath("log.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_every_run_builds_a_valid_config():
    for name in RUNS:
        cfg = make_config(name)
        assert cfg.run_name == f"abl-{name}"
        tokens = cfg.micro_batch * cfg.accum * cfg.seq_len * cfg.total_steps
        assert tokens == pytest.approx(100e6, rel=0.01)
    assert make_config("mha").n_kv_heads == 6 and make_config("base").n_kv_heads == 0
    assert BASE["samples"] is False


def test_common_loss_uses_the_planned_mix():
    row = {"val_fineweb": 7.0, "val_stack": 3.0, "val_pairs": 1.0}
    assert common_loss(row) == pytest.approx(0.59 * 7 + 0.40 * 3 + 0.01 * 1)


def test_final_metrics_uses_the_last_evaluation():
    rows = [
        {"step": 1, "loss": 3.0, "tokens_per_sec": 10.0},
        {
            "step": 2,
            "loss": 1.0,
            "tokens_per_sec": 20.0,
            "val_fineweb": 2.0,
            "val_stack": 2.0,
            "val_pairs": 2.0,
            "val_loss": 2.0,
        },
    ]
    m = final_metrics(rows)
    assert m["step"] == 2 and m["val_common"] == pytest.approx(2.0)
    assert m["train_loss"] == pytest.approx(2.0)


def test_summary_table_and_noise_floor(tmp_path):
    write_run(tmp_path, "base", 4.00)
    write_run(tmp_path, "base-seed2", 4.02)
    write_run(tmp_path, "mha", 3.95)
    table = build_table(load_runs(tmp_path))
    assert "| mha |" in table and "-0.0500" in table
    assert "Noise floor" in table and "0.0200" in table
