from scripts.training_report import render, summarize


def fake_rows(n=400):
    rows = []
    for i in range(1, n + 1):
        loss = 10.0 / (1 + i / 20)
        if i == 350:
            loss += 2.0
        row = {"step": i, "loss": loss, "grad_norm": 1.0, "step_time": 20.0}
        row["tokens_per_sec"] = 3000.0 if 100 <= i < 110 else 12000.0
        if i % 100 == 0:
            row.update({"val_loss": 3.0, "val_fineweb": 4.0, "val_stack": 1.5, "val_pairs": 1.0})
        rows.append(row)
    return rows


def test_summary_counts_slow_steps_and_spikes():
    s = summarize(fake_rows())
    assert s["steps"] == 400 and s["tokens"] == 400 * 262_144
    assert s["slow_steps"] == 10
    assert s["loss_spikes"] == 1
    assert s["compute_hours"] == 400 * 20 / 3600


def test_render_contains_the_key_numbers():
    text = render(summarize(fake_rows()))
    assert "Tokens trained" in text and "Validation loss" in text
    assert "| 400 | 3.0000 | 4.0000 | 1.5000 | 1.0000 |" in text
