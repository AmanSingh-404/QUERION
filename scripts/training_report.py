import json
import statistics
import sys
from pathlib import Path

TOKENS_PER_STEP = 262_144
WARMUP = 300
DOCS = Path(__file__).resolve().parents[1] / "docs"


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mean_loss(rows, lo, hi):
    part = [r["loss"] for r in rows[lo:hi]]
    return sum(part) / len(part)


def summarize(rows):
    speeds = [r["tokens_per_sec"] for r in rows]
    median_speed = statistics.median(speeds)
    spikes = 0
    for i in range(max(WARMUP, 20), len(rows)):
        if rows[i]["loss"] > mean_loss(rows, i - 20, i) + 0.5:
            spikes += 1
    evals = [r for r in rows if "val_loss" in r]
    return {
        "steps": len(rows),
        "tokens": len(rows) * TOKENS_PER_STEP,
        "compute_hours": sum(r["step_time"] for r in rows) / 3600,
        "median_tokens_per_sec": median_speed,
        "slow_steps": sum(s < 0.5 * median_speed for s in speeds),
        "loss_spikes": spikes,
        "max_grad_norm_after_warmup": max(r["grad_norm"] for r in rows if r["step"] > WARMUP),
        "first_loss": rows[0]["loss"],
        "final_train_loss": mean_loss(rows, -20, len(rows)),
        "evals": evals,
    }


def render(s):
    final = s["evals"][-1]
    out = ["# Pretraining report", ""]
    out += ["| Item | Value |", "|---|---|"]
    out += [f"| Steps | {s['steps']:,} |", f"| Tokens trained | {s['tokens']:,} |"]
    out += [f"| Compute time (sum of step times) | {s['compute_hours']:.1f} hours |"]
    out += [f"| Median speed | {s['median_tokens_per_sec']:,.0f} tokens/s |"]
    out += [f"| Steps below half of median speed | {s['slow_steps']} |"]
    out += [f"| Loss spikes after warmup (+0.5 over recent average) | {s['loss_spikes']} |"]
    out += [f"| Largest gradient norm after warmup | {s['max_grad_norm_after_warmup']:.2f} |"]
    out += [f"| Training loss, first step / last 20 steps | {s['first_loss']:.3f} / "]
    out[-1] += f"{s['final_train_loss']:.3f} |"
    out += ["", "## Validation loss", "", "| Step | Planned mix | FineWeb | Stack SQL | Pairs |"]
    out += ["|---|---|---|---|---|"]
    keep = {250, 1000, 2500, 5000, final["step"]}
    for r in s["evals"]:
        if r["step"] in keep:
            out.append(
                f"| {r['step']:,} | {r['val_loss']:.4f} | {r['val_fineweb']:.4f} "
                f"| {r['val_stack']:.4f} | {r['val_pairs']:.4f} |"
            )
    return "\n".join(out) + "\n"


def plot(rows, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    steps = [r["step"] for r in rows]
    smooth = [mean_loss(rows, max(0, i - 19), i + 1) for i in range(len(rows))]
    axes[0].plot(steps, smooth)
    axes[0].set(title="Training loss (20-step average)", xlabel="step", ylim=(2, 8))
    evals = [r for r in rows if "val_loss" in r]
    for key, label in (("fineweb", "FineWeb"), ("stack", "Stack SQL"), ("pairs", "Pairs")):
        axes[1].plot([r["step"] for r in evals], [r[f"val_{key}"] for r in evals], label=label)
    axes[1].set(title="Validation loss", xlabel="step")
    axes[1].legend()
    axes[2].plot(steps, [r["tokens_per_sec"] for r in rows], linewidth=0.6)
    axes[2].set(title="Throughput (tokens/s)", xlabel="step")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "C:/querion-runs/main/log.jsonl"
    rows = read_rows(path)
    text = render(summarize(rows))
    (DOCS / "training_report.md").write_text(text, encoding="utf-8", newline="\n")
    plot(rows, DOCS / "training_curves.png")
    print(text)


if __name__ == "__main__":
    main()
