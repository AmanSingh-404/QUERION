import json
import sys
from pathlib import Path

BASE_WEIGHTS = {"fineweb": 0.59, "stack": 0.40, "pairs": 0.01}
GROUPS = {
    "Attention (GQA vs MHA)": ["base", "mha"],
    "Positions (RoPE vs learned)": ["base", "learned-pos"],
    "Data mix": ["base", "mix-sql-heavy", "mix-sql-heavy-p1"],
    "Learning rate": ["lr-3e-4", "base", "lr-3e-3", "lr-6e-3"],
}
DOCS = Path(__file__).resolve().parents[1] / "docs" / "experiments"


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def common_loss(row):
    """Validation loss re-weighted with the planned training mix, so runs are comparable."""
    return sum(w * row[f"val_{k}"] for k, w in BASE_WEIGHTS.items())


def final_metrics(rows):
    last = [r for r in rows if "val_loss" in r][-1]
    tail = rows[-20:]
    out = {"step": last["step"]}
    out.update({k: last[f"val_{k}"] for k in BASE_WEIGHTS})
    out["val_common"] = common_loss(last)
    out["train_loss"] = sum(r["loss"] for r in tail) / len(tail)
    out["tokens_per_sec"] = sum(r["tokens_per_sec"] for r in tail) / len(tail)
    return out


def load_runs(root):
    runs = {}
    for p in sorted(Path(root).iterdir()):
        if (p / "log.jsonl").exists():
            runs[p.name] = final_metrics(read_rows(p / "log.jsonl"))
    return runs


def build_table(runs):
    base = runs.get("base")
    header = ["Run", "Val loss (planned mix)", "FineWeb", "Stack SQL", "Pairs"]
    header += ["Train loss", "Tokens/s", "Delta vs base"]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, m in runs.items():
        delta = f"{m['val_common'] - base['val_common']:+.4f}" if base and name != "base" else "-"
        lines.append(
            f"| {name} | {m['val_common']:.4f} | {m['fineweb']:.4f} | {m['stack']:.4f} "
            f"| {m['pairs']:.4f} | {m['train_loss']:.4f} | {m['tokens_per_sec']:.0f} | {delta} |"
        )
    text = "\n".join(lines)
    seed2 = runs.get("base-seed2")
    if base and seed2:
        noise = abs(base["val_common"] - seed2["val_common"])
        text += f"\n\nNoise floor (base vs base-seed2, same setup, different seed): {noise:.4f}\n"
    return text


def plot_curves(root, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (title, names) in zip(axes.flat, GROUPS.items(), strict=True):
        for name in names:
            path = Path(root) / name / "log.jsonl"
            if not path.exists():
                continue
            evals = [r for r in read_rows(path) if "val_loss" in r]
            ax.plot([r["step"] for r in evals], [common_loss(r) for r in evals], label=name)
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("validation loss (planned mix)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "C:/querion-runs/ablations"
    DOCS.mkdir(parents=True, exist_ok=True)
    table = build_table(load_runs(root))
    (DOCS / "ablations.md").write_text(
        "# Ablation results\n\nProxy model, 100M tokens per run.\n\n" + table + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plot_curves(root, DOCS / "val_curves.png")
    print(table)


if __name__ == "__main__":
    main()
