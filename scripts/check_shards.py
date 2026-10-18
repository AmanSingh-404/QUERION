import sys
from pathlib import Path

EXPECTED = {"fineweb": 12, "stack": 4, "pairs": 1}


def missing_files(root):
    out = []
    for name, n in EXPECTED.items():
        need = [f"train-{i:04d}.bin" for i in range(n)] + ["val.bin", "summary.json", "state.json"]
        out += [f"{name}/{f}" for f in need if not (Path(root) / name / f).exists()]
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "C:/querion-data/shards"
    missing = missing_files(root)
    print("all files present" if not missing else "missing:\n" + "\n".join(missing))
