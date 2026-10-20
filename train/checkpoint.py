import os
import shutil
from pathlib import Path

import torch


def save_checkpoint(out_dir, state, milestone=False):
    """Atomic save: write a temp file, keep the previous checkpoint, then swap in the new one."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp, latest, prev = out / "latest.pt.tmp", out / "latest.pt", out / "prev.pt"
    torch.save(state, tmp)
    if latest.exists():
        os.replace(latest, prev)
    os.replace(tmp, latest)
    if milestone:
        shutil.copy2(latest, out / f"step-{state['step']:06d}.pt")


def load_latest(out_dir):
    """Newest readable checkpoint (falls back to the previous one if latest is damaged)."""
    for name in ("latest.pt", "prev.pt"):
        path = Path(out_dir) / name
        if path.exists():
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"could not read {path}: {type(e).__name__}", flush=True)
    return None
