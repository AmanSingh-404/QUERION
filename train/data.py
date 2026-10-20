from pathlib import Path

import numpy as np
import torch


class ShardedTokens:
    """Random-access windows over one source's train shards (memory-mapped, uint16)."""

    def __init__(self, source_dir):
        paths = sorted(Path(source_dir).glob("train-*.bin"))
        if not paths:
            raise FileNotFoundError(f"no train-*.bin files in {source_dir}")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.sizes = np.array([len(a) for a in self.arrays])

    def window(self, rng, length):
        starts = np.maximum(self.sizes - length + 1, 0)
        k = rng.choice(len(self.arrays), p=starts / starts.sum())
        i = int(rng.integers(0, starts[k]))
        return self.arrays[k][i : i + length]


class MixedLoader:
    """Stateless loader: the batch for (step, micro) depends only on the seed, so a resumed
    run sees exactly the data an uninterrupted run would have seen."""

    def __init__(self, data_root, mix, seq_len, seed):
        self.names = [name for name, _ in mix]
        weights = np.array([w for _, w in mix], dtype=float)
        self.probs = weights / weights.sum()
        self.sources = {name: ShardedTokens(Path(data_root) / name) for name in self.names}
        self.seq_len, self.seed = seq_len, seed

    def batch(self, step, micro, batch_size):
        rng = np.random.default_rng([self.seed, step, micro])
        xs, ys = [], []
        for _ in range(batch_size):
            name = self.names[rng.choice(len(self.names), p=self.probs)]
            w = self.sources[name].window(rng, self.seq_len + 1).astype(np.int64)
            xs.append(w[:-1])
            ys.append(w[1:])
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def load_val_sets(data_root, names):
    out = {}
    for name in names:
        path = Path(data_root) / name / "val.bin"
        if path.exists():
            out[name] = np.memmap(path, dtype=np.uint16, mode="r")
    return out


def val_windows(arr, length, n):
    """Up to n evenly spaced, deterministic windows of the given length."""
    if len(arr) < length:
        return []
    starts = sorted(set(np.linspace(0, len(arr) - length, n).astype(int).tolist()))
    return [np.asarray(arr[s : s + length]).astype(np.int64) for s in starts]
