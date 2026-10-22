import statistics


class SlowdownWatch:
    """Warns when training throughput collapses (unplugged laptop, overheating, sleep)."""

    def __init__(self, warmup=10, threshold=0.5, patience=5):
        self.warmup, self.threshold, self.patience = warmup, threshold, patience
        self.history, self.baseline, self.slow = [], None, 0

    def update(self, tokens_per_sec):
        if self.baseline is None:
            self.history.append(tokens_per_sec)
            if len(self.history) >= self.warmup:
                self.baseline = statistics.median(self.history[2:])
            return None
        if tokens_per_sec >= self.threshold * self.baseline:
            self.slow = 0
            return None
        self.slow += 1
        over = self.slow - self.patience
        if over == 0 or (over > 0 and over % 50 == 0):
            return (
                f"WARNING: throughput is {100 * tokens_per_sec / self.baseline:.0f}% of normal "
                f"for {self.slow} steps. Is the laptop unplugged, overheating or asleep?"
            )
        return None
