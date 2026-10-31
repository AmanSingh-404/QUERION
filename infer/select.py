from collections import Counter

from eval.compare import norm_cell
from eval.executor import execute


def signature(rows):
    cells = [tuple(norm_cell(v) for v in row) for row in rows]
    return tuple(sorted(cells, key=repr))


def choose(db_path, candidates, timeout=5.0):
    """Execution-guided pick: the most common non-empty result among runnable candidates.

    Ties go to the earliest candidate (candidate 0 is the greedy one)."""
    runs = []
    for i, sql in enumerate(candidates):
        if not sql:
            continue
        rows, err = execute(db_path, sql, timeout)
        if err is None:
            runs.append((i, signature(rows), bool(rows)))
    if not runs:
        return candidates[0]
    pool = [r for r in runs if r[2]] or runs
    votes = Counter(sig for _, sig, _ in pool)
    best = max(votes.values())
    for i, sig, _ in pool:
        if votes[sig] == best:
            return candidates[i]
    return candidates[0]
