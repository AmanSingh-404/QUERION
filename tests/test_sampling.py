import sqlite3

import pytest
import torch

from infer.engine import generate_batch, generate_cached
from infer.select import choose
from model.config import TINY
from model.transformer import Transformer

EOS = 1


def make():
    torch.manual_seed(0)
    return Transformer(TINY).eval()


def prompt():
    torch.manual_seed(1)
    return torch.randint(2, TINY.vocab_size, (6,)).tolist()


def test_row_zero_matches_greedy_decoding():
    model, ids = make(), prompt()
    ref = generate_cached(model, ids, 12, EOS, "cpu")[len(ids) :]
    if ref and ref[-1] == EOS:
        ref = ref[:-1]
    assert generate_batch(model, ids, 4, 12, EOS, "cpu")[0] == ref


def test_sampled_rows_differ_and_are_reproducible():
    model, ids = make(), prompt()
    a = generate_batch(model, ids, 6, 12, EOS, "cpu", temperature=1.5, seed=3)
    b = generate_batch(model, ids, 6, 12, EOS, "cpu", temperature=1.5, seed=3)
    assert a == b
    assert len({tuple(row) for row in a}) > 1


class EvenOnly:
    eos_id = EOS

    def reset(self):
        pass

    def accepts(self, cid):
        return cid % 2 == 0

    def advance(self, cid):
        pass


def test_constraints_apply_to_every_row():
    model, ids = make(), prompt()
    outs = generate_batch(model, ids, 4, 10, EOS, "cpu", constraints=[EvenOnly() for _ in range(4)])
    assert all(t % 2 == 0 for row in outs for t in row)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1), (2);")
    conn.commit()
    conn.close()
    return path


def test_choose_ignores_errors_and_empty_results_and_takes_the_majority(db):
    candidates = [
        "SELECT bad FROM t",
        "SELECT a FROM t WHERE a > 100",
        "SELECT a FROM t WHERE a = 1",
        "SELECT a FROM t",
        "SELECT a FROM t ORDER BY a DESC",
    ]
    assert choose(db, candidates) == "SELECT a FROM t"


def test_choose_falls_back_to_the_first_candidate(db):
    assert choose(db, ["SELECT bad FROM t", "SELECT worse FROM t"]) == "SELECT bad FROM t"
    assert choose(db, ["SELECT a FROM t WHERE a > 100"]) == "SELECT a FROM t WHERE a > 100"
