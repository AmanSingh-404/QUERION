from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from data.tokenize_shards import batches, encode_all, has_special

TOK = Tokenizer.from_file(str(Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"))
DOCS = [
    None if i % 17 == 0 else f"<|bos|>SELECT col{i} FROM table{i % 5} WHERE x > {i}<|eos|>"
    for i in range(1000)
]


def make_factory(fail_after=None):
    def factory(skip):
        for i in range(skip, len(DOCS)):
            if fail_after is not None and i >= fail_after:
                raise RuntimeError("simulated crash")
            yield DOCS[i]

    return factory


def read_train(out_dir, n_shards):
    parts = [np.fromfile(f"{out_dir}/train-{k:04d}.bin", dtype=np.uint16) for k in range(n_shards)]
    return np.concatenate(parts)


def test_has_special_and_batches():
    assert has_special("x <|sql|> y")
    assert not has_special("SELECT 1")
    assert [len(b) for b in batches(range(7), 3)] == [3, 3, 1]


def test_token_counts_match_direct_encoding(tmp_path):
    out = str(tmp_path / "a")
    state = encode_all(TOK, out, make_factory(), shard_tokens=2000, batch_size=64)
    assert state["done"]
    kept = [d for d in DOCS if d is not None]
    lengths = [len(e.ids) for e in TOK.encode_batch(kept)]
    val_idx = set(range(0, len(kept), 200))
    assert state["train_tokens"] == sum(n for i, n in enumerate(lengths) if i not in val_idx)
    assert state["val_bytes"] // 2 == sum(lengths[i] for i in val_idx)
    train = read_train(out, state["train_shards"])
    assert len(train) == state["train_tokens"]
    assert int(train.max()) < 24000
    assert state["drops"]["filtered_in_source"] == sum(d is None for d in DOCS)


def test_crash_and_resume_gives_identical_output(tmp_path):
    clean = str(tmp_path / "clean")
    crashed = str(tmp_path / "crashed")
    a = encode_all(TOK, clean, make_factory(), shard_tokens=1500, batch_size=50)
    with pytest.raises(RuntimeError):
        encode_all(TOK, crashed, make_factory(fail_after=620), shard_tokens=1500, batch_size=50)
    b = encode_all(TOK, crashed, make_factory(), shard_tokens=1500, batch_size=50)
    assert a["train_tokens"] == b["train_tokens"]
    assert a["val_bytes"] == b["val_bytes"]
    assert a["train_shards"] == b["train_shards"]
    assert np.array_equal(
        read_train(clean, a["train_shards"]), read_train(crashed, b["train_shards"])
    )
    assert Path(f"{clean}/val.bin").read_bytes() == Path(f"{crashed}/val.bin").read_bytes()


def test_max_len_drops_long_docs(tmp_path):
    out = str(tmp_path / "m")
    state = encode_all(TOK, out, make_factory(), max_len=5, shard_tokens=2000, batch_size=64)
    assert state["drops"]["too_long"] > 0
