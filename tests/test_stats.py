import json

import numpy as np
import pytest

from data.stats import analyze_source, doc_lengths, render


def write_source(tmp_path, docs, val_docs):
    train = np.array([t for d in docs for t in d], dtype=np.uint16)
    val = np.array([t for d in val_docs for t in d], dtype=np.uint16)
    train.tofile(tmp_path / "train-0000.bin")
    val.tofile(tmp_path / "val.bin")
    summary = {"train_tokens": len(train), "val_tokens": len(val)}
    (tmp_path / "summary.json").write_text(json.dumps(summary))


def test_doc_lengths():
    arr = np.array([1, 5, 6, 2, 1, 7, 2, 1, 9, 9, 9, 2], dtype=np.uint16)
    assert doc_lengths(arr).tolist() == [4, 3, 5]


def test_analyze_source(tmp_path):
    docs = [[1, 5, 6, 2], [1, 7, 2], [1] + [9] * 3000 + [2]]
    write_source(tmp_path, docs, [[1, 8, 2]])
    r = analyze_source(tmp_path)
    assert r["docs"] == 3
    assert r["train_tokens"] == 4 + 3 + 3002
    assert r["val_tokens"] == 3
    assert r["max"] == 3002
    assert r["problems"] == []
    assert round(r["docs_over_ctx_pct"], 1) == 33.3
    assert sum(r["hist"]) == 3


def test_problems_are_reported(tmp_path):
    write_source(tmp_path, [[1, 5, 2], [1, 7, 2]], [[1, 8, 2]])
    (tmp_path / "summary.json").write_text(json.dumps({"train_tokens": 99, "val_tokens": 3}))
    assert analyze_source(tmp_path)["problems"]


def test_render_contains_tables(tmp_path):
    write_source(tmp_path, [[1, 5, 6, 2], [1, 7, 2]], [[1, 8, 2]])
    r = analyze_source(tmp_path)
    text = render({"pairs": r})
    assert "| pairs |" in text and "All checks passed." in text


def test_missing_shards_give_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_source(tmp_path)
