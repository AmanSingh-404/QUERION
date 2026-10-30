import json
import sqlite3
from pathlib import Path

import torch
from tokenizers import Tokenizer

from data.leakage import build_needles
from tokenizer.special_tokens import EOS_ID, PAD_ID, TOKEN_ID
from train.sft_data import (
    collate,
    drop_contaminated,
    encode_example,
    fix_commas,
    is_val_db,
    spider_examples,
    split_examples,
)

TOK = Tokenizer.from_file(str(Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"))
EX = {
    "source": "synth",
    "schema": ["CREATE TABLE t (a INT)"],
    "question": "How many rows?",
    "sql": "SELECT COUNT(*) FROM t;",
}


def test_fix_commas():
    assert fix_commas("SELECT name ,  country ,  age FROM singer") == (
        "SELECT name, country, age FROM singer"
    )


def test_only_the_answer_has_targets():
    x, y = encode_example(TOK, EX)
    assert len(x) == len(y)
    sql_marker = x.index(TOKEN_ID["<|sql|>"])
    assert all(t == -100 for t in y[:sql_marker])
    assert all(t != -100 for t in y[sql_marker:])
    assert y[-1] == EOS_ID
    answer = TOK.decode(y[sql_marker:-1])
    assert answer == "SELECT COUNT(*) FROM t"


def test_too_long_examples_are_dropped():
    assert encode_example(TOK, EX, max_len=5) is None


def test_collate_pads_inputs_and_masks_padding():
    a, b = (
        encode_example(TOK, EX),
        encode_example(TOK, {**EX, "sql": "SELECT 1 FROM t WHERE a > 5"}),
    )
    xs, ys = collate([a, b])
    assert xs.shape == ys.shape == (2, max(len(a[0]), len(b[0])))
    short = 0 if len(a[0]) < len(b[0]) else 1
    n = len(min(a[0], b[0], key=len))
    assert (xs[short, n:] == PAD_ID).all() and (ys[short, n:] == -100).all()
    assert xs.dtype == torch.long


def test_contaminated_examples_are_dropped():
    needles = build_needles(
        [{"question": "How many singers do we have in total?", "sql": "SELECT 1"}]
    )
    bad = {**EX, "question": "how many  singers do we have in total?"}
    kept, dropped = drop_contaminated([EX, bad], needles)
    assert kept == [EX] and dropped == {"synth": 1}


def test_split_is_by_database_for_spider():
    ex = [{"source": "spider", "db_id": f"db{i}", "question": "q", "sql": "s"} for i in range(60)]
    ex += [{**ex[0], "source": "spider"} for _ in range(3)]
    train, val = split_examples(ex)
    val_dbs = {e["db_id"] for e in val}
    assert val_dbs and val_dbs.isdisjoint({e["db_id"] for e in train})
    assert all(is_val_db(d) for d in val_dbs)


def test_split_holds_out_random_non_spider_examples():
    ex = [{"source": "gretel", "question": str(i), "sql": "s"} for i in range(900)]
    ex += [{"source": "synth", "question": str(i), "sql": "s"} for i in range(500)]
    train, val = split_examples(ex)
    assert sum(e["source"] == "gretel" for e in val) == 500
    assert sum(e["source"] == "synth" for e in val) == 300
    assert len(train) + len(val) == len(ex)


def test_spider_examples_skip_broken_gold(tmp_path):
    d = tmp_path / "database" / "shop"
    d.mkdir(parents=True)
    conn = sqlite3.connect(d / "shop.sqlite")
    conn.executescript("CREATE TABLE item (id INTEGER, price REAL);")
    conn.commit()
    conn.close()
    rows = [
        {"db_id": "shop", "question": "How many items?", "query": "SELECT count(*) FROM item"},
        {"db_id": "shop", "question": "Broken", "query": "SELECT nope FROM item"},
        {"db_id": "gone", "question": "No database", "query": "SELECT 1"},
    ]
    (tmp_path / "train_spider.json").write_text(json.dumps(rows))
    out, skipped = spider_examples(tmp_path)
    assert skipped == 2 and len(out) == 1
    assert out[0]["schema"][0].startswith("CREATE TABLE item") and out[0]["source"] == "spider"


def test_prompt_tokens_are_a_stable_prefix_for_varied_answers():
    for sql in ("SELECT 1", "select a,b from t where a = 'x'", "SELECT *"):
        assert encode_example(TOK, {**EX, "sql": sql}) is not None
