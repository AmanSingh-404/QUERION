from pathlib import Path

from tokenizers import Tokenizer

from data.prompt import (
    clean_sql,
    format_example,
    format_prompt,
    statements_from_gretel,
    statements_from_lines,
)
from tokenizer.special_tokens import TOKEN_ID

PATH = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"


def test_gretel_keeps_only_create_statements():
    ctx = "CREATE TABLE t (a INT); INSERT INTO t VALUES (1); CREATE VIEW v AS SELECT a FROM t;"
    assert statements_from_gretel(ctx) == [
        "CREATE TABLE t (a INT)",
        "CREATE VIEW v AS SELECT a FROM t",
    ]


def test_lines_format():
    text = "CREATE TABLE a (x INT)\nCREATE TABLE b (y INT);"
    assert statements_from_lines(text) == ["CREATE TABLE a (x INT)", "CREATE TABLE b (y INT)"]


def test_clean_sql_strips_semicolons():
    assert clean_sql("  SELECT 1 ;; ") == "SELECT 1"


def test_prompt_layout_text():
    p = format_prompt(["CREATE TABLE t (a INT)"], "How many rows?")
    assert p == "<|bos|><|schema|>CREATE TABLE t (a INT)<|question|>How many rows?<|sql|>"


def test_example_ends_with_eos():
    e = format_example(["CREATE TABLE t (a INT)"], "How many rows?", "SELECT COUNT(*) FROM t;")
    assert e.endswith("<|sql|>SELECT COUNT(*) FROM t<|eos|>")


def test_example_token_structure():
    tok = Tokenizer.from_file(str(PATH))
    e = format_example(["CREATE TABLE t (a INT)"], "How many rows?", "SELECT COUNT(*) FROM t")
    ids = tok.encode(e).ids
    marks = [TOKEN_ID[t] for t in ("<|bos|>", "<|schema|>", "<|question|>", "<|sql|>", "<|eos|>")]
    positions = [ids.index(m) for m in marks]
    assert positions == sorted(positions)
    assert ids[0] == TOKEN_ID["<|bos|>"] and ids[-1] == TOKEN_ID["<|eos|>"]
    assert all(ids.count(m) == 1 for m in marks)
