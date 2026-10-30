import random
import sqlite3
from types import SimpleNamespace

from train.augment import augment, augment_rows, identifiers

SCHEMA = [
    'CREATE TABLE "singer" (\n"Singer_ID" int,\n"Name" text,\n"Country" text,\n"Age" int,\n'
    'PRIMARY KEY ("Singer_ID")\n)',
    'CREATE TABLE "concert" (\n"concert_ID" int,\n"Year" text,\n"Singer_ID" int,\n'
    'PRIMARY KEY ("concert_ID"),\nFOREIGN KEY ("Singer_ID") REFERENCES `singer`("Singer_ID")\n)',
]
SQL = (
    "SELECT count(*) FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id "
    "WHERE T1.country = 'France' AND T2.year = '2014'"
)
EX = {"source": "spider", "db_id": "d", "schema": SCHEMA, "question": "How many?", "sql": SQL}


def test_identifiers_skip_constraints():
    tables, columns = identifiers(SCHEMA)
    assert tables == ["singer", "concert"]
    assert columns == ["Singer_ID", "Name", "Country", "Age", "concert_ID", "Year", "Singer_ID"]


def test_augmented_examples_still_run():
    changed = 0
    for seed in range(200):
        out = augment(EX, random.Random(seed))
        if out is None:
            continue
        changed += out != EX
        conn = sqlite3.connect(":memory:")
        for stmt in out["schema"]:
            conn.execute(stmt)
        conn.execute(out["sql"]).fetchall()
        conn.close()
        assert "'France'" in out["sql"] and "'2014'" in out["sql"] and "count(*)" in out["sql"]
    assert changed > 100


def test_string_literals_are_left_alone():
    ex = {**EX, "sql": "SELECT name FROM singer WHERE country = 'singer'"}
    for seed in range(50):
        out = augment(ex, random.Random(seed))
        if out:
            assert out["sql"].endswith("= 'singer'")


def test_unsupported_schema_is_skipped():
    ex = {**EX, "schema": ["CREATE VIEW v AS SELECT 1"]}
    assert augment(ex, random.Random(0)) is None


def test_augment_rows_keeps_original_and_adds_variants():
    cfg = SimpleNamespace(seed=0, aug_copies=3, aug_other=1.0)
    synth = {
        "source": "synth",
        "schema": ["CREATE TABLE people (id INTEGER, name TEXT)"],
        "question": "q",
        "sql": "SELECT name FROM people",
    }
    out = augment_rows([EX, synth], cfg)
    assert out[0] == EX
    assert len(out) >= 2
    assert out[-1]["source"] == "synth"
