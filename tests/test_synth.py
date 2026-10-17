import sqlite3

from data.synth import generate


def test_deterministic():
    assert generate(200, seed=1) == generate(200, seed=1)


def test_count_and_unique():
    rows = generate(300, seed=0)
    assert len(rows) == 300
    assert len({(r["schema"], r["question"], r["sql"]) for r in rows}) == 300


def test_sql_is_valid_and_nonempty_on_its_schema():
    for r in generate(50, seed=2):
        conn = sqlite3.connect(":memory:")
        for stmt in r["schema"].split("\n"):
            conn.execute(stmt)
        conn.execute(r["sql"])
        assert r["sql"].startswith("SELECT")
        assert "{" not in r["question"]
