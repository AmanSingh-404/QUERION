import sqlite3

import pytest

from eval.compare import exec_match, has_order_by, norm_cell, same_result
from eval.executor import execute


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE emp (id INTEGER, name TEXT, dept TEXT, salary REAL);
        INSERT INTO emp VALUES (1, 'Ann', 'A', 10.0), (2, 'Bob', 'A', 20.0),
                               (3, 'Cy', 'B', 30.0), (4, 'Di', 'B', 30.0);
        """
    )
    conn.commit()
    conn.close()
    return path


def test_execute_returns_rows_and_errors(db):
    rows, err = execute(db, "SELECT name FROM emp WHERE id = 1")
    assert rows == [("Ann",)] and err is None
    rows, err = execute(db, "SELECT nope FROM emp")
    assert rows is None and "OperationalError" in err


def test_database_is_read_only(db):
    rows, err = execute(db, "DELETE FROM emp")
    assert rows is None and err is not None
    assert execute(db, "SELECT COUNT(*) FROM emp")[0] == [(4,)]


def test_timeout_interrupts_runaway_queries(db):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    rows, err = execute(db, sql, timeout=0.3)
    assert rows is None and err is not None


def test_row_limit(db):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 50) SELECT x FROM c"
    assert execute(db, sql, max_rows=10) == (None, "too many rows")


def test_norm_cell_and_order_by_detection():
    assert norm_cell(3.0) == 3 and norm_cell(2.0000001) == 2.0 and norm_cell(b"x") == "x"
    assert has_order_by("select a from t ORDER  BY a") and not has_order_by("select border from t")


def test_same_result_ignores_row_and_column_order_unless_ordered():
    a = [(1, "x"), (2, "y")]
    assert same_result([(2, "y"), (1, "x")], a, ordered=False)
    assert not same_result([(2, "y"), (1, "x")], a, ordered=True)
    assert same_result([("x", 1), ("y", 2)], a, ordered=False)
    assert same_result([(1.0, "x"), (2.0, "y")], a, ordered=False)


def test_same_result_counts_duplicates_and_sizes():
    assert not same_result([(1,), (1,)], [(1,), (2,)], ordered=False)
    assert not same_result([(1,)], [(1,), (1,)], ordered=False)
    assert same_result([], [], ordered=False)
    assert not same_result([(1, 2)], [(1,)], ordered=False)


def test_exec_match_statuses(db):
    gold = "SELECT name FROM emp WHERE dept = 'A'"
    assert exec_match(db, "SELECT name FROM emp WHERE id < 3", gold)["status"] == "match"
    assert exec_match(db, "SELECT name FROM emp", gold)["status"] == "mismatch"
    assert exec_match(db, "SELECT bad FROM emp", gold)["status"] == "pred_error"
    assert exec_match(db, gold, "SELECT bad FROM emp")["status"] == "gold_error"


def test_ordered_gold_requires_matching_order(db):
    gold = "SELECT name FROM emp ORDER BY salary DESC, name"
    assert exec_match(db, "SELECT name FROM emp ORDER BY salary DESC, name", gold)["match"]
    assert not exec_match(db, "SELECT name FROM emp ORDER BY salary, name", gold)["match"]
