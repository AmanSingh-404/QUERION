import sqlite3

from eval.spider import classify_error, db_file, evaluate, schema_statements


def make_db(root, db_id):
    path = db_file(root, db_id)
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE singer (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER
        );
        INSERT INTO singer VALUES (1, 'Ann', 30), (2, 'Bob', 40);
        """
    )
    conn.commit()
    conn.close()


def test_schema_statements_are_normalized(tmp_path):
    make_db(tmp_path, "music")
    stmts = schema_statements(db_file(tmp_path, "music"))
    assert len(stmts) == 1 and stmts[0].startswith("CREATE TABLE singer (")
    assert "    " not in stmts[0]


def test_classify_error():
    assert classify_error("OperationalError: no such column: x") == "unknown column"
    assert classify_error("OperationalError: no such table: t") == "unknown table"
    assert classify_error("OperationalError: ambiguous column name: id") == "ambiguous column"
    assert classify_error('OperationalError: near "FROM": syntax error') == "syntax error"
    assert classify_error("something else") == "other error"


def test_evaluate_counts_every_outcome(tmp_path):
    make_db(tmp_path, "music")
    gold = "SELECT name FROM singer WHERE age > 35"
    items = [
        {"db_id": "music", "question": "q1", "sql": gold},
        {"db_id": "music", "question": "q2", "sql": gold},
        {"db_id": "music", "question": "q3", "sql": gold},
        {"db_id": "music", "question": "q4", "sql": gold},
    ]
    answers = {
        "q1": "SELECT name FROM singer WHERE age >= 40",
        "q2": "SELECT name FROM singer",
        "q3": "SELECT nope FROM singer",
        "q4": None,
    }
    summary, records = evaluate(items, tmp_path, lambda stmts, q: answers[q])
    assert summary["items"] == 4 and summary["exec_accuracy"] == 0.25
    assert summary["status"] == {
        "match": 1,
        "mismatch": 1,
        "pred_error": 1,
        "prompt_too_long": 1,
    }
    assert summary["error_types"] == {"unknown column": 1}
    assert summary["exec_accuracy_nonempty_gold"] == 0.25 and summary["nonempty_gold_items"] == 4
    assert [r["match"] for r in records] == [True, False, False, False]


def test_empty_gold_results_are_excluded_from_the_strict_metric(tmp_path):
    make_db(tmp_path, "music")
    items = [
        {"db_id": "music", "question": "empty", "sql": "SELECT name FROM singer WHERE age > 99"},
        {"db_id": "music", "question": "real", "sql": "SELECT name FROM singer WHERE age = 30"},
    ]
    answers = {"empty": "SELECT name FROM singer WHERE age < 0", "real": "SELECT name FROM singer"}
    summary, _ = evaluate(items, tmp_path, lambda stmts, q: answers[q])
    assert summary["exec_accuracy"] == 0.5
    assert summary["exec_accuracy_nonempty_gold"] == 0.0 and summary["nonempty_gold_items"] == 1
