import sqlite3

from scripts.check_spider import check


def make_db(root, db_id):
    d = root / "database" / db_id
    d.mkdir(parents=True)
    conn = sqlite3.connect(d / f"{db_id}.sqlite")
    conn.executescript("CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1);")
    conn.commit()
    conn.close()


def test_check_reports_missing_errors_and_empty(tmp_path):
    make_db(tmp_path, "one")
    rows = [
        {"db_id": "one", "question": "q1", "sql": "SELECT a FROM t"},
        {"db_id": "one", "question": "q2", "sql": "SELECT b FROM t"},
        {"db_id": "one", "question": "q3", "sql": "SELECT a FROM t WHERE a > 5"},
        {"db_id": "two", "question": "q4", "sql": "SELECT 1"},
    ]
    official = [{"db_id": r["db_id"], "question": r["question"]} for r in rows]
    official[2]["question"] = "different"
    report = check(rows, tmp_path, official)
    assert report["missing_databases"] == ["two"]
    assert report["gold_errors"] == 1 and report["gold_empty"] == 1
    assert report["dev_json_mismatches"] == 1
