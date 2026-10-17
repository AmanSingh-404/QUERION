from data.verify_gretel import is_select, runs_ok

CTX = "CREATE TABLE t (a INT); INSERT INTO t VALUES (1);"


def test_valid_select_runs():
    assert runs_ok(CTX, "SELECT a FROM t") == "ok"


def test_unknown_column_fails():
    assert runs_ok(CTX, "SELECT b FROM t").startswith("error")


def test_ambiguous_column_fails():
    ctx = "CREATE TABLE a (id INT); CREATE TABLE b (id INT);"
    assert runs_ok(ctx, "SELECT id FROM a JOIN b ON a.id = b.id").startswith("error")


def test_bad_context_fails():
    assert runs_ok("CREATE TABL oops", "SELECT 1").startswith("error")


def test_infinite_query_is_interrupted():
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    assert runs_ok(CTX, sql, timeout=0.2).startswith("error")


def test_is_select():
    assert is_select("  SELECT 1")
    assert is_select("WITH x AS (SELECT 1) SELECT * FROM x")
    assert not is_select("UPDATE t SET a = 1")
    assert not is_select("INSERT INTO t VALUES (1)")
