from data.clean import reject_reason

GOOD = "CREATE TABLE users (\n  id INT PRIMARY KEY,\n  name VARCHAR(50)\n);\nSELECT * FROM users;"


def test_good_sql_passes():
    assert reject_reason(GOOD) is None


def test_nul_byte():
    assert reject_reason(GOOD + "\x00") == "nul_byte"


def test_data_dump():
    dump = "\n".join(f"INSERT INTO t VALUES ({i}, 'abc');" for i in range(50))
    assert reject_reason(dump) == "data_dump"


def test_long_line():
    assert reject_reason("SELECT " + "a," * 1000 + "b FROM t") == "long_line"


def test_non_english():
    assert reject_reason("\u4f60\u597d\u4e16\u754c" * 20 + "\nSELECT 1;") == "non_english"
