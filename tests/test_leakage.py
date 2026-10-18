from data.leakage import build_needles, doc_hits, norm

ROWS = [
    {
        "question": "How many singers do we have in total?",
        "sql": "SELECT count(*) FROM singer WHERE age > 30 AND country = 'France'",
    }
]


def test_norm():
    assert norm("  SELECT  a\nFROM t ; ") == "select a from t"


def test_short_strings_are_ignored():
    assert build_needles([{"question": "Too short?", "sql": "SELECT 1"}]) == {}


def test_finds_single_line_sql():
    text = "-- gold\nSELECT count(*) FROM singer WHERE age > 30 AND country = 'France'\nSELECT 2"
    assert doc_hits(text, build_needles(ROWS))


def test_finds_wrapped_statement():
    text = "SELECT count(*)\n  FROM singer\n  WHERE age > 30\n  AND country = 'France';\nSELECT 2;"
    assert doc_hits(text, build_needles(ROWS))


def test_finds_question_line():
    assert doc_hits("# How many singers do we have in total?", build_needles(ROWS)) == set()
    assert doc_hits("How many singers do we have in total?", build_needles(ROWS))


def test_clean_document_has_no_hits():
    assert not doc_hits("CREATE TABLE a (x INT);\nSELECT * FROM a;", build_needles(ROWS))
