import json

from data.leakage import build_needles, doc_hits, norm, scan_pairs

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


def test_scan_pairs_returns_matching_ids(tmp_path):
    rows = [
        {"question": "How many singers do we have in total?", "sql": "SELECT 1"},
        {"question": "Something else entirely, no match here", "sql": "SELECT 2"},
        {"question": "how  many singers do we have in total?", "sql": "SELECT 3"},
    ]
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out, bad = scan_pairs(str(path), "question", "sql", build_needles(ROWS))
    assert bad == [0, 2]
    assert out["question_matches"] == 2
