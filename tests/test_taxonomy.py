from eval.taxonomy import classify_failure, classify_mismatch


def test_each_mismatch_category():
    cases = [
        ("SELECT name FROM singer", "SELECT name FROM concert", "wrong tables or joins"),
        (
            "SELECT name FROM singer WHERE age > 30",
            "SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)",
            "wrong query shape (set operation or subquery)",
        ),
        ("SELECT min(age) FROM singer", "SELECT max(age) FROM singer", "wrong aggregate"),
        (
            "SELECT name FROM singer ORDER BY age",
            "SELECT name FROM singer ORDER BY age DESC",
            "wrong grouping, ordering or limit",
        ),
        (
            "SELECT name FROM singer WHERE country = 'Spain'",
            "SELECT name FROM singer WHERE country = 'France'",
            "wrong filter",
        ),
        ("SELECT age FROM singer", "SELECT name FROM singer", "wrong selected columns"),
        (
            "SELECT name FROM singer WHERE age >= 30",
            "SELECT name FROM singer WHERE age > 30",
            "same structure, different result",
        ),
    ]
    for pred, gold, expected in cases:
        assert classify_mismatch(pred, gold) == expected, (pred, gold)


def test_quotes_and_aliases_do_not_count_as_differences():
    gold = "SELECT T1.name FROM singer AS T1 WHERE T1.country = 'France'"
    pred = 'SELECT t2.name FROM singer AS t2 WHERE t2.country = "France"'
    assert classify_mismatch(pred, gold) == "same structure, different result"


def test_failure_labels_for_errors():
    syntax = {"status": "pred_error", "error": 'OperationalError: near "AS": syntax error'}
    assert classify_failure(syntax) == "syntax error"
    column = {"status": "pred_error", "error": "OperationalError: no such column: x"}
    assert classify_failure(column, "wrong alias") == "unknown column: wrong alias"
    assert classify_failure({"status": "prompt_too_long"}) == "prompt too long"
