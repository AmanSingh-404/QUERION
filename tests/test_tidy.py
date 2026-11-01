from eval.tidy import tidy_literals


def test_trims_spaces_inside_quotes():
    sql = 'SELECT a FROM t WHERE b = " Ward"'
    assert tidy_literals(sql) == 'SELECT a FROM t WHERE b = "Ward"'
    sql = "SELECT a FROM t WHERE c = ' JetBlue Airways '"
    assert tidy_literals(sql) == "SELECT a FROM t WHERE c = 'JetBlue Airways'"


def test_leaves_everything_else_alone():
    inner = "SELECT a FROM t WHERE name = 'New York' AND b > 3"
    assert tidy_literals(inner) == inner
    plain = "SELECT a FROM t"
    assert tidy_literals(plain) == plain
