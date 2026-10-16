from tokenizer.normalize import normalize


def test_nfkc():
    assert normalize("\ufb01le \u00b2") == "file 2"


def test_whitespace_collapse():
    assert normalize("SELECT   a,\t b\r\nFROM  t") == "SELECT a, b\nFROM t"


def test_blank_lines_collapse():
    assert normalize("a\n\n\n\nb") == "a\n\nb"


def test_control_chars_removed():
    assert normalize("a\x00b\x1bc") == "abc"


def test_case_preserved():
    assert normalize("SELECT Name FROM Users") == "SELECT Name FROM Users"


def test_idempotent():
    s = "  CREATE TABLE t (a INT)  \n\n\n\nHow many?  "
    assert normalize(normalize(s)) == normalize(s)
