from data.dedup import shingles


def test_short_text_single_shingle():
    assert shingles("select a from t") == {"select a from t"}


def test_shingle_count():
    assert len(shingles("a b c d e f g")) == 3
