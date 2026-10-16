import random
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tokenizer.special_tokens import SPECIAL_TOKENS, TOKEN_ID

PATH = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"
tok = Tokenizer.from_file(str(PATH))

RANGES = [(32, 126), (160, 2000), (0x900, 0x97F), (0x4E00, 0x4F00), (0x1F600, 0x1F64F)]

SAMPLES = [
    "SELECT COUNT(*) FROM head WHERE age > 56",
    "CREATE TABLE t (a INT, b VARCHAR)\nHow many rows?",
    "naive cafe - \u65e5\u672c\u8a9e \U0001f600",
    "  leading spaces and\ttabs\n\n\nnewlines",
    "\u0915\u0932 \u092e\u093f\u0932\u0924\u0947 \u0939\u0948\u0902",
]


def roundtrip(s):
    return tok.decode(tok.encode(s).ids, skip_special_tokens=False)


def test_vocab_size():
    assert tok.get_vocab_size() == 24000


def test_special_ids():
    for t in SPECIAL_TOKENS:
        assert tok.token_to_id(t) == TOKEN_ID[t]


@pytest.mark.parametrize("s", SAMPLES)
def test_roundtrip(s):
    assert roundtrip(s) == s


def test_random_unicode_roundtrip():
    rng = random.Random(0)
    for _ in range(300):
        s = ""
        for _ in range(rng.randint(1, 40)):
            lo, hi = rng.choice(RANGES)
            s += chr(rng.randint(lo, hi))
        assert roundtrip(s) == s


def test_special_token_is_single_id():
    assert tok.encode("<|sql|>").ids == [TOKEN_ID["<|sql|>"]]


def test_prompt_layout():
    s = (
        "<|bos|><|schema|>CREATE TABLE t (a INT)<|question|>How many?"
        "<|sql|>SELECT COUNT(*) FROM t<|eos|>"
    )
    ids = tok.encode(s).ids
    assert ids[0] == TOKEN_ID["<|bos|>"]
    assert ids[-1] == TOKEN_ID["<|eos|>"]
    assert roundtrip(s) == s


def test_ids_in_range():
    ids = tok.encode(SAMPLES[0] + SAMPLES[2]).ids
    assert all(0 <= i < 24000 for i in ids)
