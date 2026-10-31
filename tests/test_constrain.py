from pathlib import Path

import torch
from tokenizers import Tokenizer

from infer.constrain import SqlConstraint, token_texts
from infer.engine import generate_cached
from model.config import TINY
from model.transformer import Transformer
from tokenizer.special_tokens import EOS_ID

TOK = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"
TEXTS = ["SELECT", "FROM", "AS", "name", "nam", "e", "singer", "s", "T", "1", "5"]
TEXTS += [" ", ",", ".", "'", "x", "fake", None]
ID = {t: i for i, t in enumerate(TEXTS) if t is not None}
EOS = len(TEXTS) - 1
SCHEMA = "CREATE TABLE singer (name TEXT, age INT)"


def make():
    return SqlConstraint(SCHEMA, TEXTS, EOS)


def feed(c, *tokens):
    for t in tokens:
        assert c.accepts(ID[t]), t
        c.advance(ID[t])


def test_valid_query_is_accepted():
    c = make()
    feed(c, "SELECT", " ", "name", " ", "FROM", " ", "singer")
    assert c.accepts(EOS)


def test_glued_keyword_is_rejected():
    c = make()
    feed(c, "SELECT", " ", "name", " ", "FROM")
    assert not c.accepts(ID["singer"])
    assert c.accepts(ID[" "])


def test_plural_table_name_is_rejected():
    c = make()
    feed(c, "SELECT", " ", "name", " ", "FROM", " ", "singer")
    assert not c.accepts(ID["s"])
    assert c.accepts(ID[","])


def test_partial_word_cannot_end_the_query():
    c = make()
    feed(c, "SELECT", " ", "nam")
    assert not c.accepts(EOS)
    feed(c, "e")
    assert c.accepts(EOS)


def test_unknown_words_are_rejected():
    c = make()
    assert not c.accepts(ID["fake"])
    assert not c.accepts(ID["x"])


def test_alias_defined_with_as_is_accepted_afterwards():
    c = make()
    feed(c, "SELECT", " ", "name", " ", "FROM", " ", "singer", " ", "AS", " ", "x", ",", " ")
    assert c.accepts(ID["x"])


def test_t_aliases_are_always_allowed():
    c = make()
    feed(c, "T", "1", ".", "name")


def test_string_literals_are_unconstrained():
    c = make()
    feed(c, "'", "fake", " ")
    assert not c.accepts(EOS)
    feed(c, "'")
    assert c.accepts(EOS)


def test_numbers_are_allowed():
    c = make()
    feed(c, "5", " ")


def test_token_texts_keep_the_leading_space_and_hide_special_tokens():
    tok = Tokenizer.from_file(str(TOK))
    texts = token_texts(tok)
    assert texts[tok.token_to_id("Ġconcert")] == " concert"
    assert texts[EOS_ID] is None


class EvenOnly:
    eos_id = -1

    def reset(self):
        pass

    def accepts(self, cid):
        return cid % 2 == 0

    def advance(self, cid):
        pass


def test_constrained_generation_respects_the_constraint():
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    ids = torch.randint(0, TINY.vocab_size, (6,)).tolist()
    out = generate_cached(model, ids, 10, -1, "cpu", constraint=EvenOnly())
    assert all(t % 2 == 0 for t in out[len(ids) :])
