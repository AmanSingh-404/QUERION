import re

WORD_CHAR = re.compile(r"[A-Za-z0-9_]")
T_ALIAS_PREFIX = re.compile(r"t\d*")
T_ALIAS = re.compile(r"t\d+")
SPECIAL = re.compile(r"<\|.*\|>")

SQL_WORDS = (
    "select from where group by order having limit offset distinct all as on join inner left "
    "right outer cross natural using and or not in is null like glob between exists case when "
    "then else end union intersect except asc desc with recursive values if collate nocase "
    "escape true false count sum avg min max total group_concat length abs round lower upper "
    "substr substring replace trim ltrim rtrim coalesce ifnull nullif iif cast strftime date "
    "datetime time julianday random instr typeof current_date current_time current_timestamp "
    "integer int text real numeric varchar char float double boolean"
).split()


def token_texts(tok):
    """Text of every token id; special tokens become None."""
    out = []
    for i in range(tok.get_vocab_size()):
        text = tok.decode([i], skip_special_tokens=False)
        out.append(None if SPECIAL.fullmatch(text) else text)
    return out


class SqlConstraint:
    """Keeps every word of the generated SQL inside a known set: SQL words, the schema's
    own words, T1-style aliases and aliases the query defines itself with AS. Text inside
    quotes is unconstrained."""

    def __init__(self, schema_text, texts, eos_id):
        words = {w.lower() for w in SQL_WORDS}
        words |= {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", schema_text)}
        self.words = words
        self.prefixes = {w[:i] for w in words for i in range(1, len(w) + 1)}
        self.texts = texts
        self.eos_id = eos_id
        self.reset()

    def reset(self):
        # (current word, open quote, next word is an alias, in an alias, defined aliases)
        self.state = ("", None, False, False, frozenset())

    def _prefix_ok(self, buf, alias_def, aliases):
        if alias_def or buf.isdigit():
            return True
        return (
            buf in self.prefixes
            or T_ALIAS_PREFIX.fullmatch(buf) is not None
            or any(a.startswith(buf) for a in aliases)
        )

    def _complete_ok(self, buf, alias_def, aliases):
        return (
            alias_def
            or buf.isdigit()
            or buf in self.words
            or T_ALIAS.fullmatch(buf) is not None
            or buf in aliases
        )

    def _run(self, text):
        buf, quote, expect_alias, alias_def, aliases = self.state
        for ch in text:
            if quote:
                if ch == quote:
                    quote = None
                continue
            if WORD_CHAR.match(ch):
                if not buf:
                    alias_def, expect_alias = expect_alias, False
                buf += ch.lower()
                if not self._prefix_ok(buf, alias_def, aliases):
                    return None
                continue
            if buf:
                if not self._complete_ok(buf, alias_def, aliases):
                    return None
                if alias_def:
                    aliases = aliases | {buf}
                expect_alias = buf == "as"
                buf, alias_def = "", False
            if ch in "'\"`":
                quote = ch
        return (buf, quote, expect_alias, alias_def, aliases)

    def accepts(self, cid):
        if cid == self.eos_id:
            buf, quote, _, alias_def, aliases = self.state
            return quote is None and (not buf or self._complete_ok(buf, alias_def, aliases))
        text = self.texts[cid]
        return text is not None and self._run(text) is not None

    def advance(self, cid):
        self.state = self._run(self.texts[cid])
