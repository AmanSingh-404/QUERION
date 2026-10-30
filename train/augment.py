import random
import re

KEYWORDS = set(
    "select from where group by order having limit offset distinct all as on join inner left "
    "right outer cross natural using and or not in is null like between exists case when then "
    "else end union intersect except count sum avg min max asc desc create table view primary "
    "key foreign references unique check constraint default integer int text real number "
    "numeric varchar char date datetime time timestamp boolean float double blob values "
    "insert update delete into set index if cast".split()
)
SIMPLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NAME = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)'
CREATE = re.compile(
    rf"^\s*create\s+table\s+(?:if\s+not\s+exists\s+)?({NAME})\s*\((.*)\)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
LITERAL = re.compile(r"('(?:[^']|'')*'|\"[^\"]*\")")
CONSTRAINTS = {"primary", "foreign", "unique", "check", "constraint", "key"}
STYLES = ["keep", "keep", "snake", "camel", "pascal", "upper", "flat", "title"]


def _bare(token):
    return token[1:-1] if token[0] in '"`[' else token


def split_top(body):
    """Split a column list on commas that are not inside parentheses or quotes."""
    parts, cur, depth, quote = [], [], 0, None
    for ch in body:
        cur.append(ch)
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            cur.pop()
            parts.append("".join(cur))
            cur = []
    parts.append("".join(cur))
    return parts


def identifiers(statements):
    """Table and column names from CREATE TABLE statements; None if any is unsupported."""
    tables, columns = [], []
    for stmt in statements:
        m = CREATE.match(stmt)
        if not m:
            return None
        tables.append(_bare(m.group(1)))
        for item in split_top(m.group(2)):
            first = re.match(NAME, item.strip())
            if not first:
                continue
            token = first.group(0)
            if token[0] not in '"`[' and token.lower() in CONSTRAINTS:
                continue
            columns.append(_bare(token))
    return tables, columns


def words(name):
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return [w.lower() for w in spaced.split("_") if w]


def render(ws, style):
    if style == "snake":
        return "_".join(ws)
    if style == "camel":
        return ws[0] + "".join(w.title() for w in ws[1:])
    if style == "pascal":
        return "".join(w.title() for w in ws)
    if style == "upper":
        return "_".join(ws).upper()
    if style == "flat":
        return "".join(ws)
    return "_".join(w.title() for w in ws)


def flip_plural(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ss"):
        return w + "es"
    if w.endswith("s"):
        return w[:-1]
    if w.endswith("y") and w[-2:-1] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def flip_name(name):
    m = re.match(r"^(.*?)([A-Za-z]+)$", name)
    if not m:
        return name
    head, tail = m.groups()
    flipped = flip_plural(tail.lower())
    if tail.isupper():
        flipped = flipped.upper()
    elif tail[0].isupper():
        flipped = flipped.capitalize()
    return head + flipped


def make_mapping(tables, columns, rng):
    """lowercase original name -> new name; None if the renaming would collide."""
    style = rng.choice(STYLES)
    mapping, fixed = {}, set()
    for kind, names in (("t", tables), ("c", columns)):
        for name in names:
            key = name.lower()
            if key in mapping or key in fixed:
                continue
            if not SIMPLE.match(name) or key in KEYWORDS or re.fullmatch(r"t\d+", key):
                fixed.add(key)
                continue
            base = name
            if kind == "t" and len(name) > 3 and rng.random() < 0.3:
                base = flip_name(name)
            ws = words(base)
            if not ws:
                fixed.add(key)
                continue
            mapping[key] = base if style == "keep" else render(ws, style)
    news = [v.lower() for v in mapping.values()]
    if len(set(news)) != len(news) or fixed & set(news):
        return None
    if any(n in KEYWORDS or re.fullmatch(r"t\d+", n) for n in news):
        return None
    return mapping


def unquote_schema(statements):
    return [re.sub(r"([\"`])([A-Za-z_]\w*)\1", r"\2", s) for s in statements]


def augment(ex, rng):
    """A renamed copy of one example (schema and SQL stay consistent), or None if unsafe."""
    ids = identifiers(ex["schema"])
    if ids is None:
        return None
    mapping = make_mapping(*ids, rng)
    if not mapping:
        return None
    for lit in LITERAL.findall(ex["sql"]):
        if lit[0] == '"' and lit[1:-1].lower() in mapping:
            return None
    keys = sorted(map(re.escape, mapping), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(keys) + r")\b", re.IGNORECASE)

    def sub(m):
        return mapping[m.group(0).lower()]

    parts = LITERAL.split(ex["sql"])
    for i in range(0, len(parts), 2):
        parts[i] = pattern.sub(sub, parts[i])
    schema = [pattern.sub(sub, s) for s in ex["schema"]]
    if rng.random() < 0.5:
        schema = unquote_schema(schema)
    if rng.random() < 0.5:
        rng.shuffle(schema)
    return {**ex, "schema": schema, "sql": "".join(parts)}


def augment_rows(rows, cfg):
    """Spider: the original plus aug_copies variants. Others: renamed in place with aug_other."""
    rng = random.Random(cfg.seed)
    out, spider_new, other_new = [], 0, 0
    for ex in rows:
        if ex["source"] == "spider":
            out.append(ex)
            for _ in range(cfg.aug_copies):
                v = augment(ex, rng)
                if v is not None and v != ex:
                    out.append(v)
                    spider_new += 1
        elif rng.random() < cfg.aug_other:
            v = augment(ex, rng)
            out.append(v if v is not None else ex)
            other_new += v is not None
        else:
            out.append(ex)
    print(f"augmented: {spider_new} spider variants, {other_new} others renamed", flush=True)
    return out
