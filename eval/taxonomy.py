import re

from eval.spider import classify_error

AGG = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)
TABLE = re.compile(r"\b(?:from|join)\s+(\w+)", re.IGNORECASE)
LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"|\b\d+(?:\.\d+)?\b")
SELECT_LIST = re.compile(r"\bselect\b(.*?)\bfrom\b", re.IGNORECASE | re.DOTALL)
WORD = re.compile(r"[A-Za-z_]\w*")
NOT_COLUMNS = {"select", "distinct", "as", "all", "count", "sum", "avg", "min", "max"}


def structure(sql):
    """A rough structural summary of a query (not a parser)."""
    text = " ".join(sql.lower().split())
    aliases = set(re.findall(r"\bas\s+(\w+)", text))
    found = SELECT_LIST.search(text)
    words = set(WORD.findall(found.group(1))) if found else set()
    return {
        "tables": set(TABLE.findall(text)),
        "joins": len(re.findall(r"\bjoin\b", text)),
        "aggs": {m.lower() for m in AGG.findall(text)},
        "group": "group by" in text,
        "having": re.search(r"\bhaving\b", text) is not None,
        "order": "order by" in text,
        "desc": re.search(r"\bdesc\b", text) is not None,
        "limit": re.search(r"\blimit\b", text) is not None,
        "where": re.search(r"\bwhere\b", text) is not None,
        "setop": sorted(set(re.findall(r"\b(union|intersect|except)\b", text))),
        "subqueries": text.count("select") - 1,
        "literals": {lit.strip("'\"").lower() for lit in LITERAL.findall(text)},
        "columns": words - NOT_COLUMNS - aliases,
    }


def classify_mismatch(pred, gold):
    """Why a query that runs still gives a different answer than the gold query."""
    p, g = structure(pred), structure(gold)
    if p["tables"] != g["tables"] or p["joins"] != g["joins"]:
        return "wrong tables or joins"
    if p["setop"] != g["setop"] or p["subqueries"] != g["subqueries"]:
        return "wrong query shape (set operation or subquery)"
    if p["aggs"] != g["aggs"]:
        return "wrong aggregate"
    if any(p[k] != g[k] for k in ("group", "having", "order", "desc", "limit")):
        return "wrong grouping, ordering or limit"
    if p["where"] != g["where"] or p["literals"] != g["literals"]:
        return "wrong filter"
    if p["columns"] != g["columns"]:
        return "wrong selected columns"
    return "same structure, different result"


def classify_failure(record, column_kind=None):
    """Label one failed record. column_kind says what kind of unknown column it was."""
    status = record["status"]
    if status == "prompt_too_long":
        return "prompt too long"
    if status == "mismatch":
        return classify_mismatch(record["pred"], record["sql"])
    kind = classify_error(record["error"])
    if kind == "unknown column" and column_kind:
        return f"unknown column: {column_kind}"
    return kind
