import difflib
import re
import sqlite3
from pathlib import Path

from eval.executor import execute


def schema_names(db_path):
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        query = "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        tables = [r[0] for r in conn.execute(query)]
        columns = set()
        for t in tables:
            columns.update(r[1] for r in conn.execute(f'PRAGMA table_info("{t}")'))
        return tables, sorted(columns)
    finally:
        conn.close()


def _key(name):
    return name.lower().replace("_", "")


def closest(name, options):
    """Nearest real name, comparing case-insensitively and ignoring underscores."""
    by_key = {_key(o): o for o in options}
    match = difflib.get_close_matches(_key(name), list(by_key), n=1, cutoff=0.6)
    return by_key[match[0]] if match else None


def repair(db_path, sql, max_tries=3):
    """Fix glued keywords, then swap unknown table/column names for the closest real ones."""
    sql = re.sub(r"\b(FROM|JOIN)(?=[A-Za-z_])", r"\1 ", sql)
    tables, columns = schema_names(db_path)
    for _ in range(max_tries):
        _, err = execute(db_path, sql)
        if err is None:
            return sql
        m = re.search(r"no such (table|column): (?:\w+\.)?(\w+)", err)
        if not m:
            return sql
        kind, bad = m.groups()
        good = closest(bad, tables if kind == "table" else columns)
        if good is None or good.lower() == bad.lower():
            return sql
        sql = re.sub(rf"\b{re.escape(bad)}\b", good, sql, flags=re.IGNORECASE)
    return sql
