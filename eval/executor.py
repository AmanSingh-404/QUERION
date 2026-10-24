import sqlite3
import time
from pathlib import Path


def execute(db_path, sql, timeout=5.0, max_rows=10000):
    """Run one query against a SQLite file opened read-only. Returns (rows, error)."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
    try:
        deadline = time.time() + timeout
        conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100000)
        rows = conn.execute(sql).fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return None, "too many rows"
        return rows, None
    except sqlite3.Error as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        conn.close()
