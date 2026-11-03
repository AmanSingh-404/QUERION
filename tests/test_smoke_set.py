import json
from pathlib import Path

from eval.executor import execute

SMOKE = Path(__file__).resolve().parent / "smoke"


def test_smoke_set_is_consistent():
    with open(SMOKE / "items.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    assert len(items) >= 20
    for it in items:
        rows, err = execute(SMOKE / it["db"], it["sql"])
        assert err is None and rows, it
