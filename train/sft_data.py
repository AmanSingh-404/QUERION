import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch

from data.leakage import norm
from data.prompt import EOS, clean_sql, format_prompt, statements_from_gretel, statements_from_lines
from data.synth import generate
from eval.executor import execute
from eval.spider import db_file, schema_statements
from tokenizer.normalize import normalize
from tokenizer.special_tokens import PAD_ID


def fix_commas(sql):
    """Spider writes 'a ,  b'; use the usual 'a, b'."""
    return re.sub(r"\s+,\s*", ", ", sql)


def spider_examples(root, json_file="train_spider.json"):
    with open(Path(root) / json_file, encoding="utf-8") as f:
        rows = json.load(f)
    out, skipped, schemas = [], 0, {}
    for r in rows:
        db = db_file(root, r["db_id"])
        if not db.exists():
            skipped += 1
            continue
        if r["db_id"] not in schemas:
            schemas[r["db_id"]] = schema_statements(db)
        sql = fix_commas(clean_sql(r["query"]))
        if execute(db, sql)[1]:
            skipped += 1
            continue
        out.append(
            {
                "source": "spider",
                "db_id": r["db_id"],
                "schema": schemas[r["db_id"]],
                "question": normalize(r["question"]),
                "sql": sql,
            }
        )
    return out, skipped


def gretel_examples(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            statements = statements_from_gretel(r["sql_context"])
            if statements:
                out.append(
                    {
                        "source": "gretel",
                        "schema": statements,
                        "question": normalize(r["sql_prompt"]),
                        "sql": clean_sql(r["sql"]),
                    }
                )
    return out


def synth_examples(n=40000, seed=0):
    return [
        {
            "source": "synth",
            "schema": statements_from_lines(r["schema"]),
            "question": r["question"],
            "sql": r["sql"],
        }
        for r in generate(n, seed=seed)
    ]


def drop_contaminated(examples, needles):
    kept, dropped = [], Counter()
    for ex in examples:
        if needles.get(norm(ex["question"])) == "question" or needles.get(norm(ex["sql"])) == "sql":
            dropped[ex["source"]] += 1
        else:
            kept.append(ex)
    return kept, dict(dropped)


def is_val_db(db_id):
    return int(hashlib.md5(db_id.encode()).hexdigest(), 16) % 10 == 0


def split_examples(examples, gretel_val=500, synth_val=300, seed=0):
    """Spider is split by database (so validation uses unseen schemas); others at random."""
    rng = random.Random(seed)
    wanted = {"gretel": gretel_val, "synth": synth_val}
    val_ids = {}
    for src, n in wanted.items():
        ids = [i for i, e in enumerate(examples) if e["source"] == src]
        val_ids[src] = set(rng.sample(ids, min(n, len(ids))))
    train, val = [], []
    for i, ex in enumerate(examples):
        if ex["source"] == "spider":
            is_val = is_val_db(ex["db_id"])
        else:
            is_val = i in val_ids[ex["source"]]
        (val if is_val else train).append(ex)
    return train, val


def encode_example(tok, ex, max_len=2048):
    """(inputs, targets) for one example; only the SQL tokens and EOS have real targets."""
    prompt = format_prompt(ex["schema"], ex["question"])
    prompt_ids = tok.encode(prompt).ids
    full_ids = tok.encode(prompt + clean_sql(ex["sql"]) + EOS).ids
    if len(full_ids) > max_len:
        return None
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokens changed when the answer was appended")
    return full_ids[:-1], [-100] * (len(prompt_ids) - 1) + full_ids[len(prompt_ids) :]


def collate(pairs, pad_id=PAD_ID):
    width = max(len(x) for x, _ in pairs)
    xs = torch.full((len(pairs), width), pad_id, dtype=torch.long)
    ys = torch.full((len(pairs), width), -100, dtype=torch.long)
    for i, (x, y) in enumerate(pairs):
        xs[i, : len(x)] = torch.tensor(x)
        ys[i, : len(y)] = torch.tensor(y)
    return xs, ys
