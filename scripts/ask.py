import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

from data.prompt import format_prompt
from eval.executor import execute
from eval.spider import schema_statements
from infer.constrain import SqlConstraint, token_texts
from infer.engine import generate_batch
from infer.select import choose, signature
from model.config import QUERION_100M
from model.transformer import Transformer
from tokenizer.special_tokens import EOS_ID

MAX_PROMPT = 1800
MAX_NEW = 200


def load(weights, device):
    """weights: a folder with model.safetensors and tokenizer.json, or a Hugging Face repo id."""
    folder = Path(weights)
    if not folder.is_dir():
        folder = Path(snapshot_download(weights))
    model = Transformer(QUERION_100M)
    result = model.load_state_dict(load_file(folder / "model.safetensors"), strict=False)
    if result.missing_keys != ["lm_head.weight"] or result.unexpected_keys:
        raise SystemExit(f"unexpected weights file: {result}")
    tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
    return model.to(device).eval(), tok


def main():
    p = argparse.ArgumentParser(description="Ask a question about a SQLite database.")
    p.add_argument("db", help="path to a SQLite file")
    p.add_argument("question")
    p.add_argument("--weights", default="SinghBro/querion-100m-sql")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(args.weights, device)
    statements = schema_statements(args.db)
    ids = tok.encode(format_prompt(statements, args.question)).ids
    if len(ids) > MAX_PROMPT:
        raise SystemExit(f"schema too long: {len(ids)} tokens (limit {MAX_PROMPT})")
    texts = token_texts(tok)
    schema_text = "\n".join(statements)
    cons = [SqlConstraint(schema_text, texts, EOS_ID) for _ in range(args.samples)]
    outs = generate_batch(
        model,
        ids,
        args.samples,
        MAX_NEW,
        EOS_ID,
        device,
        constraints=cons,
        temperature=args.temperature,
    )
    candidates = [tok.decode(o, skip_special_tokens=False).strip().rstrip(";") for o in outs]
    sql = choose(args.db, candidates)
    rows, err = execute(args.db, sql)
    print(sql)
    if err:
        print("error:", err)
        return
    sig = signature(rows)
    agree = 0
    for c in candidates:
        r, e = execute(args.db, c)
        agree += e is None and signature(r) == sig
    print(f"\n{agree} of {len(candidates)} candidates returned the same result")
    for row in rows[:20]:
        print(row)
    if len(rows) > 20:
        print(f"... {len(rows)} rows in total")


if __name__ == "__main__":
    main()
