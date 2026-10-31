import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from data.prompt import format_prompt
from eval.spider import evaluate
from infer.constrain import SqlConstraint, token_texts
from infer.engine import generate_cached
from model.config import QUERION_100M
from model.transformer import Transformer
from tokenizer.special_tokens import EOS_ID
from train.train import TOK_PATH

DEV = Path(__file__).resolve().parents[1] / "eval" / "spider_dev.jsonl"
MAX_PROMPT = 1800
MAX_NEW = 200


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "C:/querion-runs/main/latest.pt"
    root = sys.argv[2] if len(sys.argv) > 2 else "C:/querion-data/spider_data"
    out_name = sys.argv[3] if len(sys.argv) > 3 else "pretrained"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    extra = sys.argv[5:]
    unquote = "unquote" in extra
    constrain = "constrain" in extra
    dev_path = next((a for a in extra if a.endswith(".jsonl")), DEV)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = Transformer(QUERION_100M)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    tok = Tokenizer.from_file(str(TOK_PATH))
    texts = token_texts(tok)

    def predict(statements, question):
        if unquote:
            statements = [s.replace('"', "").replace("`", "") for s in statements]
        ids = tok.encode(format_prompt(statements, question)).ids
        if len(ids) > MAX_PROMPT:
            return None
        constraint = SqlConstraint("\n".join(statements), texts, EOS_ID) if constrain else None
        generated = generate_cached(model, ids, MAX_NEW, EOS_ID, device, constraint=constraint)
        out = generated[len(ids) :]
        if out and out[-1] == EOS_ID:
            out = out[:-1]
        return tok.decode(out, skip_special_tokens=False).strip().rstrip(";")

    with open(dev_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if limit:
        items = items[:: max(1, len(items) // limit)][:limit]
    start = time.time()
    summary, records = evaluate(items, root, predict)
    summary["seconds"] = round(time.time() - start)
    summary["checkpoint_step"] = state["step"]
    out_dir = Path("C:/querion-runs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"spider_dev_{out_name}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
