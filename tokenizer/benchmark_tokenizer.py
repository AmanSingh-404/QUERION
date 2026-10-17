import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from tokenizer.normalize import normalize

N = 1000
OURS = Path(__file__).with_name("tokenizer.json")
OTHERS = {
    "gpt2": "gpt2",
    "llama2": "NousResearch/Llama-2-7b-hf",
    "llama3": "unsloth/Llama-3.2-1B",
}


def load_texts():
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="test")
    ds = ds.shuffle(seed=0).select(range(N))
    rows = [f"{r['sql_context']}\n{r['sql_prompt']}\n{r['sql']}" for r in ds]
    return [normalize(t) for t in rows]


def main():
    texts = load_texts()
    chars = sum(len(t) for t in texts)
    ours_tok = Tokenizer.from_file(str(OURS))
    ours = sum(len(ours_tok.encode(t).ids) for t in texts)
    result = {
        "examples": N,
        "chars": chars,
        "querion": {
            "vocab": ours_tok.get_vocab_size(),
            "tokens": ours,
            "chars_per_token": round(chars / ours, 2),
        },
    }
    for name, repo in OTHERS.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            n = sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
        except Exception as e:
            print("skipped", name, type(e).__name__)
            continue
        result[name] = {
            "vocab": len(tok),
            "tokens": n,
            "chars_per_token": round(chars / n, 2),
            "querion_saving_pct": round(100 * (1 - ours / n), 1),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
