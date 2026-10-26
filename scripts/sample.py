import sys

import torch
from tokenizers import Tokenizer

from data.prompt import format_prompt
from model.config import QUERION_100M
from model.transformer import Transformer
from tokenizer.special_tokens import EOS_ID
from train.sample import generate
from train.train import TOK_PATH

EMPLOYEES = "CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER)"
SINGERS = "CREATE TABLE singer (singer_id INTEGER, name TEXT, country TEXT, age INTEGER)"
ORDERS = "CREATE TABLE orders (id INTEGER, customer TEXT, total REAL, order_year INTEGER)"
PROMPTS = [
    format_prompt([EMPLOYEES], "What is the average salary of employees in each department?"),
    format_prompt([EMPLOYEES], "How many employees earn more than 50000?"),
    format_prompt([SINGERS], "List the names of singers from France, oldest first."),
    format_prompt([ORDERS], "What is the total value of orders placed in 2023?"),
    "<|bos|>SELECT name FROM",
    "<|bos|>The capital of France is",
]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "C:/querion-runs/main/latest.pt"
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = Transformer(QUERION_100M)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    tok = Tokenizer.from_file(str(TOK_PATH))
    print(f"checkpoint step {state['step']}")
    for prompt in PROMPTS:
        ids = generate(model, tok.encode(prompt).ids, max_new, EOS_ID, device)
        print(tok.decode(ids, skip_special_tokens=False), "\n-----", flush=True)


if __name__ == "__main__":
    main()
