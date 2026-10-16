import itertools
import os

from datasets import load_dataset

OUT_DIR = "/content/drive/MyDrive/querion/tokenizer"
OUT = f"{OUT_DIR}/corpus.txt"
ENGLISH_DOCS = 30_000


def write_pairs(f, name, schema_col, question_col, sql_col):
    ds = load_dataset(name, split="train")
    for row in ds:
        f.write(f"{row[schema_col]}\n{row[question_col]}\n{row[sql_col]}\n\n")
    return len(ds)


os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    n1 = write_pairs(f, "b-mc2/sql-create-context", "context", "question", "answer")
    print("sql-create-context rows:", n1)
    n2 = write_pairs(f, "gretelai/synthetic_text_to_sql", "sql_context", "sql_prompt", "sql")
    print("synthetic_text_to_sql rows:", n2)
    eng = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    n3 = 0
    for row in itertools.islice(eng, ENGLISH_DOCS):
        f.write(row["text"].strip() + "\n\n")
        n3 += 1
    print("english docs:", n3)

print("corpus size MB:", round(os.path.getsize(OUT) / 1e6, 1), flush=True)
os._exit(0)
