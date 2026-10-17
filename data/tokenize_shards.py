import glob
import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from data.prompt import BOS, EOS, format_example, statements_from_gretel, statements_from_lines
from tokenizer.normalize import normalize
from tokenizer.special_tokens import SPECIAL_TOKENS

ROOT = "/content/drive/MyDrive/querion/data"
TOK = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"
SHARD_TOKENS = 100_000_000
VAL_EVERY = 200
BATCH = 512
PAIR_MAX_LEN = 2048


def has_special(text):
    return any(t in text for t in SPECIAL_TOKENS)


def batches(it, size):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def save_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def write_shard(out_dir, k, parts):
    arr = np.concatenate(parts)
    tmp = f"{out_dir}/train-{k:04d}.bin.tmp"
    arr.tofile(tmp)
    os.replace(tmp, f"{out_dir}/train-{k:04d}.bin")


def encode_all(
    tok,
    out_dir,
    factory,
    max_train_tokens=None,
    max_len=None,
    shard_tokens=SHARD_TOKENS,
    batch_size=BATCH,
):
    """Tokenize documents into uint16 shards. Every VAL_EVERY-th document goes to val.bin.

    factory(skip) yields documents (None means filtered), skipping the first `skip`.
    State is saved after every finished shard, so a crashed run resumes where it left off.
    """
    os.makedirs(out_dir, exist_ok=True)
    state_path = f"{out_dir}/state.json"
    val_path = f"{out_dir}/val.bin"
    state = {
        "raw_docs": 0,
        "docs_kept": 0,
        "train_shards": 0,
        "train_tokens": 0,
        "val_bytes": 0,
        "drops": {},
        "done": False,
    }
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        if state["done"]:
            print("already done:", state)
            return state
        print("resuming after", state["raw_docs"], "documents")
    with open(val_path, "r+b" if os.path.exists(val_path) else "w+b") as f:
        f.truncate(state["val_bytes"])

    drops = Counter(state["drops"])
    pending, pending_n = [], 0
    consumed = state["raw_docs"]
    val_f = open(val_path, "ab")
    for chunk in batches(factory(consumed), batch_size):
        consumed += len(chunk)
        texts = [t for t in chunk if t is not None]
        drops["filtered_in_source"] += len(chunk) - len(texts)
        for enc in tok.encode_batch(texts) if texts else []:
            arr = np.asarray(enc.ids, dtype=np.uint16)
            if max_len and len(arr) > max_len:
                drops["too_long"] += 1
                continue
            if state["docs_kept"] % VAL_EVERY == 0:
                val_f.write(arr.tobytes())
                state["val_bytes"] += arr.nbytes
            else:
                pending.append(arr)
                pending_n += len(arr)
                state["train_tokens"] += len(arr)
            state["docs_kept"] += 1
        enough = bool(max_train_tokens) and state["train_tokens"] >= max_train_tokens
        if pending_n >= shard_tokens or enough:
            val_f.flush()
            write_shard(out_dir, state["train_shards"], pending)
            state["train_shards"] += 1
            pending, pending_n = [], 0
            state["raw_docs"] = consumed
            state["drops"] = dict(drops)
            save_json(state_path, state)
        if enough:
            break
    if pending:
        write_shard(out_dir, state["train_shards"], pending)
        state["train_shards"] += 1
    val_f.close()
    state["raw_docs"] = consumed
    state["drops"] = dict(drops)
    state["done"] = True
    save_json(state_path, state)
    return state


def stack_factory(skip):
    def gen():
        for path in sorted(glob.glob(f"{ROOT}/dedup/stack_sql/part-*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    text = json.loads(line)["text"]
                    yield None if has_special(text) else BOS + text + EOS

    return itertools.islice(gen(), skip, None)


def pairs_factory(skip):
    def gen():
        with open(f"{ROOT}/verified/gretel_train.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                statements = statements_from_gretel(r["sql_context"])
                texts = [r["sql_prompt"], r["sql"], *statements]
                if not statements or any(has_special(t) for t in texts):
                    yield None
                else:
                    yield format_example(statements, r["sql_prompt"], r["sql"])
        with open(f"{ROOT}/synth.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                yield format_example(statements_from_lines(r["schema"]), r["question"], r["sql"])

    return itertools.islice(gen(), skip, None)


def fineweb_factory(skip):
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    if skip:
        ds = ds.skip(skip)
    for row in ds:
        text = normalize(row["text"])
        yield None if len(text) < 200 or has_special(text) else BOS + text + EOS


def main():
    source = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else None
    factories = {"stack": stack_factory, "pairs": pairs_factory, "fineweb": fineweb_factory}
    max_len = PAIR_MAX_LEN if source == "pairs" else None
    tok = Tokenizer.from_file(str(TOK))
    out_dir = f"{ROOT}/shards/{source}"
    state = encode_all(tok, out_dir, factories[source], max_tokens, max_len)
    summary = {
        "source": source,
        "docs_kept": state["docs_kept"],
        "train_tokens": state["train_tokens"],
        "val_tokens": state["val_bytes"] // 2,
        "train_shards": state["train_shards"],
        "drops": state["drops"],
    }
    save_json(f"{out_dir}/summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
