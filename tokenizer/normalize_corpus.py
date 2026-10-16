from tokenizer.normalize import normalize

SRC = "/content/drive/MyDrive/querion/tokenizer/corpus.txt"
DST = "/content/drive/MyDrive/querion/tokenizer/corpus_norm.txt"


def main():
    with open(SRC, encoding="utf-8") as f:
        raw = f.read()
    clean = normalize(raw)
    with open(DST, "w", encoding="utf-8") as f:
        f.write(clean + "\n")
    print("chars before:", len(raw), "after:", len(clean))


if __name__ == "__main__":
    main()
