from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tokenizer.special_tokens import SPECIAL_TOKENS, TOKEN_ID

CORPUS = "/content/drive/MyDrive/querion/tokenizer/corpus_norm.txt"
OUT = "/content/drive/MyDrive/querion/tokenizer/tokenizer.json"
VOCAB_SIZE = 24_000


def main():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train([CORPUS], trainer)
    tok.save(OUT)

    print("vocab size:", tok.get_vocab_size())
    for t in SPECIAL_TOKENS:
        assert tok.token_to_id(t) == TOKEN_ID[t], t
    sample = "SELECT name, age FROM head WHERE age > 56 ORDER BY age"
    enc = tok.encode(sample)
    print(len(sample.split()), "words ->", len(enc.ids), "tokens")
    print(enc.tokens)


if __name__ == "__main__":
    main()
