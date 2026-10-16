from tokenizer.special_tokens import BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, TOKEN_ID


def test_unique():
    assert len(set(SPECIAL_TOKENS)) == len(SPECIAL_TOKENS)


def test_fixed_ids():
    assert (PAD_ID, BOS_ID, EOS_ID) == (0, 1, 2)
    assert TOKEN_ID["<|sql|>"] == 5


def test_ids_are_contiguous():
    assert sorted(TOKEN_ID.values()) == list(range(len(SPECIAL_TOKENS)))
