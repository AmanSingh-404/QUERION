SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|schema|>",
    "<|question|>",
    "<|sql|>",
    "<|error|>",
    "<|sep|>",
]

TOKEN_ID = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
PAD_ID = TOKEN_ID["<|pad|>"]
BOS_ID = TOKEN_ID["<|bos|>"]
EOS_ID = TOKEN_ID["<|eos|>"]
