# adapters.py  (repo root, not inside tests/)
from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


# ---- BPE glue ----
def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    return train_bpe(str(input_path), vocab_size, special_tokens)


def get_tokenizer(vocab, merges, special_tokens=None):
    return Tokenizer(vocab, merges, special_tokens)



