"""Special token names shared by the tokenizer script, data prep, and sampling."""
EOD = "<|end|>"
PAD = "<|pad|>"


def reserved(n: int = 0):
    return [f"<|rsv{i}|>" for i in range(n)]


# The tokenizer script creates specials = [EOD, PAD] + reserved(14) = 16 total.
def all_specials(vocab_margin: int = 16):
    return [EOD, PAD] + reserved(vocab_margin - 2)
