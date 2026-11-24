"""
STAGE 2 - PRE-TOKENIZATION
Cut the normalized text into chunks. BPE may only merge INSIDE a chunk, never
across two chunks (these are the "fences").
Metaspace cuts at every space, and keeps the space as the symbol "▁" attached
to the start of the next word, so no information is lost.
"""

from tokenizers import pre_tokenizers


def build_pre_tokenizer():
    # prepend_scheme="always": add a "▁" at the very start of the text too,
    # so the first word and later words are written the same way (▁hello, not hello)
    return pre_tokenizers.Metaspace(prepend_scheme="always")


if __name__ == "__main__":
    # quick test: run `uv run pre_tokenizer.py`
    pre_tokenizer = build_pre_tokenizer()

    # each result is (chunk, (start, end)), the positions are in the ORIGINAL text
    print(pre_tokenizer.pre_tokenize_str("hello big world"))
    # [('▁hello', (0, 5)), ('▁big', (5, 9)), ('▁world', (9, 15))]

    # Metaspace cuts only at spaces, so punctuation stays glued to the word
    print(pre_tokenizer.pre_tokenize_str("hello, world!"))
    # [('▁hello,', (0, 6)), ('▁world!', (6, 13))]

    # two spaces in a row leave a lone "▁" chunk: this is why the normalizer
    # collapses spaces BEFORE this stage
    print(pre_tokenizer.pre_tokenize_str("a   b"))
    # [('▁a', ...), ('▁', ...), ('▁', ...), ('▁b', ...)]