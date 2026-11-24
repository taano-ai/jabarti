from collections import Counter
"""
stage 3 by hand: models.BPE + BpeTrainer
"""

def count_pairs(words):
    """Count how often each pair of neighbouring symbols appears."""
    c = Counter()
    for array in words:
        for pair in zip(array, array[1:]):
            c[pair] += 1
    return c


def merge_pair(words, pair):
    """Glue every occurrence of `pair` into one symbol, in every word."""
    new_words = []
    for array in words:
        merged = []
        i = 0
        while i < len(array):
            if i < len(array) - 1 and (array[i], array[i + 1]) == pair:
                merged.append(array[i] + array[i + 1])
                i += 2
            else:
                merged.append(array[i])
                i += 1
        new_words.append(merged)
    return new_words


def train_bpe(words, num_merges):
    """Learn the merge recipe: repeat 'count pairs, merge the most frequent'."""
    merges = []
    for step in range(num_merges):
        counts = count_pairs(words)
        if not counts:
            break
        best_pair = counts.most_common(1)[0][0]
        words = merge_pair(words, best_pair)
        merges.append(best_pair)
        print(f"step {step + 1}: merge {best_pair}")
    return words, merges


def apply_merges(chars, merges):
    """Encode one word: replay the recipe in the order it was learned."""
    words = [chars]
    for pair in merges:
        words = merge_pair(words, pair)
    return words[0]


def get_alphabet(words):
    """All distinct characters in the corpus, sorted so IDs are stable."""
    uniques = set()
    for array in words:
        uniques.update(array)
    return sorted(uniques)


def build_vocab(alphabet, merges):
    """Map every piece to an ID. <unk> is 0, then characters, then merges."""
    vocab = {"<unk>": 0}
    for ch in alphabet:
        vocab[ch] = len(vocab)
    for pair in merges:
        vocab["".join(pair)] = len(vocab)
    return vocab


def encode(chars, merges, vocab):
    """text pieces -> IDs. Unknown pieces become the <unk> ID."""
    pieces = apply_merges(chars, merges)
    return [vocab.get(p, vocab["<unk>"]) for p in pieces]

# ---------------------------------------------------------------------------
# THE MISSING PARTS (commented out so this file stays standalone)
# This file only does stage 3 (BPE). On real text, stages 1 and 2 must run first:
#   raw text -> normalize -> pre-tokenize -> chunks -> BPE
# To plug them in, uncomment this block (it needs normalizer.py and pre_tokenizer.py).
#
# from normalizer import build_normalizer
# from pre_tokenizer import build_pre_tokenizer
#
# normalizer = build_normalizer()
# pre_tok = build_pre_tokenizer()
#
#
# def to_chunks(text):
#     """normalize -> pre-tokenize -> chunk strings like '▁احمد'"""
#     normalized = normalizer.normalize_str(text)           # stage 1: clean the text
#     # stage 2: pre_tokenize_str gives (chunk, (start, end)); we keep only the chunk
#     return [piece for piece, _ in pre_tok.pre_tokenize_str(normalized)]
#
#
# texts = ["أَحْمَد إِلَى المَدْرَسَة", "احمد في المدرسة", "الطالب في المدرسة"]
# # each chunk becomes a list of characters, which is what count_pairs / merge_pair expect
# corpus = [list(chunk) for text in texts for chunk in to_chunks(text)]
#
# # to encode a new sentence: chunk it first, then run apply_merges on EACH chunk
# # pieces = [p for chunk in to_chunks("أَحْمَد في المدرسة") for p in apply_merges(chunk, merges)]
# 

if __name__ == "__main__":
    # corpus = chunks that pre-tokenization would give, split into characters
    corpus = [
        ["▁", "l", "o", "w"],
        ["▁", "l", "o", "w"],
        ["▁", "l", "o", "w"],
        ["▁", "l", "o", "w", "e", "r"],
        ["▁", "n", "e", "w"],
        ["▁", "n", "e", "w", "e", "r"],
    ]
    NUM_MERGES = 4

    alphabet = get_alphabet(corpus)             # BEFORE training: training changes the words
    _, merges = train_bpe(corpus, NUM_MERGES)
    vocab = build_vocab(alphabet, merges)

    print("\nvocab :", vocab)
    print("pieces:", apply_merges("▁lowly", merges))   # ['▁low', 'l', 'y']
    print("ids   :", encode("▁lowly", merges, vocab))  # 'y' was never seen -> <unk> (0)
