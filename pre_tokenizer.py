from collections import Counter

# Corpus: what pre-tokenization hands to BPE (each chunk split into characters)
corpus = [
    ["▁", "l", "o", "w"],
    ["▁", "l", "o", "w"],
    ["▁", "l", "o", "w", "e", "r"],
]


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


def apply_merges(string, merges):
    """Encode one word: replay the recipe in the order it was learned."""
    words = [list(string)]
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


def encode(text, merges, vocab):
    """text pieces -> IDs. Unknown pieces become the <unk> ID."""
    pieces = apply_merges(text, merges)
    return [vocab.get(p, vocab["<unk>"]) for p in pieces]


if __name__ == "__main__":
    # Alphabet must be taken BEFORE training (training changes the words)
    alphabet = get_alphabet(corpus)
    _, merges = train_bpe(corpus, 13)
    vocab = build_vocab(alphabet, merges)

    print("\nmerges:", merges)
    print("vocab :", vocab)

    word = "▁lowly"
    print("\npieces:", apply_merges(word, merges))
    print("ids   :", encode(word, merges, vocab))