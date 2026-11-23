from collections import Counter
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, Regex
from normalizer import build_normalizer
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



# ---------- the two steps your toy BPE was missing ----------

pre_tok = pre_tokenizers.Metaspace(prepend_scheme="always")

texts = [
    "أَحْمَد إِلَى المَدْرَسَة",
    "احمد في المدرسة",
    "الطالب في المدرسة",
]
test = "أَحْمَد في المدرسة"
NUM_MERGES = 16


def to_chunks(text):
    """normalize -> pre-tokenize -> chunk strings like '▁احمد'"""
    return [piece for piece, _ in pre_tok.pre_tokenize_str(build_normalizer().normalize_str(text))]


if __name__ == "__main__":

    chunks = [list(c) for t in texts for c in to_chunks(t)]
    alphabet = get_alphabet(chunks)
    _, merges = train_bpe(chunks, NUM_MERGES)
    vocab = build_vocab(alphabet, merges)

    toy_pieces = [p for c in to_chunks(test) for p in apply_merges(c, merges)]
    toy_ids = [vocab.get(p, 0) for p in toy_pieces]


    print("\nMY toy pieces:", toy_pieces)
    print("\nMY toy ids:", toy_ids)
    