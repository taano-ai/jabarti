"""
ASSEMBLY - THE HUGGING FACE TOKENIZER (stages 1 + 2 + 3 together)
  stage 1 normalizer.py     -> cleaning
  stage 2 pre_tokenizer.py  -> chunks
  stage 3 BPE               -> models.BPE + BpeTrainer here (by hand: bpe_from_scratch.py)
This file plugs them together, trains BPE on corpus.txt, saves tokenizer.json,
then turns the whole corpus into token IDs for the model.
Run it: `uv run build_tokenizer.py`
"""

from pathlib import Path

from tokenizers import Tokenizer, models, trainers, decoders

from normalizer import build_normalizer            # stage 1
from pre_tokenizer import build_pre_tokenizer      # stage 2

# ---------- settings ----------
HERE = Path(__file__).parent                       # run from any folder, paths still work
CORPUS_PATH = HERE / "corpus.txt"
TOKENIZER_PATH = HERE / "tokenizer.json"
VOCAB_SIZE = 500                                   # TOTAL table size, not the number of merges
SPECIAL_TOKENS = ["<unk>", "<pad>", "<eos>"]       # reserved IDs 0, 1, 2

# ---------- 1. build the empty tokenizer and plug in the stages ----------
# the BPE model starts empty (no vocab, no merges); unk_token names the "unknown" piece
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = build_normalizer()          # slot 1: runs first on every text
tokenizer.pre_tokenizer = build_pre_tokenizer()    # slot 2: cuts the text into chunks
tokenizer.decoder = decoders.Metaspace(prepend_scheme="always")  # undoes Metaspace: "▁" -> " "

# ---------- 2. train ----------
# the trainer holds the learning rules; the tokenizer is the thing being trained
# vocab_size = special tokens + single characters + merges
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

with open(CORPUS_PATH, encoding="utf-8") as f:
    # a file is an iterator of lines; strip() removes the "\n" so it never enters the vocab
    tokenizer.train_from_iterator((line.strip() for line in f), trainer)

# ---------- 3. save: tokenizer.json is the frozen contract with the model ----------
# one file holds everything: normalizer, pre-tokenizer, vocab + merges, decoder
tokenizer.save(str(TOKENIZER_PATH))

# ---------- 4. sanity checks ----------
loaded = Tokenizer.from_file(str(TOKENIZER_PATH))  # load it back, like the model side will
text = "أَحْمَد في المدرسة"
out = loaded.encode(text)                          # runs normalize -> pre-tokenize -> BPE -> IDs
print("tokens :", out.tokens)
print("ids    :", out.ids)
print("decoded:", loaded.decode(out.ids))          # no diacritics: normalization lost them for good

vocab = loaded.get_vocab()                         # dict {piece: id}, NOT sorted
n_chars = len([p for p in vocab if len(p) == 1])   # single characters are the only pieces of length 1
print("vocab size :", loaded.get_vocab_size())
print("characters :", n_chars)
print("merges     :", VOCAB_SIZE - len(SPECIAL_TOKENS) - n_chars)
print("pieces with a newline:", [p for p in vocab if "\n" in p])   # should be []

# ---------- 5. corpus -> one long list of IDs for the model ----------
eos_id = loaded.token_to_id("<eos>")

all_ids = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:                                   # skip the blank lines between paragraphs
            all_ids.extend(loaded.encode(line).ids)

# <eos> marks the end of a whole DOCUMENT. corpus.txt is one article, so we add it once.
# With many articles, add one <eos> after each article.
all_ids.append(eos_id)

print("total IDs:", len(all_ids))
print("last IDs :", all_ids[-3:])                  # ends with the <eos> ID (2)