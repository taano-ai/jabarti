from pathlib import Path
from tokenizers import Tokenizer, models, trainers, decoders
from normalizer import build_normalizer
from pre_tokenizer import build_pretokenizer
from s1_download import load_df

VOCAB_SIZE = 2000
SPECIAL_TOKENS = ["<unk>", "<pad>", "<eos>"]
# here you have to build the path from __file__ because a bare "tokenizer.json"
#  breaks when run from another folder
TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"


def batches(df, size=1000):
    texts = df["text"].tolist()
    for i in range(0, len(texts), size):
        yield texts[i:i + size]      # one batch = a list of stories; each story stays ONE text (no joining)


tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = build_normalizer()
tokenizer.pre_tokenizer = build_pretokenizer()
tokenizer.decoder = decoders.Metaspace(prepend_scheme="always")   # the decoder is saved INSIDE tokenizer.json

trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS, show_progress=True)

df = load_df()
tokenizer.train_from_iterator(batches(df), trainer, length=len(df))   # length: only for the progress bar
tokenizer.save(str(TOKENIZER_PATH))

# checks
loaded = Tokenizer.from_file(str(TOKENIZER_PATH))
ids = loaded.encode("I like to travel to the mecca").ids
print("vocab size:", loaded.get_vocab_size())
