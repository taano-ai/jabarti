from pathlib import Path



TOKENIZER_PATH = Path(__file__).parent / "assets"/ "tokenizer.json"
from tokenizers import Tokenizer as ParentTokenizer

class Tokenizer:
    UNK = 0
    PAD = 1
    EOS = 2
    SPECIAL_TOKENS = {"<unk>": UNK, "<pad>": PAD, "<eos>":EOS}

    def __init__(self, parent_tokenizer: ParentTokenizer):
        self.tokenizer = parent_tokenizer

    def load(self, path=TOKENIZER_PATH):
        self.tokenizer = self.tokenizer.from_file(TOKENIZER_PATH)

    def normalize(self, text: str) -> str:
        return self.tokenizer.normalizer.normalize_str(text)
    def encode(self, text: str,asp=False)-> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=asp).ids
    def decode(self, inputs_ids: list[int],sst=True) -> str:
        return self.tokenizer.decode(inputs_ids,skip_special_tokens=sst)
    def tokeinze(self, text: str, asp=False)-> list[str]:
        return self.tokenizer.encode(text, add_special_tokens=asp).tokens
    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()