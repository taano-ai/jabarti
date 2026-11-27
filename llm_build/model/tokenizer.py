from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer

TOKENIZER_PATH = Path(__file__).parent / "assets" / "tokenizer.json"


class Tokenizer:
    UNK, PAD, EOS = 0, 1, 2
    SPECIAL_TOKENS = {"<unk>": UNK, "<pad>": PAD, "<eos>": EOS}

    def __init__(self, hf_tokenizer: HFTokenizer) -> None:
        self._tokenizer = hf_tokenizer

    @classmethod
    def from_file(cls, path: Path = TOKENIZER_PATH) -> "Tokenizer":
        return cls(HFTokenizer.from_file(str(path)))

    def normalize(self, text: str) -> str:
        return self._tokenizer.normalizer.normalize_str(text)

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()