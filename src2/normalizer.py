from tokenizers import Regex
from tokenizers.normalizers import NFC, Replace, Sequence, Strip


def build_normalizer():
    return Sequence([
        NFC(),
        # here you have to collapse ALL whitespace (incl. "\n") because Metaspace only cuts at spaces,
        # so "\n" would stick to words ("▁play.\n") and waste vocab slots.
        # for a model that must write paragraphs: keep "\n" and split on it in the pre-tokenizer instead
        Replace(Regex(r"\s+"), " "),
        # trim the edges because a leading/trailing space creates a lone "▁" chunk
        Strip(),
    ])