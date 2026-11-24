from tokenizers.normalizers import NFC, Sequence

def build_normalizer():
    return Sequence([
        NFC(),
    ])