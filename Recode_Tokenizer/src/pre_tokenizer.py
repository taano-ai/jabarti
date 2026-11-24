from tokenizers.pre_tokenizers import Metaspace

def build_pretokenizer():
    return Metaspace(prepend_scheme="always")

