from tokenizers import Tokenizer, models

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

print(tokenizer.get_vocab())
print(tokenizer.get_vocab_size())