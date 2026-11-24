from normalizer import build_normalizer
from pre_tokenizer import build_pretokenizer
from tokenizers import trainers, decoders, models, Tokenizer
vocab_size = 1000
special_tokens = ["<unk>","<pad>", "<eos>"]
tokenizer  = Tokenizer(models.BPE(unk_token="<unk>"))

tokenizer.normalizer = build_normalizer()
tokenizer.pre_tokenizer = build_pretokenizer()
tokenizer.decoder = decoders.Metaspace(prepend_scheme="always")

trainer = trainers.BpeTrainer(vocab_size=vocab_size,special_tokens=special_tokens)

file_name = "corpus.txt"
with open(file_name, "r") as f:
    tokenizer.train_from_iterator((line for line in f), trainer)

tokenizer.save("tokenizer.json")

loaded_tokenizer = Tokenizer.from_file("tokenizer.json")
#test 
sentence = "I like travel to the mecca"
ids  = loaded_tokenizer.encode(sentence).ids
print("ids: ", ids)
print("decoded: ", loaded_tokenizer.decode(ids))

print("Vocab size:", loaded_tokenizer.get_vocab_size())

