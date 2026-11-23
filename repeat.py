from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from normalizer import build_normalizer

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = build_normalizer()
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")
trainer = trainers.BpeTrainer(vocab_size=40, special_tokens=["<unk>"])

texts = [
    "أَحْمَد إِلَى المَدْرَسَة",
    "احمد في المدرسة",
    "الطالب في المدرسة",
]
tokenizer.train_from_iterator(texts, trainer)

# for piece, id_ in sorted(tokenizer.get_vocab().items(), key=lambda x:x[1]):
#     print(id_, piece)

# print(tokenizer.get_vocab())
# print(tokenizer.get_vocab_size())
# print(tokenizer.normalizer.normalize_str(text))
# print(tokenizer.pre_tokenizer.pre_tokenize_str("احمد الي المدرسة"))
out = tokenizer.encode("أَحْمَد في المدرسة")
# print(out.tokens)
# print(out.ids)

tokenizer.decoder = decoders.Metaspace(prepend_scheme="always")
# print(repr(tokenizer.decode(out.ids)))
# tokenizer.save("tokenizer.json")
loaded = Tokenizer.from_file("tokenizer.json")
out = loaded.encode("أَحْمَد في المدرسة")
print(out.ids)
print(loaded.decode(out.ids))