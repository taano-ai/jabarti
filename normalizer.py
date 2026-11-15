from tokenizers import normalizers
normalizer = normalizers.NFC()

text = "e\u0301"

print("Original:", text)
print("Normalized:", normalizer.normalize_str(text))
print("Original length:", len(text))
print("Normalized length:", len(normalizer.normalize_str(text)))