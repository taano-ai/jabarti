
"""
STAGE 1 - NORMALIZATION
Standardize text before tokenization.
Different spellings of the same word become one form.
Some information is intentionally removed (lossy normalization).
"""

from tokenizers import Regex
from tokenizers.normalizers import NFC, Replace, Sequence


def build_normalizer():
    # Run normalization steps in order, from top to bottom.
    return Sequence([
        # Combine equivalent Unicode representations (non-lossy).
        NFC(),
        # Remove Arabic diacritics and tatweel.
        Replace(Regex(r"[\u064B-\u0652\u0640]"), ""),
        Replace("أ", "ا"),
        Replace("إ", "ا"),
        Replace("آ", "ا"),
        Replace("ى", "ي"),

        # Replace consecutive spaces/tabs with one space.
        # Prevents Metaspace from creating empty space tokens.
        Replace(Regex(r"[ \t]+"), " "),
    ])


if __name__ == "__main__":
    normalizer = build_normalizer()
    text = "  أَحْمَـد\t\tإِلَى   المَدْرَسَة  "

    # Test normalization without running the full tokenizer.
    print(repr(normalizer.normalize_str(text)))
