from tokenizers import Regex
from tokenizers.normalizers import NFC, Replace, Sequence
text = "  أَحْمَـد\t\tإِلَى                              المَدْرَسَة  "
def build_normalizer():
    return Sequence([
        NFC(),
        Replace(Regex(r"[\u064B-\u0652\u0640]"), ""),
        Replace("أ", "ا"),
        Replace("إ", "ا"),
        Replace("آ", "ا"),
        Replace("ى", "ي"),
        Replace(Regex(r"[ \t]+"), " "),
    ])

def normalize_text(text):
    return build_normalizer().normalize_str(text)

# res = normalize_text(text)
# print(res)