from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "stanfordnlp/imdb"
MAX_CHUNK = 20

HF_FILES = {
    "train": "plain_text/train-00000-of-00001.parquet",
    "test": "plain_text/test-00000-of-00001.parquet",
    "unsupervised": "plain_text/unsupervised-00000-of-00001.parquet",
}

# get only train splits
Train_splits = ("train")
OUTPUT_DIR = Path(__file__).parent / "output"
# in this function never take more than 20 chunks  if split is more thatn so

def download_and_filter():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    for name, filename in HF_FILES.items():
        local = hf_hub_download(repo_id=REPO,repo_type="dataset", filename=filename)
        df = pd.read_parquet(local)
        if name in Train_splits: 
            df = (
                df.sort_values(["text", "label"])
                .groupby("text", sort=False)
                .head(MAX_CHUNK).reset_index(drop=True)
            )

download_and_filter()