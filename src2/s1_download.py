import pandas as pd
from huggingface_hub import hf_hub_download


def load_df():
    path = hf_hub_download(
        repo_id="roneneldan/TinyStories",
        filename="data/validation-00000-of-00001-869c898b519ad725.parquet",
        repo_type="dataset",
    )
    return pd.read_parquet(path)


# the checks live under __main__ so importing load_df from tokenizer.py doesn't run them
if __name__ == "__main__":
    df = load_df()
    print(df.shape, list(df.columns))
    print("nulls:", df["text"].isnull().sum())
    lens = df["text"].str.len()
    print("min/max length:", lens.min(), lens.max())
    # here you have to index first, then slice: a bare Series[:100] slices ROWS, not characters
    print(df.loc[lens.idxmin(), "text"][:100])
    # series count alwasy takes regex unlike normal count
    print("newlines:", df["text"].str.count("\n").sum(),
          "| tabs:", df["text"].str.count("\t").sum(),
          "| double spaces:", df["text"].str.count("  +").sum())