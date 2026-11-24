from huggingface_hub import hf_hub_download
import pandas as pd
import random

# 1. Download the dataset file
downloaded_path = hf_hub_download(
    repo_id="roneneldan/TinyStories",
    filename="data/validation-00000-of-00001-869c898b519ad725.parquet",
    repo_type="dataset",
)


# 2. Load the Parquet file into Pandas
df = pd.read_parquet(downloaded_path)


# # 3. Inspect the dataset
# print("Shape:")
# print(df.shape)

# print("\nColumns:")
# print(df.columns)

# print("\nFirst rows:")
# print(df.head())

# print("\nData types:")
# print(df.dtypes)


# # 4. Check missing values
# nulls = df.isnull().sum()
# x = nulls[nulls > 0]

# print("\nMissing values:")
# print("No null values" if x.empty else x)


# # 5. Inspect text lengths
# min_length = df["text"].str.len().min()
# max_length = df["text"].str.len().max()
# print("\nText lengths:")
# print("Minimum text length:", min_length)
# print("Maximum text length:", max_length)


# # Find the shortest story
# shortest = df[df["text"].str.len() == min_length]
# # Find the longest story
# longest = df[df["text"].str.len() == max_length]

# # Print the first 100 characters of each
# print(shortest["text"][:100])
# print("*"*30)
# print(longest["text"][:100])
# # Pick and print a random story
# print("random stroy: \n")


# print(df["text"][random.randint(1, 1000)])

# # Count total characters across all stories
print("all chars in all stories: ", df["text"].str.len().sum())
complete_corpus = " ".join(df["text"][:100])
# print(len(complete_corpus))
# print(complete_corpus[:500])
with open("corpus.txt","w", newline="") as f:
    f.write(complete_corpus)

with open("corpus.txt", "r") as f:
    text = f.read()
import re
print("number of new lines:" ,text.count("\n"))
print("number of double new lines:" ,text.count("\n\n"))
print("number of tabas: ", text.count("\t"))
print("number of consective spaces", len(re.findall(r"  +",text)))
