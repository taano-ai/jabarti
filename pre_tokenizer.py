from collections import Counter

words = [
    ["▁", "l", "o", "w"],
    ["▁", "l", "o", "w"],
    ["▁", "l", "o", "w", "e", "r"],
]

def count_pairs(words):
    c = Counter()
    for array in words:
        pairs = zip(array, array[1:])
        for pair in pairs:
            c[pair] += 1
    
    return c
def merge_pair(words, pair):
    new_words = []
    for array in words:
        single_array = []
        pairs = list(zip(array, array[1:]))
        for i in range(len(pairs)):
            if pairs[i] == pair:
                single_array.append("".join(pair))
            else:
                single_array.extend(pairs[i][-1])
        
        new_words.append(single_array)
    return new_words
print(merge_pair(words, ("o", "w")))

