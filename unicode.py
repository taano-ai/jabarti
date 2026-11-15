characters = ["A", "م", "中", "😀"]

for char in characters:
    encoded = char.encode("utf-8")

    print("Character:", char)
    print("Unicode:", hex(ord(char)))
    print("Bytes:", encoded)
    print("Hex:", encoded.hex())
    print("Size:", len(encoded))
    print("---")