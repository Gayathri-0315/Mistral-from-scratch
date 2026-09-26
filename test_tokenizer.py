from tokenizer.tokenizer import create_tokenizer

tokenizer = create_tokenizer(
    "data/train.txt",
    vocab_size=1000
)

tokenizer.save("tokenizer/tokenizer.json")

text = "Artificial intelligence learns from data."

encoded = tokenizer.encode(text)

print("Tokens:")
print(encoded.tokens)

print("Token IDs:")
print(encoded.ids)

print("Decoded:")
print(tokenizer.decode(encoded.ids))

print("Vocabulary size:")
print(tokenizer.get_vocab_size())

print("\nTokenizer saved successfully!")