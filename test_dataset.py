from training.dataset import TextDataset


dataset = TextDataset(
    "data/train.txt",
    "tokenizer/tokenizer.json",
    seq_len=8
)

print("Dataset size:", len(dataset))

x, y = dataset[0]

print("Input:", x)
print("Target:", y)

print("Input shape:", x.shape)
print("Target shape:", y.shape)

print("Dataset OK")