import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TextDataset(Dataset):
    def __init__(self, text_file, tokenizer_file, seq_len=32):
        self.seq_len = seq_len

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

        # Read training corpus
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Convert text to token IDs
        self.tokens = self.tokenizer.encode(text).ids

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, index):
        input_ids = self.tokens[
            index:index + self.seq_len
        ]

        target_ids = self.tokens[
            index + 1:index + self.seq_len + 1
        ]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )