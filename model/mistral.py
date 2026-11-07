import torch
import torch.nn as nn

from model.transformer import TransformerBlock
from model.rmsnorm import RMSNorm


class MistralModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        intermediate_size,
        num_layers,
        max_seq_len
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                num_kv_heads,
                intermediate_size,
                max_seq_len
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(hidden_size)

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False
        )

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits