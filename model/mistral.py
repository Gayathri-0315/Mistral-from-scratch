import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .transformer import MistralTransformerBlock


class MistralModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_heads,
        intermediate_size,
        num_layers,
        max_seq_len=2048
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            vocab_size,
            hidden_size
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            MistralTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(hidden_size)

        # Language-model head
        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False
        )

    def forward(self, input_ids):
        # Token IDs → embeddings
        x = self.embed_tokens(input_ids)

        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        # Hidden states → vocabulary logits
        logits = self.lm_head(x)

        return logits