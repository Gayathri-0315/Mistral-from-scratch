import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import MistralAttention
from .mlp import MistralMLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        intermediate_size,
        max_seq_len
    ):
        super().__init__()

        self.attention_norm = RMSNorm(hidden_size)

        self.attention = MistralAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            max_seq_len
        )

        self.ffn_norm = RMSNorm(hidden_size)

        self.mlp = MistralMLP(
            hidden_size,
            intermediate_size
        )
    def forward(self, x):

        # Attention + residual connection
        residual = x

        x = self.attention_norm(x)

        x = self.attention(x)

        x = residual + x

        # MLP + residual connection
        residual = x

        x = self.ffn_norm(x)

        x = self.mlp(x)

        x = residual + x

        return x