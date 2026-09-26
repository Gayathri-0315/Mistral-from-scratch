import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import MistralAttention
from .mlp import MistralMLP


class MistralTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        max_seq_len=2048
    ):
        super().__init__()

        self.input_layernorm = RMSNorm(hidden_size)

        self.self_attn = MistralAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )

        self.post_attention_layernorm = RMSNorm(hidden_size)

        self.mlp = MistralMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )

    def forward(self, x):

        # Attention + residual connection
        residual = x

        x = self.input_layernorm(x)

        x = self.self_attn(x)

        x = residual + x

        # MLP + residual connection
        residual = x

        x = self.post_attention_layernorm(x)

        x = self.mlp(x)

        x = residual + x

        return x