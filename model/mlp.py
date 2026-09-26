import torch
import torch.nn as nn
import torch.nn.functional as F


class MistralMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        # Gate projection
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        # Up projection
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        # Down projection
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False
        )

    def forward(self, x):
        # SwiGLU:
        # SiLU(gate(x)) * up(x)
        gated = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        return self.down_proj(gated * up)