import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Frequencies for each pair of dimensions
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        x shape:
        [batch, heads, sequence_length, head_dim]
        """

        seq_len = x.shape[-2]

        positions = torch.arange(
            seq_len,
            device=x.device,
            dtype=self.inv_freq.dtype
        )

        # Position × frequency
        freqs = torch.outer(positions, self.inv_freq)

        # Duplicate frequencies for rotation
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def rotate_half(x):
    """
    Split the last dimension into two halves
    and rotate them.
    """

    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply RoPE to query and key tensors.
    """

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k