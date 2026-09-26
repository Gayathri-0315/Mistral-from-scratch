import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary_pos_emb


class MistralAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_len=2048):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Query, Key and Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Rotary positional embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=max_seq_len
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project input into Q, K and V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [batch, seq, hidden]
        # -> [batch, heads, seq, head_dim]
        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rotary_emb(q)

        q, k = apply_rotary_pos_emb(
            q, k, cos, sin
        )

        # Causal self-attention
        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True
        )

        # [batch, heads, seq, head_dim]
        # -> [batch, seq, hidden]
        attention_output = attention_output.transpose(1, 2).contiguous()

        attention_output = attention_output.view(
            batch_size,
            seq_len,
            self.hidden_size
        )

        # Final projection
        return self.o_proj(attention_output)