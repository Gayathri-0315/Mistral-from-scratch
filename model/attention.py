import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RotaryEmbedding, apply_rotary_pos_emb


class MistralAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        dropout=0.0
    ):
        super().__init__()

        assert num_heads % num_kv_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = hidden_size // num_heads

        self.num_key_value_groups = (
            num_heads // num_kv_heads
        )

        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            hidden_size,
            num_kv_heads * self.head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            hidden_size,
            num_kv_heads * self.head_dim,
            bias=False
        )

        self.o_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len
        )

        self.dropout = dropout

    def repeat_kv(self, x):

        if self.num_key_value_groups == 1:
            return x

        batch, kv_heads, seq_len, head_dim = x.shape

        x = x[:, :, None, :, :]

        x = x.expand(
            batch,
            kv_heads,
            self.num_key_value_groups,
            seq_len,
            head_dim
        )

        return x.reshape(
            batch,
            kv_heads * self.num_key_value_groups,
            seq_len,
            head_dim
        )

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rope(seq_len)

        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin
        )

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
        )

        attn_output = attn_output.view(
            batch_size,
            seq_len,
            self.hidden_size
        )

        return self.o_proj(attn_output)