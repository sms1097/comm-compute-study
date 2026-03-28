from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

RMSNorm = nn.RMSNorm


def _rotate_half(x: Tensor) -> Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE frequencies for a fixed head dimension."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        freqs = torch.outer(positions.to(device=device, dtype=self.inv_freq.dtype), self.inv_freq)
        angles = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = angles.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        sin = angles.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin


@dataclass(slots=True)
class AttentionCache:
    key: Tensor
    value: Tensor


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with RoPE and grouped-query attention.

    Flash attention is provided by `scaled_dot_product_attention`, which will
    dispatch to fused kernels on supported CUDA builds.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10_000.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cache: AttentionCache | None = None,
    ) -> tuple[Tensor, AttentionCache | None]:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            start = 0 if cache is None else cache.key.size(-2)
            position_ids = torch.arange(start, start + seq_len, device=hidden_states.device)
        elif position_ids.dim() != 1:
            position_ids = position_ids.reshape(-1)

        cos, sin = self.rotary(
            position_ids,
            dtype=query.dtype,
            device=hidden_states.device,
        )
        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)

        if cache is not None:
            key = torch.cat((cache.key, key), dim=-2)
            value = torch.cat((cache.value, value), dim=-2)
        next_cache = AttentionCache(key=key, value=value)

        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_mask = None
        is_causal = True
        if attention_mask is not None:
            is_causal = False
            total_kv_len = key.size(-2)
            causal_mask = torch.ones(
                (seq_len, total_kv_len),
                device=hidden_states.device,
                dtype=torch.bool,
            ).tril(diagonal=total_kv_len - seq_len)
            padding_mask = attention_mask[:, None, None, :total_kv_len].to(torch.bool)
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) & padding_mask

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output), next_cache


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return self.dropout(hidden_states)


FeedForward = SwiGLUFeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        rope_base: float = 10_000.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout=attention_dropout,
            rope_base=rope_base,
            bias=bias,
        )
        self.mlp_norm = nn.RMSNorm(hidden_size)
        self.mlp = SwiGLUFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cache: AttentionCache | None = None,
    ) -> tuple[Tensor, AttentionCache | None]:
        attn_output, next_cache = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, next_cache
