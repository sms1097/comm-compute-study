from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import AttentionCache, TransformerBlock


@dataclass(slots=True)
class GPTConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int | None = None
    dropout: float = 0.0
    attention_dropout: float = 0.0
    rope_base: float = 10_000.0
    bias: bool = False

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    rope_base=config.rope_base,
                    bias=config.bias,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(config.hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: list[AttentionCache | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list[AttentionCache] | None]:
        if input_ids.size(-1) > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.size(-1)} exceeds max_seq_len={self.config.max_seq_len}"
            )

        hidden_states = self.token_embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        next_past_key_values: list[AttentionCache] | None = [] if use_cache else None
        for layer, layer_cache in zip(self.layers, past_key_values):
            hidden_states, next_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=layer_cache,
            )
            if use_cache and next_cache is not None:
                next_past_key_values.append(next_cache)

        hidden_states = self.final_norm(hidden_states)
        return hidden_states, next_past_key_values


class GPTLMHeadModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.token_embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        past_key_values: list[AttentionCache | None] | None = None,
        use_cache: bool = False,
    ) -> dict[str, Tensor | list[AttentionCache] | None]:
        hidden_states, next_past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)

        outputs: dict[str, Tensor | list[AttentionCache] | None] = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
            outputs["loss"] = loss
        if use_cache:
            outputs["past_key_values"] = next_past_key_values
        return outputs
