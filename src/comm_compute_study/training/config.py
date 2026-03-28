from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for TinyStories-style GPT pretraining."""

    data_path: Path | None = None
    steps: int = 100
    batch_size: int = 8
    grad_accum_steps: int = 1
    sequence_length: int = 256
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 10
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 1337
    log_interval: int = 10
    checkpoint_interval: int = 0
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "auto"
    dtype: str = "auto"
    compile_model: bool = False
    use_torchtitan: bool = True
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
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be at least 1")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate must be non-negative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be less than or equal to learning_rate")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.log_interval < 1:
            raise ValueError("log_interval must be at least 1")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads < 1:
            raise ValueError("num_key_value_heads must be positive")
