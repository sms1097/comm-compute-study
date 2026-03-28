from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelSpec:
    """Lightweight model metadata for training scripts to consume."""

    name: str
    hidden_size: int = 768
    num_layers: int = 12
    metadata: dict[str, str] = field(default_factory=dict)
