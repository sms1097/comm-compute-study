"""Top-level package for training and experiment code."""

from .models import ModelSpec
from .training import TrainingConfig, run_training_loop

__all__ = ["ModelSpec", "TrainingConfig", "run_training_loop"]
