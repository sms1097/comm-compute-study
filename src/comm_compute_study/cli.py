from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from .models import ModelSpec
from .training import TrainingConfig, run_training_loop


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="comm-compute-study",
        description="TinyStories-style GPT pretraining entrypoint.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to a TinyStories text/json/jsonl file or a directory containing shards.",
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of optimizer steps to run.")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size.")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps before each optimizer update.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        help="Number of tokens per training sample.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=3e-5,
        help="Minimum learning rate after cosine decay.",
    )
    parser.add_argument("--warmup-steps", type=int, default=10, help="Linear warmup steps.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay.")
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set to 0 to disable clipping.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between logs.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Steps between checkpoints. Set to 0 to disable checkpointing.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to write checkpoints into.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Compute dtype: auto, float32, float16, or bfloat16.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Compile the model with torch.compile when available.",
    )
    parser.add_argument(
        "--disable-torchtitan",
        action="store_true",
        help="Skip the optional TorchTitan availability check.",
    )
    parser.add_argument("--model-name", default="baseline", help="Display name for the model configuration.")
    parser.add_argument("--hidden-size", type=int, default=768, help="Transformer hidden size.")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers.")
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=12,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--num-key-value-heads",
        type=int,
        default=4,
        help="Number of key/value heads for grouped-query attention.",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=None,
        help="Feed-forward width. Defaults to 4x hidden size.",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Residual dropout.")
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.0,
        help="Attention dropout.",
    )
    parser.add_argument(
        "--rope-base",
        type=float,
        default=10_000.0,
        help="RoPE base frequency.",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Enable bias terms in linear projections.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = TrainingConfig(
        data_path=args.data_path,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile_model,
        use_torchtitan=not args.disable_torchtitan,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        rope_base=args.rope_base,
        bias=args.bias,
    )
    model = ModelSpec(
        name=args.model_name,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )

    print(f"Running training for {model.name}")
    run_training_loop(config)
    return 0
