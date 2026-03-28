from __future__ import annotations

import math
import time
from dataclasses import asdict

import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast

from ..models import GPTConfig, GPTLMHeadModel
from .config import TrainingConfig
from .data import TokenizedCorpus, load_tokenized_corpus, sample_batch


def run_training_loop(config: TrainingConfig) -> None:
    device = _resolve_device(config.device)
    training_dtype = _resolve_training_dtype(config.dtype, device)
    dtype_name = str(training_dtype).replace("torch.", "")

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    corpus = load_tokenized_corpus(config.data_path)
    model = _build_model(config, corpus).to(device=device)

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda" and training_dtype == torch.float16)

    titan_status = _torchtitan_status(config.use_torchtitan)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    tokens_per_step = config.batch_size * config.grad_accum_steps * config.sequence_length

    print(
        f"Starting pretraining on {device} "
        f"(dtype={dtype_name}, params={num_parameters:,}, "
        f"documents={corpus.num_documents:,}, tokens={corpus.num_tokens:,})"
    )
    print(titan_status)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, config.steps + 1):
        step_start = time.perf_counter()
        loss_value = 0.0

        for _micro_step in range(config.grad_accum_steps):
            input_ids = sample_batch(corpus.tokens, config.batch_size, config.sequence_length, device)

            with autocast(
                device_type=device.type,
                dtype=training_dtype,
                enabled=device.type == "cuda" and training_dtype in {torch.float16, torch.bfloat16},
            ):
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs["loss"]
                if not isinstance(loss, Tensor):
                    raise TypeError("Expected the model to return a tensor loss")
                scaled_loss = loss / config.grad_accum_steps

            scaler.scale(scaled_loss).backward()
            loss_value += float(loss.detach())

        grad_norm = None
        if config.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        learning_rate = _compute_learning_rate(
            step=step,
            total_steps=config.steps,
            base_learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_steps=config.warmup_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step_time = time.perf_counter() - step_start
        tokens_per_second = tokens_per_step / step_time

        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            grad_norm_str = "n/a" if grad_norm is None else f"{float(grad_norm):.3f}"
            print(
                "step="
                f"{step} "
                f"loss={loss_value / config.grad_accum_steps:.4f} "
                f"lr={learning_rate:.6f} "
                f"grad_norm={grad_norm_str} "
                f"step_time={step_time:.3f}s "
                f"tok/s={tokens_per_second:,.0f}"
            )

        if config.checkpoint_interval and step % config.checkpoint_interval == 0:
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                corpus=corpus,
            )


def _build_model(config: TrainingConfig, corpus: TokenizedCorpus) -> GPTLMHeadModel:
    model_config = GPTConfig(
        vocab_size=corpus.vocab_size,
        max_seq_len=config.sequence_length,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        intermediate_size=config.intermediate_size,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        rope_base=config.rope_base,
        bias=config.bias,
    )
    return GPTLMHeadModel(model_config)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _resolve_training_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def _compute_learning_rate(
    *,
    step: int,
    total_steps: int,
    base_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_learning_rate * step / warmup_steps

    if total_steps <= warmup_steps:
        return base_learning_rate

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine_decay * (base_learning_rate - min_learning_rate)


def _save_checkpoint(
    *,
    model: GPTLMHeadModel,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    step: int,
    corpus: TokenizedCorpus,
) -> None:
    checkpoint_dir = config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"step-{step:06d}.pt"
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_config": asdict(config),
        "vocab_size": corpus.vocab_size,
    }
    torch.save(payload, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def _torchtitan_status(use_torchtitan: bool) -> str:
    if not use_torchtitan:
        return "TorchTitan integration disabled for this run."

    try:
        import torchtitan  # type: ignore
    except ImportError:
        return (
            "TorchTitan requested but not installed in this environment; "
            "running the local PyTorch loop instead."
        )

    version = getattr(torchtitan, "__version__", "unknown")
    return f"TorchTitan detected (version={version}); using the local loop with a compatible config layout."
