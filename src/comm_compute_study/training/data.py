from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


class ByteTokenizer:
    """Tiny dependency-free tokenizer that works well for raw text corpora."""

    eos_token_id = 256
    vocab_size = 257

    def encode(self, text: str) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        token_ids.append(self.eos_token_id)
        return token_ids


@dataclass(slots=True)
class TokenizedCorpus:
    tokens: Tensor
    num_documents: int
    vocab_size: int

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.numel())


def load_tokenized_corpus(path: Path | None) -> TokenizedCorpus:
    tokenizer = ByteTokenizer()
    token_ids: list[int] = []
    num_documents = 0

    for document in _iter_documents(path):
        stripped = document.strip()
        if not stripped:
            continue
        token_ids.extend(tokenizer.encode(stripped))
        num_documents += 1

    if not token_ids:
        raise ValueError(
            "No training text was found. Pass --data-path pointing to a .txt, .jsonl, .json, "
            "or a directory containing those files."
        )

    tokens = torch.tensor(token_ids, dtype=torch.long)
    return TokenizedCorpus(
        tokens=tokens,
        num_documents=num_documents,
        vocab_size=tokenizer.vocab_size,
    )


def sample_batch(tokens: Tensor, batch_size: int, sequence_length: int, device: torch.device) -> Tensor:
    if tokens.numel() <= sequence_length:
        raise ValueError(
            f"Corpus has {tokens.numel()} tokens, which is not enough for sequence_length={sequence_length}."
        )

    max_start = tokens.numel() - sequence_length
    starts = torch.randint(0, max_start + 1, (batch_size,))
    batch = torch.stack([tokens[start : start + sequence_length] for start in starts.tolist()])
    return batch.to(device=device, non_blocking=device.type == "cuda")


def _iter_documents(path: Path | None) -> list[str]:
    if path is None:
        return [_fallback_text()]

    resolved_path = path.expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Training data path does not exist: {resolved_path}")

    if resolved_path.is_dir():
        documents: list[str] = []
        for file_path in sorted(resolved_path.rglob("*")):
            if file_path.suffix.lower() not in {".txt", ".jsonl", ".json"}:
                continue
            documents.extend(_read_documents_from_file(file_path))
        return documents

    return _read_documents_from_file(resolved_path)


def _read_documents_from_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [path.read_text(encoding="utf-8")]
    if suffix == ".jsonl":
        documents: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            text = _extract_text_field(payload)
            if text:
                documents.append(text)
        return documents
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [text for item in payload if (text := _extract_text_field(item))]
        text = _extract_text_field(payload)
        return [text] if text else []
    raise ValueError(f"Unsupported training file type: {path}")


def _extract_text_field(payload: object) -> str | None:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "story", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return None


def _fallback_text() -> str:
    return (
        "Once upon a time there was a tiny transformer who wanted to learn from stories. "
        "It practiced predicting the next token, one batch at a time, until the loop worked."
    )
