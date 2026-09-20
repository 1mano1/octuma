"""Evaluacion: perplejidad, tamano en memoria y velocidad de generacion."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn

from .quant.qlinear import QuantLinear


@dataclass
class PerplexityResult:
    perplexity: float
    n_windows: int
    seq_len: int
    seconds: float
    dataset: str


@torch.no_grad()
def perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    seq_len: int = 2048,
    device: torch.device | str = "cpu",
    dataset: str = "custom",
    max_windows: int | None = None,
    progress=None,
) -> PerplexityResult:
    """Perplejidad por ventanas deslizantes sin solape (metodo estandar)."""
    device = torch.device(device)
    model.eval()
    ids = input_ids.reshape(-1)
    n = ids.numel() // seq_len
    if max_windows:
        n = min(n, max_windows)
    if n == 0:
        raise ValueError("el texto es mas corto que una ventana")

    nll_sum = torch.tensor(0.0, dtype=torch.float64)
    n_tokens = 0
    t0 = time.perf_counter()

    for i in range(n):
        window = ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
        logits = model(window).logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = window[:, 1:]
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        nll_sum += loss.double().cpu()
        n_tokens += shift_labels.numel()
        if progress:
            progress(f"ventana {i + 1}/{n} · ppl parcial {torch.exp(nll_sum / n_tokens):.3f}")

    ppl = torch.exp(nll_sum / n_tokens).item()
    return PerplexityResult(
        perplexity=ppl,
        n_windows=n,
        seq_len=seq_len,
        seconds=time.perf_counter() - t0,
        dataset=dataset,
    )


def wikitext2_ids(tokenizer, split: str = "test") -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids


def model_size_bytes(model: nn.Module) -> dict[str, int]:
    """Bytes por tipo de capa: util para reportar el ahorro real."""
    quantized = 0
    dense = 0
    other = 0
    for module in model.modules():
        if isinstance(module, QuantLinear):
            quantized += module.memory_bytes()
        elif isinstance(module, nn.Linear):
            dense += module.weight.numel() * module.weight.element_size()
            if module.bias is not None:
                dense += module.bias.numel() * module.bias.element_size()
        elif isinstance(module, nn.Embedding):
            other += module.weight.numel() * module.weight.element_size()
    return {"quantized": quantized, "dense": dense, "embeddings": other,
            "total": quantized + dense + other}


@torch.no_grad()
def generation_speed(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_tokens: int = 32,
    device: torch.device | str = "cpu",
) -> float:
    """Tokens por segundo generando de forma autoregresiva (greedy)."""
    device = torch.device(device)
    ids = input_ids.to(device)
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        logits = model(ids).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return n_tokens / (time.perf_counter() - t0)
