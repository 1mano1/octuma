"""Cuantizacion con compensacion de error (estilo GPTQ).

Idea: al redondear una columna de pesos se introduce un error; en lugar de
ignorarlo, se reparte entre las columnas que aun no se cuantizan usando la
inversa de la Hessiana H = 2 * X X^T estimada con datos de calibracion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .core import pack_bits, QuantizedTensor
from .search import search_group_params


@dataclass
class GPTQConfig:
    bits: int = 4
    group_size: int = 64
    symmetric: bool = False
    damp_percent: float = 0.01
    block_size: int = 128
    search_scale: bool = True


@dataclass
class LayerStats:
    """Acumula H = X X^T con las activaciones de entrada de una capa."""

    in_features: int
    device: torch.device = torch.device("cpu")
    H: torch.Tensor = field(init=False)
    n_samples: int = 0

    def __post_init__(self) -> None:
        self.H = torch.zeros(
            (self.in_features, self.in_features), dtype=torch.float32, device=self.device
        )

    def add_batch(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, x.shape[-1]).float()
        if x.shape[0] == 0:
            return
        n = x.shape[0]
        self.H *= self.n_samples / (self.n_samples + n)
        self.n_samples += n
        x = x * (2.0 / self.n_samples) ** 0.5
        self.H += x.T @ x


def _quantize_column(
    w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, qmax: int
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.clamp(torch.round(w / scale) + zero, 0, qmax)
    return q, (q - zero) * scale


def _group_params(
    block: torch.Tensor, bits: int, symmetric: bool, search: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    if search:
        return search_group_params(block, bits, symmetric)

    qmax = (1 << bits) - 1
    if symmetric:
        absmax = block.abs().amax(dim=1, keepdim=True)
        scale = (absmax / (qmax // 2)).clamp(min=1e-8)
        zero = torch.full_like(scale, float(qmax // 2))
    else:
        vmin = block.amin(dim=1, keepdim=True)
        vmax = block.amax(dim=1, keepdim=True)
        scale = ((vmax - vmin) / qmax).clamp(min=1e-8)
        zero = torch.round(-vmin / scale).clamp(0, qmax)
    return scale, zero


def gptq_quantize(
    weight: torch.Tensor,
    H: torch.Tensor,
    cfg: GPTQConfig,
) -> tuple[QuantizedTensor, dict[str, float]]:
    """Cuantiza `weight` [out, in] usando la Hessiana `H` [in, in]."""
    w = weight.detach().float().clone()
    out_features, in_features = w.shape
    group_size = cfg.group_size if cfg.group_size > 0 else in_features
    qmax = (1 << cfg.bits) - 1

    H = H.clone().to(w.device)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    w[:, dead] = 0.0

    damp = cfg.damp_percent * torch.mean(torch.diag(H))
    H[range(in_features), range(in_features)] += damp

    # Hinv superior triangular via Cholesky (metodo del paper de GPTQ)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(w)
    n_groups = in_features // group_size
    # mismo dispositivo que los pesos: si no, falla en GPU
    scales = torch.zeros((out_features, n_groups), dtype=torch.float32, device=w.device)
    zeros = torch.zeros((out_features, n_groups), dtype=torch.float32, device=w.device)
    qint = torch.zeros_like(w, dtype=torch.uint8)
    total_err = 0.0

    for start in range(0, in_features, cfg.block_size):
        end = min(start + cfg.block_size, in_features)
        W1 = w[:, start:end].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[start:end, start:end]

        for i in range(end - start):
            col = start + i
            if col % group_size == 0:
                g = col // group_size
                block = w[:, col : col + group_size]
                scale, zero = _group_params(
                    block, cfg.bits, cfg.symmetric, search=cfg.search_scale
                )
                scales[:, g] = scale.squeeze(1)
                zeros[:, g] = zero.squeeze(1)

            g = col // group_size
            scale = scales[:, g : g + 1]
            zero = zeros[:, g : g + 1]
            wc = W1[:, i : i + 1]
            q, deq = _quantize_column(wc, scale, zero, qmax)
            qint[:, col] = q.squeeze(1).to(torch.uint8)
            Q1[:, i : i + 1] = deq

            d = Hinv1[i, i]
            err = (wc - deq) / d
            total_err += (err**2).sum().item()
            if i + 1 < end - start:
                W1[:, i + 1 :] -= err @ Hinv1[i, i + 1 :].unsqueeze(0)
            E1[:, i : i + 1] = err

        Q[:, start:end] = Q1
        if end < in_features:
            w[:, end:] -= E1 @ Hinv[start:end, end:]

    qt = QuantizedTensor(
        qweight=pack_bits(qint, cfg.bits),
        scales=scales.half(),
        zeros=zeros.to(torch.uint8),
        bits=cfg.bits,
        group_size=group_size,
        in_features=in_features,
        symmetric=cfg.symmetric,
    )
    stats = {
        "gptq_err": total_err,
        "rel_fro": (
            (weight.float() - qt.dequantize()).norm() / weight.float().norm().clamp(min=1e-12)
        ).item(),
    }
    return qt, stats

