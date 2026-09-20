"""Matematica de cuantizacion por grupos (INT2..INT8) y empaquetado de bits."""

from __future__ import annotations

from dataclasses import dataclass

import torch

SUPPORTED_BITS = (2, 3, 4, 8)


@dataclass
class QuantizedTensor:
    """Pesos cuantizados por grupos a lo largo de la dimension de entrada."""

    qweight: torch.Tensor  # uint8 empaquetado, [out, in * bits / 8]
    scales: torch.Tensor  # float16, [out, n_groups]
    zeros: torch.Tensor  # uint8, [out, n_groups]
    bits: int
    group_size: int
    in_features: int
    symmetric: bool = False

    @property
    def n_groups(self) -> int:
        return self.scales.shape[1]

    def nbytes(self) -> int:
        return self.qweight.numel() + self.scales.numel() * 2 + self.zeros.numel()

    def dequantize(self) -> torch.Tensor:
        q = unpack_bits(self.qweight, self.bits, self.in_features)
        return dequantize_groupwise(q, self.scales, self.zeros, self.group_size)


def _grouped(w: torch.Tensor, group_size: int) -> torch.Tensor:
    out_features, in_features = w.shape
    if in_features % group_size:
        raise ValueError(
            f"in_features={in_features} no es multiplo de group_size={group_size}"
        )
    return w.reshape(out_features, in_features // group_size, group_size)


def quantize_groupwise(
    w: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
    symmetric: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Devuelve (q, scales, zeros) con q en [0, 2**bits - 1] y forma igual a w."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits={bits} no soportado, usa {SUPPORTED_BITS}")
    if group_size <= 0:
        group_size = w.shape[1]

    g = _grouped(w.float(), group_size)
    qmax = (1 << bits) - 1

    if symmetric:
        absmax = g.abs().amax(dim=-1, keepdim=True)
        scales = (absmax / (qmax // 2)).clamp(min=1e-8)
        zeros = torch.full_like(scales, float(qmax // 2))
    else:
        vmin = g.amin(dim=-1, keepdim=True)
        vmax = g.amax(dim=-1, keepdim=True)
        scales = ((vmax - vmin) / qmax).clamp(min=1e-8)
        zeros = torch.round(-vmin / scales).clamp(0, qmax)

    q = torch.clamp(torch.round(g / scales) + zeros, 0, qmax)
    q = q.reshape(w.shape).to(torch.uint8)
    return q, scales.squeeze(-1).half(), zeros.squeeze(-1).to(torch.uint8)


def dequantize_groupwise(
    q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruye los pesos a partir de q, scales y zeros."""
    out_features, in_features = q.shape
    g = q.reshape(out_features, in_features // group_size, group_size).to(torch.float32)
    s = scales.to(torch.float32).unsqueeze(-1)
    z = zeros.to(torch.float32).unsqueeze(-1)
    return ((g - z) * s).reshape(out_features, in_features).to(dtype)


def pack_bits(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Empaqueta enteros sin signo de `bits` en un tensor uint8 contiguo."""
    if bits == 8:
        return q.contiguous().to(torch.uint8)
    if 8 % bits:
        raise ValueError(f"bits={bits} no divide a 8; usa 2, 4 u 8")

    per_byte = 8 // bits
    out_features, in_features = q.shape
    if in_features % per_byte:
        raise ValueError(f"in_features={in_features} no es multiplo de {per_byte}")

    v = q.to(torch.uint8).reshape(out_features, in_features // per_byte, per_byte)
    packed = torch.zeros(v.shape[:2], dtype=torch.uint8, device=q.device)
    for i in range(per_byte):
        packed |= v[..., i] << (bits * i)
    return packed


def unpack_bits(packed: torch.Tensor, bits: int, in_features: int) -> torch.Tensor:
    """Operacion inversa de `pack_bits`."""
    if bits == 8:
        return packed.to(torch.uint8)

    per_byte = 8 // bits
    mask = (1 << bits) - 1
    cols = [(packed >> (bits * i)) & mask for i in range(per_byte)]
    v = torch.stack(cols, dim=-1)
    return v.reshape(packed.shape[0], -1)[:, :in_features].to(torch.uint8)


def quantize_tensor(
    w: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
    symmetric: bool = False,
) -> QuantizedTensor:
    """Cuantiza y empaqueta una matriz de pesos [out_features, in_features]."""
    q, scales, zeros = quantize_groupwise(w, bits, group_size, symmetric)
    return QuantizedTensor(
        qweight=pack_bits(q, bits),
        scales=scales,
        zeros=zeros,
        bits=bits,
        group_size=group_size if group_size > 0 else w.shape[1],
        in_features=w.shape[1],
        symmetric=symmetric,
    )


def quantization_error(w: torch.Tensor, w_hat: torch.Tensor) -> dict[str, float]:
    """Metricas de error entre los pesos originales y los reconstruidos."""
    diff = (w.float() - w_hat.float()).abs()
    denom = w.float().norm().clamp(min=1e-12)
    return {
        "mae": diff.mean().item(),
        "max": diff.max().item(),
        "rel_fro": (diff.norm() / denom).item(),
    }
