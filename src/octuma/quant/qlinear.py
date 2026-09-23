"""Capa lineal con pesos cuantizados (dequantiza al vuelo)."""

from __future__ import annotations

import torch
from torch import nn

from .core import QuantizedTensor, dequantize_groupwise, quantize_tensor, unpack_bits


class QuantLinear(nn.Module):
    """Reemplazo de `nn.Linear` que guarda los pesos empaquetados.

    El forward reconstruye los pesos por bloques: es mas lento que un kernel
    dedicado, pero el objetivo aqui es la memoria y que corra en cualquier lado.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 64,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size > 0 else in_features
        n_groups = in_features // self.group_size
        per_byte = 8 // bits if bits < 8 else 1

        self.register_buffer(
            "qweight",
            torch.zeros((out_features, in_features // per_byte), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales", torch.zeros((out_features, n_groups), dtype=torch.float16)
        )
        self.register_buffer(
            "zeros", torch.zeros((out_features, n_groups), dtype=torch.uint8)
        )
        self.register_buffer("bias", torch.zeros(out_features, dtype=dtype) if bias else None)
        self._dtype = dtype

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 64,
        symmetric: bool = False,
        weight: torch.Tensor | None = None,
    ) -> QuantLinear:
        w = linear.weight.data if weight is None else weight
        qt = quantize_tensor(w, bits=bits, group_size=group_size, symmetric=symmetric)
        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            group_size=qt.group_size,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
        )
        mod.load_quantized(qt, linear.bias.data if linear.bias is not None else None)
        return mod

    def load_quantized(
        self, qt: QuantizedTensor, bias: torch.Tensor | None = None
    ) -> None:
        self.qweight = qt.qweight
        self.scales = qt.scales
        self.zeros = qt.zeros
        if bias is not None:
            self.bias = bias.to(self._dtype)

    def dequantized_weight(self) -> torch.Tensor:
        q = unpack_bits(self.qweight, self.bits, self.in_features)
        return dequantize_groupwise(
            q, self.scales, self.zeros, self.group_size, dtype=self._dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized_weight()
        return torch.nn.functional.linear(x, w.to(x.dtype), self.bias)

    def memory_bytes(self) -> int:
        total = self.qweight.numel() + self.scales.numel() * 2 + self.zeros.numel()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.bits}, group={self.group_size}"
        )
