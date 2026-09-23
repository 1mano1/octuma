"""Octuma: cuantizacion de modelos de lenguaje para equipos modestos y Android."""

__version__ = "0.1.2"

from .quant import (
    GPTQConfig,
    QuantizedTensor,
    QuantLinear,
    gptq_quantize,
    quantize_tensor,
)

__all__ = [
    "GPTQConfig",
    "QuantLinear",
    "QuantizedTensor",
    "__version__",
    "gptq_quantize",
    "quantize_tensor",
]
