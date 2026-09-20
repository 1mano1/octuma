"""TinyQ: cuantizacion de modelos de lenguaje para equipos modestos y Android."""

__version__ = "0.1.0"

from .quant import (
    GPTQConfig,
    QuantLinear,
    QuantizedTensor,
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
