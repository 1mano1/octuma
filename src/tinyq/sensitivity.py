"""Analisis de sensibilidad por capa para elegir precision mixta.

El error que de verdad importa no es cuanto cambian los pesos, sino cuanto
cambia la salida de la capa. Con la Hessiana de calibracion eso se calcula
exacto y barato:

    ||ΔW·X||_F² = tr(ΔW · H · ΔWᵀ)   con   H = X·Xᵀ

Asi se puede comparar peras con peras entre capas de distinto tamano y decidir
cuales aguantan 4 bits y cuales necesitan 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .calibrate import CalibrationSet
from .quant.core import quantize_tensor
from .quantizer import (
    QuantConfig,
    _Catcher,
    capture_block_inputs,
    find_blocks,
    named_linears,
)
from .quant.gptq import LayerStats


@dataclass
class LayerSensitivity:
    name: str
    n_params: int
    error_by_bits: dict[int, float]  # error de salida normalizado

    def cost_of(self, bits: int) -> float:
        return self.error_by_bits[bits]

    def gain_8_over_4(self) -> float:
        """Cuanto error se evita subiendo esa capa de 4 a 8 bits."""
        return self.error_by_bits.get(4, 0.0) - self.error_by_bits.get(8, 0.0)


@dataclass
class SensitivityReport:
    layers: list[LayerSensitivity] = field(default_factory=list)

    def ranked(self) -> list[LayerSensitivity]:
        return sorted(self.layers, key=lambda l: l.gain_8_over_4(), reverse=True)

    def plan_mixed_precision(
        self, target_avg_bits: float = 4.5, high_bits: int = 8, low_bits: int = 4
    ) -> dict[str, int]:
        """Sube a `high_bits` las capas mas sensibles sin pasarse del promedio."""
        total = sum(l.n_params for l in self.layers)
        budget = (target_avg_bits - low_bits) * total / (high_bits - low_bits)
        plan: dict[str, int] = {}
        spent = 0
        for layer in self.ranked():
            if layer.gain_8_over_4() <= 0:
                break
            if spent + layer.n_params > budget:
                continue
            plan[layer.name] = high_bits
            spent += layer.n_params
        return plan

    def table(self, top: int = 15) -> list[tuple[str, float, float]]:
        return [
            (l.name, l.error_by_bits.get(4, 0.0), l.error_by_bits.get(8, 0.0))
            for l in self.ranked()[:top]
        ]


def output_error(weight: torch.Tensor, w_hat: torch.Tensor, H: torch.Tensor) -> float:
    """tr(ΔW·H·ΔWᵀ) normalizado por la energia de la salida original."""
    dw = (weight.float() - w_hat.float())
    num = torch.einsum("oi,ij,oj->", dw, H, dw)
    den = torch.einsum("oi,ij,oj->", weight.float(), H, weight.float()).clamp(min=1e-12)
    return (num / den).item()


@torch.no_grad()
def analyze_sensitivity(
    model: nn.Module,
    calib: CalibrationSet,
    bits_options: tuple[int, ...] = (4, 8),
    group_size: int = 64,
    device: torch.device | str = "cpu",
    progress=None,
) -> SensitivityReport:
    """Recorre los bloques midiendo el error de salida de cada capa por bits."""
    device = torch.device(device)
    log = progress or (lambda msg: None)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    blocks, _ = find_blocks(model)
    inputs = capture_block_inputs(model, blocks, calib, device)
    report = SensitivityReport()

    for idx, block in enumerate(blocks):
        block.to(device)
        linears = named_linears(block)
        stats = {
            n: LayerStats(in_features=m.in_features, device=device)
            for n, m in linears.items()
        }
        handles = []
        for name, lin in linears.items():
            def hook(_m, args, _o, _name=name):
                stats[_name].add_batch(args[0])

            handles.append(lin.register_forward_hook(hook))
        for args, kwargs in inputs:
            block(*args, **kwargs)
        for h in handles:
            h.remove()

        for name, lin in linears.items():
            errs = {}
            for bits in bits_options:
                qt = quantize_tensor(lin.weight.data, bits=bits, group_size=group_size)
                errs[bits] = output_error(lin.weight.data, qt.dequantize(), stats[name].H)
            report.layers.append(
                LayerSensitivity(
                    name=f"blocks.{idx}.{name}",
                    n_params=lin.weight.numel(),
                    error_by_bits=errs,
                )
            )

        new_inputs = []
        for args, kwargs in inputs:
            out = block(*args, **kwargs)
            hidden = out[0] if isinstance(out, tuple) else out
            new_inputs.append(((hidden,) + tuple(args[1:]), kwargs))
        inputs = new_inputs
        log(f"bloque {idx + 1}/{len(blocks)} analizado")

    return report


def config_from_plan(
    plan: dict[str, int], base_bits: int = 4, group_size: int = 64
) -> QuantConfig:
    """Convierte el plan en overrides por nombre de capa."""
    overrides = {name.split(".", 2)[-1]: bits for name, bits in plan.items()}
    return QuantConfig(bits=base_bits, group_size=group_size, bits_overrides=overrides)
