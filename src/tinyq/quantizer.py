"""Cuantizacion secuencial de modelos tipo decoder de Hugging Face."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from .calibrate import CalibrationSet
from .quant.core import quantize_tensor
from .quant.gptq import GPTQConfig, LayerStats, gptq_quantize
from .quant.qlinear import QuantLinear

# Capas que nunca conviene cuantizar: la de salida pesa poco y su error se
# propaga directo a los logits.
DEFAULT_SKIP = ("lm_head", "embed_out", "score", "classifier")


@dataclass
class QuantConfig:
    bits: int = 4
    group_size: int = 64
    symmetric: bool = False
    method: str = "gptq"  # "gptq" | "rtn"
    damp_percent: float = 0.01
    skip: tuple[str, ...] = DEFAULT_SKIP
    bits_overrides: dict[str, int] = field(default_factory=dict)

    def bits_for(self, layer_name: str) -> int:
        for pattern, bits in self.bits_overrides.items():
            if pattern in layer_name:
                return bits
        return self.bits


@dataclass
class LayerReport:
    name: str
    bits: int
    group_size: int
    rel_fro: float
    fp_bytes: int
    q_bytes: int
    seconds: float


@dataclass
class QuantReport:
    layers: list[LayerReport] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def fp_bytes(self) -> int:
        return sum(l.fp_bytes for l in self.layers)

    @property
    def q_bytes(self) -> int:
        return sum(l.q_bytes for l in self.layers)

    @property
    def compression(self) -> float:
        return self.fp_bytes / max(1, self.q_bytes)

    def summary(self) -> dict[str, Any]:
        worst = max(self.layers, key=lambda l: l.rel_fro) if self.layers else None
        return {
            "n_layers": len(self.layers),
            "fp_gb": self.fp_bytes / 1e9,
            "q_gb": self.q_bytes / 1e9,
            "compression": self.compression,
            "mean_rel_fro": (
                sum(l.rel_fro for l in self.layers) / len(self.layers) if self.layers else 0.0
            ),
            "worst_layer": (worst.name, worst.rel_fro) if worst else None,
            "seconds": self.seconds,
        }


def find_blocks(model: nn.Module) -> tuple[nn.ModuleList, str]:
    """Localiza la lista de bloques del transformer (llama, qwen, gpt2, ...)."""
    candidates = [
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
        "transformer.blocks",
    ]
    for path in candidates:
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj, path
    raise ValueError(
        "no se encontro la lista de bloques del modelo; agrega su ruta a find_blocks()"
    )


def named_linears(module: nn.Module) -> dict[str, nn.Linear]:
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


def _set_module(parent: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


class _Catcher(nn.Module):
    """Intercepta las entradas del primer bloque y corta la pasada."""

    class Stop(Exception):
        pass

    def __init__(self, block: nn.Module, store: list[tuple[tuple, dict]]) -> None:
        super().__init__()
        self.block = block
        self.store = store

    def forward(self, *args, **kwargs):
        self.store.append((args, kwargs))
        raise _Catcher.Stop


@torch.no_grad()
def capture_block_inputs(
    model: nn.Module, blocks: nn.ModuleList, calib: CalibrationSet, device: torch.device
) -> list[tuple[tuple, dict]]:
    """Corre el modelo hasta el primer bloque para guardar sus entradas."""
    captured: list[tuple[tuple, dict]] = []
    blocks[0] = _Catcher(blocks[0], captured)
    for batch in calib.batches:
        try:
            model(batch.to(device))
        except _Catcher.Stop:
            pass
    blocks[0] = blocks[0].block
    return captured


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    calib: CalibrationSet,
    cfg: QuantConfig | None = None,
    device: torch.device | str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> QuantReport:
    """Cuantiza el modelo en el sitio, bloque por bloque."""
    cfg = cfg or QuantConfig()
    device = torch.device(device)
    log = progress or (lambda msg: None)
    started = time.perf_counter()

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    blocks, path = find_blocks(model)
    log(f"bloques encontrados en {path}: {len(blocks)}")

    inputs = capture_block_inputs(model, blocks, calib, device)
    log(f"entradas capturadas: {len(inputs)} lotes")

    report = QuantReport()

    for idx, block in enumerate(blocks):
        block.to(device)
        linears = {
            n: m
            for n, m in named_linears(block).items()
            if not any(s in n for s in cfg.skip)
        }
        stats = {
            n: LayerStats(in_features=m.in_features, device=device)
            for n, m in linears.items()
        }

        handles = []
        for name, lin in linears.items():
            def hook(_mod, args, _out, _name=name):
                stats[_name].add_batch(args[0])

            handles.append(lin.register_forward_hook(hook, with_kwargs=False))

        for args, kwargs in inputs:
            block(*args, **kwargs)
        for h in handles:
            h.remove()

        for name, lin in linears.items():
            t0 = time.perf_counter()
            bits = cfg.bits_for(f"{idx}.{name}")
            if cfg.method == "gptq":
                qt, info = gptq_quantize(
                    lin.weight.data,
                    stats[name].H,
                    GPTQConfig(
                        bits=bits,
                        group_size=cfg.group_size,
                        symmetric=cfg.symmetric,
                        damp_percent=cfg.damp_percent,
                    ),
                )
                rel = info["rel_fro"]
            else:
                qt = quantize_tensor(
                    lin.weight.data, bits=bits, group_size=cfg.group_size,
                    symmetric=cfg.symmetric,
                )
                w = lin.weight.data.float()
                rel = ((w - qt.dequantize()).norm() / w.norm().clamp(min=1e-12)).item()

            ql = QuantLinear(
                in_features=lin.in_features,
                out_features=lin.out_features,
                bits=bits,
                group_size=qt.group_size,
                bias=lin.bias is not None,
                dtype=lin.weight.dtype,
            )
            ql.load_quantized(qt, lin.bias.data if lin.bias is not None else None)
            ql.to(device)
            _set_module(block, name, ql)

            report.layers.append(
                LayerReport(
                    name=f"blocks.{idx}.{name}",
                    bits=bits,
                    group_size=qt.group_size,
                    rel_fro=rel,
                    fp_bytes=lin.weight.numel() * 2,
                    q_bytes=qt.nbytes(),
                    seconds=time.perf_counter() - t0,
                )
            )
            del lin
        stats.clear()

        # las salidas del bloque ya cuantizado son la entrada del siguiente:
        # asi el error se acumula de forma realista en vez de ocultarse
        new_inputs = []
        for args, kwargs in inputs:
            out = block(*args, **kwargs)
            hidden = out[0] if isinstance(out, tuple) else out
            new_args = (hidden,) + tuple(args[1:])
            new_inputs.append((new_args, kwargs))
        inputs = new_inputs

        mean_rel = sum(
            l.rel_fro for l in report.layers if l.name.startswith(f"blocks.{idx}.")
        ) / max(1, len(linears))
        log(f"bloque {idx + 1}/{len(blocks)} listo · error medio {mean_rel:.4f}")
        gc.collect()

    report.seconds = time.perf_counter() - started
    return report
