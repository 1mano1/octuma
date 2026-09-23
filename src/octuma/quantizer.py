"""Cuantizacion secuencial de modelos tipo decoder de Hugging Face."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .calibrate import CalibrationSet
from .quant.awq import awq_scale_block
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
    awq: bool = False
    awq_samples: int = 1024
    search_scale: bool = True

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
        return sum(capa.fp_bytes for capa in self.layers)

    @property
    def q_bytes(self) -> int:
        return sum(capa.q_bytes for capa in self.layers)

    @property
    def compression(self) -> float:
        return self.fp_bytes / max(1, self.q_bytes)

    def summary(self) -> dict[str, Any]:
        worst = max(self.layers, key=lambda c: c.rel_fro) if self.layers else None
        return {
            "n_layers": len(self.layers),
            "fp_gb": self.fp_bytes / 1e9,
            "q_gb": self.q_bytes / 1e9,
            "compression": self.compression,
            "mean_rel_fro": (
                sum(capa.rel_fro for capa in self.layers) / len(self.layers) if self.layers else 0.0
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

    # GPTQ estima H = X·Xt: con menos tokens que dimensiones la matriz es
    # singular y la compensacion de error se vuelve ruido. Pasa callado y da
    # resultados peores que no usar GPTQ, asi que conviene avisar.
    if cfg.method == "gptq":
        widest = max(m.in_features for m in named_linears(blocks[0]).values())
        if calib.n_tokens < widest:
            log(
                f"AVISO: {calib.n_tokens} tokens de calibracion para capas de "
                f"hasta {widest} dimensiones. GPTQ necesita bastantes mas "
                f"(idealmente 10x) o el resultado sera peor que RTN."
            )

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

        samples: dict[str, list[torch.Tensor]] = {n: [] for n in linears}
        handles = []
        for name, lin in linears.items():
            def hook(_mod, args, _out, _name=name):
                x = args[0]
                stats[_name].add_batch(x)
                if cfg.awq:
                    flat = x.detach().reshape(-1, x.shape[-1])
                    keep = min(flat.shape[0], cfg.awq_samples)
                    samples[_name].append(flat[:keep].float().cpu())

            handles.append(lin.register_forward_hook(hook, with_kwargs=False))

        for args, kwargs in inputs:
            block(*args, **kwargs)
        for h in handles:
            h.remove()

        if cfg.awq:
            hf_cfg = getattr(model, "config", None)
            n_heads = getattr(hf_cfg, "num_attention_heads", 1)
            n_kv = getattr(hf_cfg, "num_key_value_heads", n_heads)
            inputs_by_layer = {
                linears[n]: torch.cat(v)[: cfg.awq_samples]
                for n, v in samples.items()
                if v
            }
            awq_report, applied = awq_scale_block(
                block, inputs_by_layer, n_heads, n_kv,
                bits=cfg.bits, group_size=cfg.group_size,
            )
            # al plegar 1/s en la capa previa, la entrada de estas capas queda
            # dividida por s, asi que la Hessiana se corrige analiticamente en
            # vez de volver a correr el bloque
            for name, lin in linears.items():
                s = applied.get(lin)
                if s is not None:
                    inv = (1.0 / s).to(stats[name].H.device)
                    stats[name].H = stats[name].H * inv.unsqueeze(0) * inv.unsqueeze(1)
            log(
                "  awq: "
                + ", ".join(f"{r['group']} a={r['alpha']:.2f}" for r in awq_report)
            )
        samples.clear()

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
                        search_scale=cfg.search_scale,
                    ),
                )
                rel = info["rel_fro"]
            else:
                qt = quantize_tensor(
                    lin.weight.data, bits=bits, group_size=cfg.group_size,
                    symmetric=cfg.symmetric, search=cfg.search_scale,
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
            capa.rel_fro for capa in report.layers if capa.name.startswith(f"blocks.{idx}.")
        ) / max(1, len(linears))
        log(f"bloque {idx + 1}/{len(blocks)} listo · error medio {mean_rel:.4f}")
        gc.collect()

    report.seconds = time.perf_counter() - started
    return report
