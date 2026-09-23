"""Formato nativo .tq: safetensors con los pesos empaquetados + metadatos."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from ..quant.qlinear import QuantLinear

FORMAT_VERSION = 1


def _quant_layers(model: nn.Module) -> dict[str, QuantLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, QuantLinear)}


def save_quantized(
    model: nn.Module,
    out_dir: str | Path,
    cfg: Any = None,
    extra: dict[str, Any] | None = None,
    fp16_dense: bool = True,
) -> Path:
    """Guarda pesos cuantizados + los densos restantes en `out_dir`.

    `fp16_dense` baja a FP16 lo que no se cuantiza (embeddings y normas), que
    es donde se va la mayor parte del archivo en modelos con vocabulario grande.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    layers_meta: dict[str, dict[str, int]] = {}

    qlayers = _quant_layers(model)
    for name, mod in qlayers.items():
        tensors[f"{name}.qweight"] = mod.qweight.contiguous()
        tensors[f"{name}.scales"] = mod.scales.contiguous()
        tensors[f"{name}.zeros"] = mod.zeros.contiguous()
        if mod.bias is not None:
            tensors[f"{name}.bias"] = mod.bias.contiguous()
        layers_meta[name] = {
            "bits": mod.bits,
            "group_size": mod.group_size,
            "in_features": mod.in_features,
            "out_features": mod.out_features,
            "bias": mod.bias is not None,
        }

    qnames = set(qlayers)
    # Muchos modelos atan lm_head a los embeddings: safetensors rechaza dos
    # nombres que apuntan a la misma memoria, asi que se guarda una sola copia.
    seen: dict[tuple[int, int], str] = {}
    tied: dict[str, str] = {}
    for name, param in model.state_dict().items():
        owner = name.rsplit(".", 1)[0]
        if owner in qnames:
            continue
        key = (param.data_ptr(), param.numel())
        if key in seen:
            tied[name] = seen[key]
            continue
        seen[key] = name
        if fp16_dense and param.dtype == torch.float32:
            param = param.half()
        tensors[name] = param.contiguous()

    save_file(tensors, out / "model.tq.safetensors")

    meta = {
        "format_version": FORMAT_VERSION,
        "layers": layers_meta,
        "tied": tied,
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else (cfg or {}),
        "model_type": getattr(getattr(model, "config", None), "model_type", "unknown"),
        **(extra or {}),
    }
    (out / "octuma.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_quantized(model: nn.Module, in_dir: str | Path, device: str = "cpu") -> nn.Module:
    """Reconstruye un modelo ya cuantizado sobre un esqueleto de HF."""
    path = Path(in_dir)
    meta = json.loads((path / "octuma.json").read_text(encoding="utf-8"))
    tensors = load_file(str(path / "model.tq.safetensors"), device=device)

    for name, spec in meta["layers"].items():
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        ql = QuantLinear(
            in_features=spec["in_features"],
            out_features=spec["out_features"],
            bits=spec["bits"],
            group_size=spec["group_size"],
            bias=spec["bias"],
            dtype=getattr(old, "weight", torch.zeros(1)).dtype
            if hasattr(old, "weight")
            else torch.float32,
        )
        ql.qweight = tensors[f"{name}.qweight"]
        ql.scales = tensors[f"{name}.scales"]
        ql.zeros = tensors[f"{name}.zeros"]
        if spec["bias"]:
            ql.bias = tensors[f"{name}.bias"]
        setattr(parent, parts[-1], ql)

    dense = {
        k: v
        for k, v in tensors.items()
        if not any(k.startswith(f"{n}.") for n in meta["layers"])
    }
    for dup, canonical in meta.get("tied", {}).items():
        dense[dup] = dense[canonical]

    missing, unexpected = model.load_state_dict(dense, strict=False)
    if unexpected:
        raise RuntimeError(f"tensores inesperados en el archivo: {unexpected[:5]}")
    return model


def disk_size(out_dir: str | Path) -> int:
    return sum(f.stat().st_size for f in Path(out_dir).glob("*") if f.is_file())
