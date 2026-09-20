"""Compara un GGUF exportado contra el modelo .tq del que salio.

Lee cada tensor del GGUF, lo descomprime con la libreria oficial y lo contrasta
con el peso equivalente en PyTorch. Si algun tensor no coincide, el problema
esta en el exportador; si todos coinciden, el problema son los metadatos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from tinyq.export.tq import load_quantized
from tinyq.quant.qlinear import QuantLinear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tq_dir")
    ap.add_argument("gguf_file")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.tq_dir)
    model = AutoModelForCausalLM.from_config(cfg)
    model = load_quantized(model, args.tq_dir)

    arch = {"qwen2": gguf.MODEL_ARCH.QWEN2, "llama": gguf.MODEL_ARCH.LLAMA}[cfg.model_type]
    name_map = gguf.TensorNameMap(arch, cfg.num_hidden_layers)

    # que espera el GGUF por cada tensor de PyTorch
    expected: dict[str, torch.Tensor] = {}
    for hf_name, mod in model.named_modules():
        if isinstance(mod, QuantLinear):
            gg = name_map.get_name(hf_name)
            if gg:
                expected[f"{gg}.weight"] = mod.dequantized_weight().float()
    for hf_name, t in model.state_dict().items():
        base, _, kind = hf_name.rpartition(".")
        if kind not in ("weight", "bias"):
            continue
        gg = name_map.get_name(base)
        if gg and f"{gg}.{kind}" not in expected:
            expected[f"{gg}.{kind}"] = t.float()

    reader = gguf.GGUFReader(args.gguf_file)
    found = {t.name: t for t in reader.tensors}

    print(f"tensores en el GGUF: {len(found)} · esperados: {len(expected)}")
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing:
        print(f"FALTAN en el GGUF ({len(missing)}): {missing[:8]}")
    if extra:
        print(f"SOBRAN en el GGUF ({len(extra)}): {extra[:8]}")

    rows = []
    for name, t in found.items():
        if name not in expected:
            continue
        data = gguf.quants.dequantize(t.data, t.tensor_type)
        got = torch.from_numpy(np.ascontiguousarray(data)).float()
        ref = expected[name].cpu()
        if got.shape != ref.shape:
            rows.append((name, float("inf"), f"forma {tuple(got.shape)} vs {tuple(ref.shape)}"))
            continue
        denom = ref.norm().clamp(min=1e-9)
        rel = ((got - ref).norm() / denom).item()
        rows.append((name, rel, str(t.tensor_type).split(".")[-1]))

    rows.sort(key=lambda r: -r[1])
    print(f"\n{'tensor':34} {'error rel':>10}  tipo")
    for name, rel, kind in rows[: args.top]:
        flag = "  <-- MAL" if rel > 0.2 else ""
        print(f"{name:34} {rel:10.5f}  {kind}{flag}")

    worst = rows[0][1] if rows else 0.0
    print(f"\npeor error relativo: {worst:.5f}")
    print("veredicto:", "tensores OK, revisar metadatos" if worst < 0.2 else "hay tensores mal escritos")


if __name__ == "__main__":
    main()
