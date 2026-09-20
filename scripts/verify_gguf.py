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


def revisar_metadatos(gguf_file: Path, tq_dir: Path) -> list[str]:
    """Contrasta los metadatos del GGUF con la configuracion del modelo.

    Los pesos pueden estar perfectos y el modelo salir roto igual: un
    `rope_theta` de 10000 donde el modelo usa 1000000, o un pre-tokenizador
    equivocado, duplican la perplejidad sin tocar un solo peso. Esta funcion
    existe porque ese bug llego a estar publicado sin que nadie lo notara.
    """
    import json

    reader = gguf.GGUFReader(str(gguf_file))
    campos = reader.fields
    cfg = json.loads((tq_dir / "config.json").read_text(encoding="utf-8"))
    arch = campos["general.architecture"].contents()
    fallos: list[str] = []

    def leer(clave):
        campo = campos.get(clave)
        return campo.contents() if campo else None

    # desde transformers 5 el valor vive dentro de rope_parameters
    esperado = cfg.get("rope_theta") or (cfg.get("rope_parameters") or {}).get("rope_theta")
    escrito = leer(f"{arch}.rope.freq_base")
    if esperado and escrito and abs(float(escrito) - float(esperado)) > 1:
        fallos.append(f"rope.freq_base={escrito} pero el modelo usa {esperado}")

    pre = leer("tokenizer.ggml.pre")
    if arch == "qwen2" and pre != "qwen2":
        fallos.append(f"tokenizer.pre='{pre}' cuando qwen2 necesita 'qwen2'")

    if leer("tokenizer.ggml.eos_token_id") is None:
        fallos.append("falta tokenizer.eos_token_id: el modelo no sabra cuando parar")

    for clave, campo_cfg in (
        (f"{arch}.block_count", "num_hidden_layers"),
        (f"{arch}.embedding_length", "hidden_size"),
        (f"{arch}.attention.head_count", "num_attention_heads"),
        (f"{arch}.attention.head_count_kv", "num_key_value_heads"),
        (f"{arch}.feed_forward_length", "intermediate_size"),
    ):
        v, esperado_v = leer(clave), cfg.get(campo_cfg)
        if esperado_v is not None and v is not None and int(v) != int(esperado_v):
            fallos.append(f"{clave}={v} pero config dice {esperado_v}")

    print()
    print(f"{'metadato':34} valor")
    for clave in (f"{arch}.rope.freq_base", "tokenizer.ggml.pre",
                  "tokenizer.ggml.eos_token_id", f"{arch}.block_count"):
        print(f"{clave:34} {leer(clave)}")
    return fallos


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

    fallos = revisar_metadatos(Path(args.gguf_file), Path(args.tq_dir))
    if fallos:
        print()
        print("METADATOS MAL:")
        for f in fallos:
            print(f"  - {f}")

    print()
    if worst >= 0.2:
        print("veredicto: hay tensores mal escritos")
        raise SystemExit(1)
    if fallos:
        print("veredicto: los pesos estan bien, pero los metadatos rompen el modelo")
        raise SystemExit(1)
    print("veredicto: pesos y metadatos correctos")


if __name__ == "__main__":
    main()
