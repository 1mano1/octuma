"""Compara TinyQ contra cuantizadores de terceros en igualdad de condiciones.

Todo lo de este repo se habia medido contra si mismo: se sabia que metodo de
TinyQ era el mejor, pero no si TinyQ compite con lo que ya existe. Aqui se
cuantiza el MISMO modelo con otra herramienta y se evalua con el MISMO
evaluador (misma prueba, mismas ventanas, mismo corpus), que es la unica forma
de que los numeros se puedan poner en la misma tabla.

    python scripts/compare_baselines.py --model Qwen/Qwen2.5-3B-Instruct

Los resultados caen en runs/baseline__<herramienta>__<modelo>.json con la
misma forma que el resto del barrido.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RUNS = Path("runs")


def medir(model, tok, short: str, etiqueta: str, windows: int, seq_len: int) -> dict:
    from tinyq.evaluate import model_size_bytes, perplexity, wikitext2_ids

    ids = wikitext2_ids(tok)
    res = perplexity(
        model, ids, seq_len=seq_len, device="cuda",
        dataset="wikitext2", max_windows=windows,
    )
    mem = model_size_bytes(model)
    print(f"  {etiqueta}: ppl={res.perplexity:.4f}  ({mem['total'] / 1e9:.2f} GB)")
    return {
        "id": f"baseline__{etiqueta}__{short}",
        "tool": etiqueta,
        "short": short,
        "perplexity": res.perplexity,
        "memory_bytes": mem,
        "eval": {"windows": windows, "seq_len": seq_len, "dataset": "wikitext2"},
        "eval_seconds": res.seconds,
    }


def cargar_fp16(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": "cuda"},
    ).eval()
    return model, tok


def cargar_bnb(model_id: str, tipo: str):
    """NF4 o FP4 de bitsandbytes, el cuantizador por defecto de Hugging Face."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=tipo,
        bnb_4bit_compute_dtype=torch.float16,
        # doble cuantizacion: comprime tambien las escalas, como hace QLoRA
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=cfg, low_cpu_mem_usage=True, device_map={"": "cuda"},
    ).eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--short", default="")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()

    short = args.short or args.model.split("/")[-1].split("-")[1].lower()
    RUNS.mkdir(exist_ok=True)

    pruebas = [
        ("bnb-nf4", lambda: cargar_bnb(args.model, "nf4")),
        ("bnb-fp4", lambda: cargar_bnb(args.model, "fp4")),
    ]

    for etiqueta, cargar in pruebas:
        destino = RUNS / f"baseline__{etiqueta}__{short}.json"
        if destino.exists():
            print(f"[skip] {etiqueta}")
            continue
        print(f"[run ] {etiqueta} sobre {args.model}")
        t0 = time.perf_counter()
        try:
            model, tok = cargar()
            payload = medir(model, tok, short, etiqueta, args.windows, args.seq_len)
            payload["total_seconds"] = time.perf_counter() - t0
            destino.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"       FALLO: {type(exc).__name__}: {exc}")

    print("listo")


if __name__ == "__main__":
    main()
