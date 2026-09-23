"""Compara el modelo original contra el cuantizado y escribe una tabla.

    python scripts/benchmark.py Qwen/Qwen2.5-0.5B-Instruct out/qwen05b-int4 \
        --windows 10 --seqlen 512 --out runs/qwen05b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from octuma.evaluate import model_size_bytes, perplexity, wikitext2_ids
from octuma.export.tq import disk_size, load_quantized


def measure(model, ids, seq_len, windows, device, label):
    res = perplexity(
        model, ids, seq_len=seq_len, device=device, dataset="wikitext2",
        max_windows=windows,
    )
    sizes = model_size_bytes(model)
    print(f"{label}: ppl={res.perplexity:.3f}  mem={sizes['total'] / 1e9:.2f} GB")
    return {
        "perplexity": res.perplexity,
        "memory_bytes": sizes,
        "seconds": res.seconds,
        "windows": res.n_windows,
        "seq_len": res.seq_len,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_model")
    ap.add_argument("quant_dir")
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/benchmark.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    ids = wikitext2_ids(tok)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, low_cpu_mem_usage=True
    ).to(args.device).eval()
    fp = measure(base, ids, args.seqlen, args.windows, args.device, "FP32")
    del base

    cfg = AutoConfig.from_pretrained(args.quant_dir)
    q = AutoModelForCausalLM.from_config(cfg)
    q = load_quantized(q, args.quant_dir, device=args.device).to(args.device).eval()
    qt = measure(q, ids, args.seqlen, args.windows, args.device, "INT4")

    meta = json.loads((Path(args.quant_dir) / "octuma.json").read_text(encoding="utf-8"))
    payload = {
        "base_model": args.base_model,
        "quant_dir": str(args.quant_dir),
        "config": meta.get("config", {}),
        "fp": fp,
        "quant": qt,
        "delta_ppl_pct": (qt["perplexity"] / fp["perplexity"] - 1) * 100,
        "memory_ratio": fp["memory_bytes"]["total"] / max(1, qt["memory_bytes"]["total"]),
        "disk_bytes": disk_size(args.quant_dir),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("| Modelo | Formato | Memoria | Perplejidad | Perdida |")
    print("|---|---|---|---|---|")
    print(
        f"| {args.base_model} | FP32 | {fp['memory_bytes']['total'] / 1e9:.2f} GB | "
        f"{fp['perplexity']:.3f} | - |"
    )
    print(
        f"| {args.base_model} | INT{meta.get('config', {}).get('bits', 4)} | "
        f"{qt['memory_bytes']['total'] / 1e9:.2f} GB | {qt['perplexity']:.3f} | "
        f"+{payload['delta_ppl_pct']:.1f}% |"
    )
    print(f"\nReporte -> {args.out}")


if __name__ == "__main__":
    main()
