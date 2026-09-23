"""Diagnostico: ¿AWQ reduce el error de salida por capa, o estorba?

Compara, con las MISMAS activaciones reales de calibracion, el error de salida
de cada capa lineal de un bloque:

    sin AWQ:  cuantizar W
    con AWQ:  cuantizar W*diag(s) y dividir por s

Si AWQ no gana aqui, no va a ganar en perplejidad.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from octuma.calibrate import load_calibration
from octuma.quant.awq import llama_like_groups, search_scales
from octuma.quant.core import quantize_tensor
from octuma.quantizer import capture_block_inputs, find_blocks, named_linears


def layer_error(w: torch.Tensor, x: torch.Tensor, bits: int, group: int,
                scales: torch.Tensor | None = None) -> float:
    ref = x @ w.T
    if scales is None:
        w_hat = quantize_tensor(w, bits, group).dequantize()
    else:
        ws = w * scales.unsqueeze(0)
        w_hat = quantize_tensor(ws, bits, group).dequantize() / scales.unsqueeze(0)
    return ((x @ w_hat.T - ref).norm() / ref.norm()).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=3, help="cuantos bloques revisar")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--groups", type=int, nargs="+", default=[32, 64, 128, 0])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    cal = load_calibration("wikitext2", tok, n_samples=args.samples, seq_len=args.seqlen)

    blocks, _ = find_blocks(model)
    inputs = capture_block_inputs(model, blocks, cal, torch.device("cpu"))
    cfg = model.config

    totals = {g: [0.0, 0.0] for g in args.groups}

    for idx in range(min(args.blocks, len(blocks))):
        block = blocks[idx]
        linears = named_linears(block)
        captured: dict[str, list[torch.Tensor]] = {n: [] for n in linears}
        handles = [
            lin.register_forward_hook(
                lambda _m, a, _o, _n=n: captured[_n].append(
                    a[0].detach().reshape(-1, a[0].shape[-1])[:512].float()
                )
            )
            for n, lin in linears.items()
        ]
        with torch.no_grad():
            for a, k in inputs:
                block(*a, **k)
        for h in handles:
            h.remove()
        x_of = {n: torch.cat(v)[:1024] for n, v in captured.items() if v}

        groups = llama_like_groups(block, cfg.num_attention_heads, cfg.num_key_value_heads)
        for group_size in args.groups:
            print(f"\n== bloque {idx} · grupo {group_size or 'por fila'} ==")
            for g in groups:
                x = x_of[[n for n, m in linears.items() if m is g.layers[0]][0]]
                scales, alpha, _ = search_scales(
                    g.layers, x, bits=args.bits, group_size=group_size
                )
                for lin in g.layers:
                    name = [n for n, m in linears.items() if m is lin][0]
                    xi = x_of[name]
                    w = lin.weight.data.float()
                    e0 = layer_error(w, xi, args.bits, group_size)
                    e1 = layer_error(w, xi, args.bits, group_size, scales)
                    totals[group_size][0] += e0
                    totals[group_size][1] += e1
                    flag = "mejor" if e1 < e0 else "PEOR "
                    print(
                        f"  {name:28} a={alpha:.2f}  sin={e0:.5f}  con={e1:.5f}  {flag}"
                    )

    print("\n== resumen (suma de errores) ==")
    for g, (e0, e1) in totals.items():
        delta = (e1 - e0) / e0 * 100
        print(
            f"  grupo {str(g or 'fila'):5}: sin AWQ {e0:.4f} · con AWQ {e1:.4f} "
            f"({delta:+.1f}%)"
        )


if __name__ == "__main__":
    main()
