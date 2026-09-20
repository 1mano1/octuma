"""¿Por que AWQ empeora a GPTQ? Se prueba la correccion de la Hessiana.

Tras AWQ los pesos son W' = W·diag(s) y la entrada pasa a ser x' = x/s.
La Hessiana correcta para GPTQ es entonces D^-1·H·D^-1. Aqui se compara esa
correccion contra las alternativas equivocadas, midiendo el error de salida
real de la capa.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyq.calibrate import load_calibration
from tinyq.quant.awq import llama_like_groups, search_scales
from tinyq.quant.core import quantize_tensor
from tinyq.quant.gptq import GPTQConfig, LayerStats, gptq_quantize
from tinyq.quantizer import capture_block_inputs, find_blocks, named_linears


def err(x: torch.Tensor, w_ref: torch.Tensor, w_hat: torch.Tensor) -> float:
    ref = x @ w_ref.T
    return ((x @ w_hat.T - ref).norm() / ref.norm()).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    cal = load_calibration("wikitext2", tok, n_samples=args.samples, seq_len=args.seqlen)
    blocks, _ = find_blocks(model)
    inputs = capture_block_inputs(model, blocks, cal, torch.device("cpu"))

    block = blocks[1]
    linears = named_linears(block)
    caught: dict[str, list[torch.Tensor]] = {n: [] for n in linears}
    stats = {n: LayerStats(in_features=m.in_features) for n, m in linears.items()}
    handles = []
    for n, lin in linears.items():
        def hook(_m, a, _o, _n=n):
            x = a[0].detach().reshape(-1, a[0].shape[-1]).float()
            caught[_n].append(x[:512])
            stats[_n].add_batch(x)

        handles.append(lin.register_forward_hook(hook))
    with torch.no_grad():
        for a, k in inputs:
            block(*a, **k)
    for h in handles:
        h.remove()

    cfg = model.config
    groups = llama_like_groups(block, cfg.num_attention_heads, cfg.num_key_value_heads)
    gcfg = GPTQConfig(bits=args.bits, group_size=args.group)

    print(f"{'capa':26} {'gptq':>8} {'awq+rtn':>9} {'D^-1HD^-1':>10} {'H sin tocar':>12} {'DHD':>8}")
    for g in groups:
        first = [n for n, m in linears.items() if m is g.layers[0]][0]
        x = torch.cat(caught[first])[:1024]
        scales, _, _ = search_scales(g.layers, x, bits=args.bits, group_size=args.group)

        for lin in g.layers:
            name = [n for n, m in linears.items() if m is lin][0]
            xi = torch.cat(caught[name])[:1024]
            w = lin.weight.data.float()
            H = stats[name].H

            # referencia: GPTQ puro
            qt, _ = gptq_quantize(w, H, gcfg)
            e_gptq = err(xi, w, qt.dequantize())

            # AWQ + RTN
            ws = w * scales.unsqueeze(0)
            w_hat = quantize_tensor(ws, args.bits, args.group).dequantize() / scales.unsqueeze(0)
            e_awq_rtn = err(xi, w, w_hat)

            # AWQ + GPTQ con cada version de la Hessiana
            xs = xi / scales.unsqueeze(0)
            results = {}
            inv = 1.0 / scales
            for label, Hx in (
                ("inv", H * inv.unsqueeze(0) * inv.unsqueeze(1)),
                ("orig", H),
                ("dir", H * scales.unsqueeze(0) * scales.unsqueeze(1)),
            ):
                qts, _ = gptq_quantize(ws, Hx, gcfg)
                results[label] = err(xi, w, qts.dequantize() / scales.unsqueeze(0))

            print(
                f"{name:26} {e_gptq:8.5f} {e_awq_rtn:9.5f} "
                f"{results['inv']:10.5f} {results['orig']:12.5f} {results['dir']:8.5f}"
            )


if __name__ == "__main__":
    main()
