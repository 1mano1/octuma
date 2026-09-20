"""Corredor de experimentos: cuantiza, evalua y guarda cada combinacion.

    python scripts/run_sweep.py experiments/sweep.yaml --only qwen0.5b
    python scripts/run_sweep.py experiments/sweep.yaml --ablations
    python scripts/run_sweep.py --table          # tabla con lo ya corrido

Cada corrida se guarda en runs/<id>.json y se salta si ya existe, asi que el
barrido se puede interrumpir y retomar sin perder trabajo (util en RunPod).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import yaml

RUNS = Path("runs")


def load_model(model_id: str, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    return model, tok


def eval_tag(d: dict) -> str:
    """La configuracion de evaluacion va en el nombre del archivo.

    Sin esto, una corrida evaluada con 4 ventanas se compara contra una base
    evaluada con 2 y el porcentaje resultante no significa nada.
    """
    return f"w{d['eval_windows']}s{d['eval_seq_len']}"


def baseline_id(short: str, d: dict) -> str:
    return f"{short}__fp16__{eval_tag(d)}"


def run_baseline(model_cfg: dict, d: dict) -> dict:
    from tinyq.evaluate import model_size_bytes, perplexity, wikitext2_ids

    rid = baseline_id(model_cfg["short"], d)
    path = RUNS / f"{rid}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    model, tok = load_model(model_cfg["id"], d["device"], d["dtype"])
    ids = wikitext2_ids(tok)
    res = perplexity(
        model, ids, seq_len=d["eval_seq_len"], device=d["device"],
        dataset="wikitext2", max_windows=d["eval_windows"],
    )
    payload = {
        "id": rid,
        "model": model_cfg["id"],
        "short": model_cfg["short"],
        "method": "fp16",
        "perplexity": res.perplexity,
        "memory_bytes": model_size_bytes(model),
        "eval_seconds": res.seconds,
        "eval": {"windows": d["eval_windows"], "seq_len": d["eval_seq_len"]},
    }
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"[base] {rid}: ppl={res.perplexity:.3f}")
    return payload


def run_one(model_cfg: dict, method: dict, d: dict, tag: str = "") -> dict | None:
    from tinyq.calibrate import load_calibration
    from tinyq.evaluate import model_size_bytes, perplexity, wikitext2_ids
    from tinyq.quantizer import QuantConfig, quantize_model

    rid = f"{model_cfg['short']}__{method['name']}{tag}__{eval_tag(d)}"
    path = RUNS / f"{rid}.json"
    if path.exists():
        print(f"[skip] {rid}")
        return json.loads(path.read_text(encoding="utf-8"))

    print(f"[run ] {rid}")
    t0 = time.perf_counter()
    try:
        model, tok = load_model(model_cfg["id"], d["device"], d["dtype"])
        cal = load_calibration(
            d["calib"], tok,
            n_samples=method.get("calib_samples", d["calib_samples"]),
            seq_len=method.get("calib_seq_len", d["calib_seq_len"]),
        )
        cfg = QuantConfig(
            bits=method["bits"],
            group_size=method["group_size"],
            method=method.get("method", "gptq"),
            awq=method.get("awq", False),
            search_scale=method.get("search_scale", True),
        )
        report = quantize_model(model, cal, cfg, device=d["device"])

        ids = wikitext2_ids(tok)
        res = perplexity(
            model, ids, seq_len=d["eval_seq_len"], device=d["device"],
            dataset="wikitext2", max_windows=d["eval_windows"],
        )
        payload = {
            "id": rid,
            "model": model_cfg["id"],
            "short": model_cfg["short"],
            "method": method["name"],
            "config": {
                "bits": cfg.bits, "group_size": cfg.group_size,
                "method": cfg.method, "awq": cfg.awq,
                "search_scale": cfg.search_scale,
                "calib_samples": method.get("calib_samples", d["calib_samples"]),
                "calib_seq_len": method.get("calib_seq_len", d["calib_seq_len"]),
            },
            "perplexity": res.perplexity,
            "memory_bytes": model_size_bytes(model),
            "eval": {"windows": d["eval_windows"], "seq_len": d["eval_seq_len"]},
            "quant_summary": report.summary(),
            "total_seconds": time.perf_counter() - t0,
        }
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"       ppl={res.perplexity:.3f}  ({payload['total_seconds'] / 60:.1f} min)")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return payload
    except Exception as exc:  # una corrida rota no debe tumbar el barrido
        err = RUNS / f"{rid}.error.txt"
        err.parent.mkdir(exist_ok=True)
        err.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"       FALLO: {exc}")
        return None


def build_table() -> str:
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "method" in r and "short" in r:  # ignora archivos de otro formato
            rows.append(r)

    def key(r: dict) -> tuple:
        ev = r.get("eval", {})
        return (r["short"], ev.get("windows"), ev.get("seq_len"))

    # la base solo vale si se evaluo con la misma configuracion
    base = {key(r): r for r in rows if r["method"] == "fp16"}

    lines = ["| Modelo | Metodo | Bits | Grupo | Memoria | Perplejidad | vs FP16 |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["short"], x["method"])):
        cfg = r.get("config", {})
        mem = r["memory_bytes"]["total"] / 1e9
        b = base.get(key(r))
        if r["method"] == "fp16":
            delta = "-"
        elif b:
            delta = f"{(r['perplexity'] / b['perplexity'] - 1) * 100:+.1f}%"
        else:
            delta = "sin base comparable"
        lines.append(
            f"| {r['short']} | {r['method']} | {cfg.get('bits', 16)} | "
            f"{cfg.get('group_size', '-')} | {mem:.2f} GB | "
            f"{r['perplexity']:.3f} | {delta} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="experiments/sweep.yaml")
    ap.add_argument("--only", help="Corre solo este modelo (short)")
    ap.add_argument("--ablations", action="store_true", help="Corre las ablaciones")
    ap.add_argument("--table", action="store_true", help="Solo imprime la tabla")
    ap.add_argument("--device", help="Sobrescribe el device del yaml")
    args = ap.parse_args()

    if args.table:
        print(build_table())
        return

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    d = cfg["defaults"]
    if args.device:
        d["device"] = args.device

    models = cfg["models"]
    if args.only:
        models = [m for m in models if m["short"] == args.only]

    for model_cfg in models:
        run_baseline(model_cfg, d)
        for method in cfg["methods"]:
            run_one(model_cfg, method, d)

    if args.ablations:
        abl = cfg["ablations"]
        for model_cfg in models:
            if model_cfg["short"] not in abl["only_models"]:
                continue
            for g in abl["group_size"]:
                run_one(
                    model_cfg,
                    {"name": f"gptq-int4-g{g}", "bits": 4, "group_size": g},
                    d,
                )
            for b in abl["bits"]:
                run_one(
                    model_cfg,
                    {"name": f"gptq-int{b}-g64", "bits": b, "group_size": 64},
                    d,
                )
            for n in abl["calib_samples"]:
                run_one(
                    model_cfg,
                    {"name": f"gptq-int4-calib{n}", "bits": 4, "group_size": 64,
                     "calib_samples": n},
                    d,
                )

    print()
    print(build_table())


if __name__ == "__main__":
    main()
