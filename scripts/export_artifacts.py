"""Genera los modelos cuantizados finales y los exporta a GGUF.

El barrido solo mide y descarta. Esto produce lo que de verdad se usa y se
publica: el modelo en formato .tq y el .gguf para llama.cpp y Android.

Se usa grupo 32 porque es lo que exige Q4_1 de GGUF, y GPTQ + AWQ, que es la
combinacion que gana en el barrido.

    python scripts/export_artifacts.py                 # todos los modelos
    python scripts/export_artifacts.py --only qwen0.5b
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import yaml

OUT = Path("out")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="experiments/sweep.yaml")
    ap.add_argument("--only", help="un modelo por su nombre corto")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=32, help="32 = compatible con GGUF")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--no-gguf", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tinyq.calibrate import load_calibration
    from tinyq.export.gguf_export import export_gguf
    from tinyq.export.tq import disk_size, save_quantized
    from tinyq.quantizer import QuantConfig, quantize_model

    cfg_all = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    models = cfg_all["models"]
    if args.only:
        models = [m for m in models if m["short"] == args.only]

    OUT.mkdir(exist_ok=True)
    resumen = []

    for m in models:
        short, model_id = m["short"], m["id"]
        dest = OUT / f"{short}-int{args.bits}-g{args.group}"
        gguf_path = OUT / f"{short}-int{args.bits}.gguf"
        if (dest / "tinyq.json").exists() and (gguf_path.exists() or args.no_gguf):
            print(f"[skip] {short}: ya existe")
            continue

        print(f"[art ] {short}: {model_id}")
        t0 = time.perf_counter()
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, low_cpu_mem_usage=True
            ).to(args.device).eval()

            cal = load_calibration(
                "wikitext2", tok, n_samples=args.samples, seq_len=args.seqlen
            )
            qcfg = QuantConfig(
                bits=args.bits, group_size=args.group, method="gptq",
                awq=True, search_scale=True,
            )
            report = quantize_model(model, cal, qcfg, device=args.device, progress=None)

            save_quantized(
                model, dest, cfg=qcfg,
                extra={"source_model": model_id, "calibration": cal.source},
            )
            tok.save_pretrained(dest)
            model.config.save_pretrained(dest)
            print(f"       .tq listo: {disk_size(dest) / 1e9:.2f} GB")

            if not args.no_gguf:
                model.to("cpu")
                export_gguf(model, gguf_path, dest, name=f"{short}-tinyq-int{args.bits}")
                print(f"       gguf listo: {gguf_path.stat().st_size / 1e9:.2f} GB")

            resumen.append({
                "short": short,
                "model": model_id,
                "tq_dir": str(dest),
                "gguf": str(gguf_path) if not args.no_gguf else None,
                "tq_bytes": disk_size(dest),
                "gguf_bytes": gguf_path.stat().st_size if gguf_path.exists() else 0,
                "quant": report.summary(),
                "seconds": time.perf_counter() - t0,
            })
            (OUT / "artifacts.json").write_text(
                json.dumps(resumen, indent=2), encoding="utf-8"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            (OUT / f"{short}.error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            print(f"       FALLO: {exc}")

    print("\nartefactos:")
    for r in resumen:
        print(
            f"  {r['short']:10} .tq {r['tq_bytes'] / 1e9:5.2f} GB · "
            f"gguf {r['gguf_bytes'] / 1e9:5.2f} GB · {r['seconds'] / 60:.1f} min"
        )


if __name__ == "__main__":
    main()
