"""Mide un modelo dentro de llama.cpp contra sus propias alternativas.

Baja el modelo original, lo convierte a GGUF F16, lo cuantiza con los formatos
de llama.cpp y mide todo —incluido el .gguf de Octuma— con `llama-perplexity`.
Es la unica forma de comparar de tu a tu: mismo motor, mismo corpus, mismas
ventanas.

    python scripts/bench_gguf.py Qwen/Qwen2.5-0.5B-Instruct \\
        --octuma out/qwen05b-int4-fix.gguf --slug qwen0.5b

Escribe runs/gguf__<slug>.json. Con --keep-f16 conserva el F16 (6 GB en el 3B)
para no reconvertirlo en otra corrida.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

LLAMA = Path(os.environ.get("LLAMA_CPP_BIN", "C:/llamacpp/bin"))
CONVERT = Path(os.environ.get("LLAMA_CPP_SRC", "C:/llamacpp/src")) / "convert_hf_to_gguf.py"


def corre(cmd: list[str], desc: str) -> str:
    """Ejecuta y revienta si falla: un paso mudo que falla es lo que ya costo caro."""
    print(f"  -> {desc}")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"FALLO ({r.returncode}): {desc}")
    return r.stdout + r.stderr


def perplejidad(gguf: Path, texto: Path, ctx: int, chunks: int, ngl: int) -> float:
    salida = corre(
        [
            str(LLAMA / "llama-perplexity"), "-m", str(gguf), "-f", str(texto),
            "-c", str(ctx), "--chunks", str(chunks), "-ngl", str(ngl),
        ],
        f"perplejidad de {gguf.name}",
    )
    m = re.search(r"Final estimate: PPL = ([\d.]+)", salida)
    if not m:
        raise SystemExit(f"no se pudo leer la perplejidad de {gguf.name}")
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo", help="repo de Hugging Face del modelo original")
    ap.add_argument("--octuma", type=Path, required=True, help=".gguf de Octuma ya exportado")
    ap.add_argument("--slug", required=True, help="nombre corto para los archivos")
    ap.add_argument("--formatos", default="Q4_K_M,Q4_0", help="formatos de llama.cpp a comparar")
    ap.add_argument("--texto", type=Path, default=Path("out/wikitext2.txt"))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--keep-f16", action="store_true")
    ap.add_argument("--trabajo", type=Path, default=Path("out/bench"))
    args = ap.parse_args()

    if not args.texto.exists():
        raise SystemExit(f"falta {args.texto}: correr scripts/make_wikitext_txt.py")

    args.trabajo.mkdir(parents=True, exist_ok=True)
    f16 = args.trabajo / f"{args.slug}-f16.gguf"
    t0 = time.perf_counter()

    if not f16.exists():
        from huggingface_hub import snapshot_download

        print(f"[1/4] bajando {args.modelo}")
        original = snapshot_download(args.modelo, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        print("[2/4] convirtiendo a GGUF F16")
        corre(
            [sys.executable, str(CONVERT), original, "--outfile", str(f16), "--outtype", "f16"],
            "convert_hf_to_gguf",
        )
    else:
        print(f"[1-2/4] reusando {f16}")

    print("[3/4] cuantizando con llama.cpp")
    cuantizados: dict[str, Path] = {}
    for fmt in args.formatos.split(","):
        dst = args.trabajo / f"{args.slug}-{fmt.lower()}.gguf"
        if not dst.exists():
            corre([str(LLAMA / "llama-quantize"), str(f16), str(dst), fmt], f"cuantizar a {fmt}")
        cuantizados[fmt] = dst

    print("[4/4] midiendo")
    filas = [
        {"etiqueta": "F16 (original)", "gguf": f16, "familia": "original"},
        {"etiqueta": "Octuma INT4", "gguf": args.octuma, "familia": "octuma"},
    ]
    filas += [
        {"etiqueta": f"llama.cpp {f}", "gguf": p, "familia": "llama.cpp"}
        for f, p in cuantizados.items()
    ]

    for fila in filas:
        fila["ppl"] = perplejidad(fila["gguf"], args.texto, args.ctx, args.chunks, args.ngl)
        fila["bytes"] = fila["gguf"].stat().st_size
        print(f"     {fila['etiqueta']:22s} {fila['ppl']:8.4f}  {fila['bytes'] / 1e9:.2f} GB")

    base = filas[0]["ppl"]
    for fila in filas:
        fila["dano_pct"] = (fila["ppl"] / base - 1.0) * 100.0
        fila["compresion"] = filas[0]["bytes"] / fila["bytes"]
        fila["gguf"] = str(fila["gguf"])

    out = Path("runs") / f"gguf__{args.slug}.json"
    out.write_text(
        json.dumps(
            {
                "modelo": args.modelo,
                "slug": args.slug,
                "motor": "llama.cpp",
                "eval": {
                    "dataset": "wikitext2-test",
                    "windows": args.chunks,
                    "seq_len": args.ctx,
                    "corpus": str(args.texto),
                },
                "resultados": filas,
                "segundos": time.perf_counter() - t0,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {out}")

    if not args.keep_f16:
        f16.unlink(missing_ok=True)
        print(f"   ({f16.name} borrado; --keep-f16 para conservarlo)")
    shutil.rmtree(args.trabajo / ".cache", ignore_errors=True)


if __name__ == "__main__":
    main()
