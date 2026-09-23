"""Publica un modelo cuantizado en Hugging Face con su ficha tecnica.

    python scripts/upload_hf.py out/qwen05b-g32 --repo 1mano1/Qwen2.5-0.5B-Octuma-INT4 \
        --gguf out/qwen05b-int4.gguf --run runs/qwen0.5b__cal8k-gptq-awq-int4__w4s512.json

El token se lee de HF_TOKEN (entorno o .env) y nunca se imprime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("Falta HF_TOKEN (ponlo en .env o en el entorno)")


def model_card(meta: dict, run: dict | None, repo: str, gguf: str | None) -> str:
    cfg = meta.get("config", {})
    base = meta.get("source_model", "desconocido")
    bits = cfg.get("bits", 4)
    group = cfg.get("group_size", 64)
    method = "GPTQ + AWQ" if cfg.get("awq") else cfg.get("method", "gptq").upper()

    tabla = ""
    if run:
        ppl = run.get("perplexity")
        ev = run.get("eval", {})
        qs = run.get("quant_summary", {})
        tabla = f"""
## Resultados medidos

| Metrica | Valor |
|---|---|
| Perplejidad (wikitext-2) | {ppl:.3f} |
| Ventanas de evaluacion | {ev.get('windows')} x {ev.get('seq_len')} tokens |
| Compresion de los pesos | {qs.get('compression', 0):.2f}x |
| Error medio por capa | {qs.get('mean_rel_fro', 0):.4f} |
| Tiempo de cuantizacion | {qs.get('seconds', 0) / 60:.1f} min |
"""

    uso_gguf = ""
    if gguf:
        uso_gguf = f"""
### Con llama.cpp (tambien en Android)

```bash
llama-cli -m {Path(gguf).name} -p "Hola"
```
"""

    return f"""---
base_model: {base}
license: apache-2.0
tags:
  - octuma
  - quantized
  - int{bits}
  - gguf
---

# {repo.split('/')[-1]}

`{base}` cuantizado a **INT{bits}** con [Octuma](https://github.com/1mano1/octuma),
usando **{method}** con grupos de {group} pesos.

Calibrado con {meta.get('calibration', 'wikitext-2')}.
{tabla}{uso_gguf}
### Con PyTorch

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from octuma.export.tq import load_quantized
from huggingface_hub import snapshot_download

ruta = snapshot_download("{repo}")
cfg = AutoConfig.from_pretrained(ruta)
modelo = load_quantized(AutoModelForCausalLM.from_config(cfg), ruta)
tok = AutoTokenizer.from_pretrained(ruta)
```

## Como se hizo

```bash
octuma quantize {base} --out salida --bits {bits} --group {group}{' --awq' if cfg.get('awq') else ''}
octuma export salida --out modelo-int{bits}.gguf
```

Octuma cuantiza por grupos con punto cero, reparte el error de redondeo entre
las columnas pendientes (GPTQ) y escala los canales segun su importancia en las
activaciones (AWQ). El detalle completo esta en el repositorio.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--repo", required=True, help="usuario/nombre en Hugging Face")
    ap.add_argument("--gguf", help="archivo .gguf a incluir")
    ap.add_argument("--run", help="JSON de resultados para la ficha")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = load_token()
    d = Path(args.model_dir)
    meta = json.loads((d / "octuma.json").read_text(encoding="utf-8"))
    run = json.loads(Path(args.run).read_text(encoding="utf-8")) if args.run else None

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    card = model_card(meta, run, args.repo, args.gguf)
    (d / "README.md").write_text(card, encoding="utf-8")

    print(f"Subiendo {d} -> {args.repo}")
    api.upload_folder(
        folder_path=str(d),
        repo_id=args.repo,
        commit_message="Modelo cuantizado con Octuma",
    )
    if args.gguf:
        print(f"Subiendo {args.gguf}")
        api.upload_file(
            path_or_fileobj=args.gguf,
            path_in_repo=Path(args.gguf).name,
            repo_id=args.repo,
            commit_message="GGUF para llama.cpp",
        )
    print(f"Listo: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
