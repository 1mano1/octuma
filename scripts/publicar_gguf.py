"""Publica el .gguf arreglado y su ficha en Hugging Face.

La ficha se genera desde runs/gguf__<slug>.json y tinyq.json: ningun numero se
escribe a mano. Reemplaza el .gguf con el mismo nombre que ya tenia el repo,
para no romper los enlaces ni los ejemplos.

    python scripts/publicar_gguf.py qwen3b            # sube modelo y ficha
    python scripts/publicar_gguf.py qwen3b --dry-run  # solo escribe la ficha

El token se lee de HF_TOKEN (entorno o .env) y nunca se imprime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from upload_hf import load_token  # mismo manejo de token que el resto

RUNS = Path("runs")

# slug -> (repo, .gguf arreglado en local, nombre dentro del repo, carpeta .tq)
MODELOS = {
    "qwen0.5b": (
        "Imanol11/qwen0.5b-int4-TinyQ",
        Path("out/qwen05b-int4-fix.gguf"),
        "qwen0.5b-int4.gguf",
        Path("out/qwen05b-tq"),
    ),
    "qwen1.5b": (
        "Imanol11/qwen1.5b-int4-TinyQ",
        Path("out/qwen15b-int4-fix.gguf"),
        "qwen1.5b-int4.gguf",
        Path("out/qwen15b-tq"),
    ),
    "qwen3b": (
        "Imanol11/qwen3b-int4-TinyQ",
        Path("out/qwen3b-int4-fix.gguf"),
        "qwen3b-int4.gguf",
        Path("out/qwen3b-tq"),
    ),
}

GITHUB = "https://github.com/1mano1/TinyQ"


def ficha(slug: str, repo: str, gguf_nombre: str, meta: dict, d: dict) -> str:
    base = meta.get("source_model", "desconocido")
    cfg = meta.get("config", {})
    grupo = cfg.get("group_size", 32)
    corto = repo.split("/")[-1]

    filas = sorted(d["resultados"], key=lambda r: (r["familia"] != "original", r["dano_pct"]))
    nuestro = next(r for r in d["resultados"] if r["familia"] == "tinyq")
    f16 = next(r for r in d["resultados"] if r["familia"] == "original")
    rival = min(
        (r for r in d["resultados"] if r["familia"] == "llama.cpp"),
        key=lambda r: r["dano_pct"],
    )

    tabla = ["| Formato | Perplejidad | Calidad perdida | Tamaño |", "|---|---|---|---|"]
    for r in filas:
        dano = "—" if r["familia"] == "original" else f"+{r['dano_pct']:.2f}%"
        nombre = f"**{r['etiqueta']}** (este modelo)" if r["familia"] == "tinyq" else r["etiqueta"]
        tabla.append(f"| {nombre} | {r['ppl']:.4f} | {dano} | {r['bytes'] / 1e9:.2f} GB |")

    if nuestro["dano_pct"] < rival["dano_pct"]:
        veredicto = (
            f"**Este modelo pierde {rival['dano_pct'] / nuestro['dano_pct']:.1f} veces menos "
            f"calidad que {rival['etiqueta']}**, el formato mas usado para correr modelos "
            f"en local, a cambio de "
            f"{(nuestro['bytes'] - rival['bytes']) / 1e9:.2f} GB mas de archivo."
        )
    else:
        veredicto = (
            f"**En este tamaño, {rival['etiqueta']} rinde mejor** "
            f"(+{rival['dano_pct']:.2f}% contra +{nuestro['dano_pct']:.2f}%) y ocupa menos. "
            f"La ventaja de TinyQ aparece en modelos mas grandes: en Qwen2.5-3B la relacion "
            f"se invierte. La tabla esta aqui para que se vea, no para esconderla."
        )

    ev = d["eval"]
    return f"""---
base_model: {base}
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
language:
  - en
  - es
tags:
  - tinyq
  - quantized
  - int4
  - gguf
  - llama.cpp
---

# {corto}

[`{base}`]({f"https://huggingface.co/{base}"}) cuantizado a **INT4** con
[TinyQ]({GITHUB}): **{f16["bytes"] / 1e9:.2f} GB → {nuestro["bytes"] / 1e9:.2f} GB**
({nuestro["compresion"]:.2f}x mas chico) perdiendo **{nuestro["dano_pct"]:.2f}%** de calidad.

Incluye las dos formas de usarlo: `{gguf_nombre}` para llama.cpp y Android, y
la carpeta `.tq` para PyTorch.

## Calidad medida

Perplejidad en wikitext-2 test, {ev["windows"]} ventanas de {ev["seq_len"]} tokens,
**todo medido dentro de llama.cpp** con `llama-perplexity` para que la
comparacion sea de tu a tu.

{chr(10).join(tabla)}

{veredicto}

> Las perplejidades solo se comparan entre si cuando salen del **mismo motor y
> el mismo corpus**. Cada motor trocea y promedia distinto, asi que no
> contrastes estos numeros con los de otra ficha sin revisar como se midieron.

La comparativa de los tres tamaños esta en
[`runs/COMPARATIVA_GGUF.md`]({GITHUB}/blob/main/runs/COMPARATIVA_GGUF.md).

## Como usarlo

### llama.cpp (tambien en Android)

```bash
hf download {repo} --local-dir {slug}
llama-cli -m {slug}/{gguf_nombre} -p "Hola"
```

### PyTorch

```bash
pip install "tinyq[hf] @ git+{GITHUB}.git"
```

```python
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tinyq.export.tq import load_quantized

ruta = snapshot_download("{repo}")
cfg = AutoConfig.from_pretrained(ruta)
modelo = load_quantized(AutoModelForCausalLM.from_config(cfg), ruta)
tok = AutoTokenizer.from_pretrained(ruta)
```

En PyTorch el modelo cuantizado **genera mas lento** que el original: TinyQ
desempaqueta los 4 bits en cada multiplicacion sin un kernel dedicado. Para
velocidad, usa el `.gguf` con llama.cpp.

## Como se hizo

```bash
tinyq quantize {base}
tinyq export <carpeta> --out {gguf_nombre}
python scripts/verify_gguf.py <carpeta> {gguf_nombre}
```

**GPTQ + AWQ** con grupos de {grupo} pesos, calibrado con
{meta.get("calibration", "wikitext-2")}. TinyQ cuantiza por grupos con punto
cero, reparte el error de redondeo entre las columnas pendientes (GPTQ) y
escala los canales segun su importancia en las activaciones (AWQ).

## Historial

El `.gguf` publicado aqui **antes del 2026-09-20** estaba degradado: tenia tres
metadatos mal escritos (`rope.freq_base`, el pre-tokenizador y el
`eos_token_id`). Los pesos siempre estuvieron bien; el archivo actual los
reescribe correctamente. Si lo descargaste antes de esa fecha, vuelve a
bajarlo.

## Licencia

Los pesos heredan la licencia de [`{base}`](https://huggingface.co/{base}).
TinyQ es MIT.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", choices=sorted(MODELOS))
    ap.add_argument("--dry-run", action="store_true", help="escribe la ficha sin subir nada")
    args = ap.parse_args()

    repo, gguf_local, gguf_nombre, tq = MODELOS[args.slug]
    d = json.loads((RUNS / f"gguf__{args.slug}.json").read_text(encoding="utf-8"))
    meta = json.loads((tq / "tinyq.json").read_text(encoding="utf-8"))

    if not gguf_local.exists():
        raise SystemExit(f"falta {gguf_local}")

    texto = ficha(args.slug, repo, gguf_nombre, meta, d)
    destino = Path(f"out/CARD_{args.slug}.md")
    destino.write_text(texto, encoding="utf-8")
    print(f"ficha -> {destino}")

    if args.dry_run:
        print("(--dry-run: no se subio nada)")
        return

    import time

    from huggingface_hub import HfApi

    api = HfApi(token=load_token())

    def subir(local: str, remoto: str, mensaje: str) -> None:
        """Con reintentos: el Hub corta la conexion de vez en cuando y un fallo
        a medias deja el repo con el .gguf nuevo y la ficha vieja."""
        for intento in range(5):
            try:
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=remoto,
                    repo_id=repo,
                    commit_message=mensaje,
                )
                return
            except Exception as e:  # noqa: BLE001 - el Hub falla de muchas formas
                espera = 5 * (intento + 1)
                print(f"  intento {intento + 1} fallo ({type(e).__name__}), reintento en {espera}s")
                time.sleep(espera)
        raise SystemExit(f"no se pudo subir {remoto} a {repo}")

    print(f"subiendo {gguf_local.name} ({gguf_local.stat().st_size / 1e9:.2f} GB) -> {repo}")
    subir(
        str(gguf_local),
        gguf_nombre,
        "GGUF re-exportado: arregla rope_theta, pre-tokenizador y eos_token_id",
    )
    subir(str(destino), "README.md", "Ficha con la calidad medida dentro de llama.cpp")
    print(f"listo: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
