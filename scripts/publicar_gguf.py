"""Publica el .gguf arreglado y su ficha en Hugging Face.

La ficha se genera desde runs/gguf__<slug>.json y octuma.json: ningun numero se
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
        "Imanol11/qwen0.5b-int4-Octuma",
        Path("out/qwen05b-int4-fix.gguf"),
        "qwen0.5b-int4.gguf",
        Path("out/qwen05b-tq"),
    ),
    "qwen1.5b": (
        "Imanol11/qwen1.5b-int4-Octuma",
        Path("out/qwen15b-int4-fix.gguf"),
        "qwen1.5b-int4.gguf",
        Path("out/qwen15b-tq"),
    ),
    "qwen3b": (
        "Imanol11/qwen3b-int4-Octuma",
        Path("out/qwen3b-int4-fix.gguf"),
        "qwen3b-int4.gguf",
        Path("out/qwen3b-tq"),
    ),
}

GITHUB = "https://github.com/1mano1/octuma"

# Exigido por la Qwen RESEARCH LICENSE §3c. Literal, no parafrasear.
AVISO_QWEN = (
    "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, "
    "Copyright (c) Alibaba Cloud. All Rights Reserved."
)


def licencia_base(base: str, token: str) -> dict:
    """La licencia del modelo original, leida del Hub en el momento de publicar.

    No se cablea: **Qwen2.5-3B-Instruct no es Apache 2.0** como los demas, sino
    la Qwen Research License, que es solo para uso no comercial. Copiarla mal en
    la ficha es tergiversar la licencia de Alibaba, asi que sale del origen.
    """
    from huggingface_hub import HfApi

    card = HfApi(token=token).model_info(base).cardData or {}
    lic = card.get("license", "")
    nombre = card.get("license_name")
    enlace = card.get("license_link") or f"https://huggingface.co/{base}/blob/main/LICENSE"
    if lic == "apache-2.0":
        return {"spdx": lic, "nombre": None, "enlace": enlace, "comercial": True}
    if nombre == "qwen-research":
        return {"spdx": "other", "nombre": nombre, "enlace": enlace, "comercial": False}
    # Una licencia que no conocemos: parar antes que adivinar.
    raise SystemExit(
        f"{base} declara license={lic!r} license_name={nombre!r}, que este script no "
        "sabe describir. Revisa las condiciones a mano antes de publicar."
    )


def ficha(slug: str, repo: str, gguf_nombre: str, meta: dict, d: dict, lic: dict) -> str:
    base = meta.get("source_model", "desconocido")
    cfg = meta.get("config", {})
    grupo = cfg.get("group_size", 32)
    corto = repo.split("/")[-1]

    filas = sorted(d["resultados"], key=lambda r: (r["familia"] != "original", r["dano_pct"]))
    nuestro = next(r for r in d["resultados"] if r["familia"] == "octuma")
    f16 = next(r for r in d["resultados"] if r["familia"] == "original")
    rival = min(
        (r for r in d["resultados"] if r["familia"] == "llama.cpp"),
        key=lambda r: r["dano_pct"],
    )

    tabla = ["| Formato | Perplejidad | Calidad perdida | Tamaño |", "|---|---|---|---|"]
    for r in filas:
        dano = "—" if r["familia"] == "original" else f"+{r['dano_pct']:.2f}%"
        nombre = f"**{r['etiqueta']}** (este modelo)" if r["familia"] == "octuma" else r["etiqueta"]
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
            f"La ventaja de Octuma aparece en modelos mas grandes: en Qwen2.5-3B la relacion "
            f"se invierte. La tabla esta aqui para que se vea, no para esconderla."
        )

    ev = d["eval"]

    campos_lic = [f"license: {lic['spdx']}"]
    if lic["nombre"]:
        campos_lic.append(f"license_name: {lic['nombre']}")
        campos_lic.append(f"license_link: {lic['enlace']}")

    if lic["comercial"]:
        aviso = ""
        bloque_lic = (
            f"Los pesos derivan de [`{base}`](https://huggingface.co/{base}), bajo "
            f"**Apache 2.0**; la copia del original va en `LICENSE`. El codigo de "
            f"Octuma es MIT.\n\nLos archivos de pesos estan **modificados** respecto "
            f"al original: cuantizados a INT4 con Octuma."
        )
    else:
        aviso = (
            f"> ### ⚠️ Solo uso no comercial\n>\n"
            f"> `{base}` no es Apache 2.0 como otros modelos de la familia: esta bajo "
            f"la [Qwen Research License]({lic['enlace']}), que permite **unicamente "
            f"investigacion y evaluacion**. Esa condicion la hereda este modelo "
            f"cuantizado. Para uso comercial hay que pedirle licencia a Alibaba Cloud.\n\n"
        )
        bloque_lic = (
            f"Los pesos derivan de [`{base}`](https://huggingface.co/{base}), bajo la "
            f"[Qwen RESEARCH LICENSE AGREEMENT]({lic['enlace']}), **no comercial**. "
            f"La copia integra del acuerdo va en `LICENSE` y el aviso de atribucion "
            f"en `NOTICE`, como pide su §3.\n\nLos archivos de pesos estan "
            f"**modificados** respecto al original: cuantizados a INT4 con Octuma (§3b).\n\n"
            f"> {AVISO_QWEN}\n\n"
            f"El codigo de Octuma es MIT, pero **eso no afloja las condiciones de los "
            f"pesos**: son dos licencias distintas sobre dos cosas distintas."
        )

    return f"""---
base_model: {base}
{chr(10).join(campos_lic)}
library_name: gguf
pipeline_tag: text-generation
language:
  - en
  - es
tags:
  - octuma
  - quantized
  - int4
  - gguf
  - llama.cpp
---

# {corto}

Built with Qwen.

[`{base}`]({f"https://huggingface.co/{base}"}) cuantizado a **INT4** con
[Octuma]({GITHUB}): **{f16["bytes"] / 1e9:.2f} GB → {nuestro["bytes"] / 1e9:.2f} GB**
({nuestro["compresion"]:.2f}x mas chico) perdiendo **{nuestro["dano_pct"]:.2f}%** de calidad.

{aviso}

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
pip install "octuma[hf] @ git+{GITHUB}.git"
```

```python
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from octuma.export.tq import load_quantized

ruta = snapshot_download("{repo}")
cfg = AutoConfig.from_pretrained(ruta)
modelo = load_quantized(AutoModelForCausalLM.from_config(cfg), ruta)
tok = AutoTokenizer.from_pretrained(ruta)
```

En PyTorch el modelo cuantizado **genera mas lento** que el original: Octuma
desempaqueta los 4 bits en cada multiplicacion sin un kernel dedicado. Para
velocidad, usa el `.gguf` con llama.cpp.

## Como se hizo

```bash
octuma quantize {base}
octuma export <carpeta> --out {gguf_nombre}
python scripts/verify_gguf.py <carpeta> {gguf_nombre}
```

**GPTQ + AWQ** con grupos de {grupo} pesos, calibrado con
{meta.get("calibration", "wikitext-2")}. Octuma cuantiza por grupos con punto
cero, reparte el error de redondeo entre las columnas pendientes (GPTQ) y
escala los canales segun su importancia en las activaciones (AWQ).

## Historial

El `.gguf` publicado aqui **antes del 2026-09-20** estaba degradado: tenia tres
metadatos mal escritos (`rope.freq_base`, el pre-tokenizador y el
`eos_token_id`). Los pesos siempre estuvieron bien; el archivo actual los
reescribe correctamente. Si lo descargaste antes de esa fecha, vuelve a
bajarlo.

## Licencia

{bloque_lic}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", choices=sorted(MODELOS))
    ap.add_argument("--dry-run", action="store_true", help="escribe la ficha sin subir nada")
    args = ap.parse_args()

    repo, gguf_local, gguf_nombre, tq = MODELOS[args.slug]
    d = json.loads((RUNS / f"gguf__{args.slug}.json").read_text(encoding="utf-8"))
    meta = json.loads((tq / "octuma.json").read_text(encoding="utf-8"))

    if not gguf_local.exists():
        raise SystemExit(f"falta {gguf_local}")

    token = load_token()
    base = meta.get("source_model", "")
    lic = licencia_base(base, token)
    print(f"licencia de {base}: {lic['nombre'] or lic['spdx']}"
          f"{'' if lic['comercial'] else '  (NO COMERCIAL)'}")

    texto = ficha(args.slug, repo, gguf_nombre, meta, d, lic)
    destino = Path(f"out/CARD_{args.slug}.md")
    destino.write_text(texto, encoding="utf-8")
    print(f"ficha -> {destino}")

    if args.dry_run:
        print("(--dry-run: no se subio nada)")
        return

    import time

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)

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

    # Idempotente: si el Hub ya tiene este mismo archivo, no repetir gigabytes.
    en_hub = {
        s.rfilename: s.size
        for s in api.model_info(repo, files_metadata=True).siblings
    }
    if en_hub.get(gguf_nombre) == gguf_local.stat().st_size:
        print(f"{gguf_nombre} ya esta arriba con el mismo tamaño, no lo resubo")
    else:
        print(f"subiendo {gguf_local.name} ({gguf_local.stat().st_size / 1e9:.2f} GB) -> {repo}")
        subir(
            str(gguf_local),
            gguf_nombre,
            "GGUF re-exportado: arregla rope_theta, pre-tokenizador y eos_token_id",
        )

    # La licencia del original viaja con los pesos derivados: Apache 2.0 §4(a) y
    # Qwen Research §3a piden ambas entregar copia a quien reciba el modelo.
    origen = hf_hub_download(base, "LICENSE", token=token)
    subir(origen, "LICENSE", f"Licencia del modelo original ({lic['nombre'] or lic['spdx']})")

    if not lic["comercial"]:
        aviso = Path(f"out/NOTICE_{args.slug}.txt")
        aviso.write_text(
            f"{AVISO_QWEN}\n\n"
            f"Este repositorio contiene una obra derivada de {base}:\n"
            f"los pesos fueron modificados (cuantizados a INT4) con Octuma,\n"
            f"{GITHUB}\n",
            encoding="utf-8",
        )
        subir(str(aviso), "NOTICE", "Aviso de atribucion que pide la Qwen Research License")

    subir(str(destino), "README.md", "Ficha con la calidad medida dentro de llama.cpp")
    print(f"listo: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
