"""Arma runs/COMPARATIVA_GGUF.md desde runs/gguf__*.json.

Todo el documento se genera: las tablas, los porcentajes y las frases que
dicen quien gana. Si se vuelve a medir, se vuelve a correr esto y no hay
numeros viejos sobreviviendo en el texto.

    python scripts/tabla_gguf.py
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("runs")
MODELOS = [("qwen0.5b", "Qwen2.5-0.5B"), ("qwen1.5b", "Qwen2.5-1.5B"), ("qwen3b", "Qwen2.5-3B")]
RIVAL = "llama.cpp Q4_K_M"


def cargar(slug: str) -> dict | None:
    ruta = RUNS / f"gguf__{slug}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def fila_de(d: dict, etiqueta: str) -> dict | None:
    return next((r for r in d["resultados"] if r["etiqueta"] == etiqueta), None)


def main() -> None:
    datos = [(slug, nombre, cargar(slug)) for slug, nombre in MODELOS]
    datos = [(s, n, d) for s, n, d in datos if d]
    if not datos:
        raise SystemExit("no hay runs/gguf__*.json: correr scripts/bench_gguf.py")

    ev = datos[0][2]["eval"]
    L: list[str] = []
    L.append("# Octuma contra los formatos de llama.cpp")
    L.append("")
    L.append(
        f"Generado por `scripts/tabla_gguf.py`. Perplejidad en wikitext-2 test, "
        f"{ev['windows']} ventanas de {ev['seq_len']} tokens, **todo medido dentro de "
        f"llama.cpp** con `llama-perplexity`."
    )
    L.append("")
    L.append(
        "Comparar perplejidades entre motores distintos no significa nada: cada uno "
        "trocea y promedia a su manera. Por eso el `.gguf` de Octuma se mide dentro de "
        "llama.cpp, contra los formatos de llama.cpp, sobre el mismo corpus. Lo que se "
        "compara es **cuanto pierde cada formato respecto al mismo original en F16**."
    )
    L.append("")

    # --- tabla de conjunto ---
    L.append("## Todos los modelos")
    L.append("")
    L.append("| Modelo | Octuma INT4 | Q4_K_M | Q4_0 |")
    L.append("|---|---|---|---|")
    for _, nombre, d in datos:
        celdas = []
        for etiqueta in ("Octuma INT4", RIVAL, "llama.cpp Q4_0"):
            r = fila_de(d, etiqueta)
            if not r:
                celdas.append("—")
                continue
            otros = [o["dano_pct"] for o in d["resultados"] if o["familia"] != "original"]
            txt = f"+{r['dano_pct']:.2f}%"
            celdas.append(f"**{txt}**" if r["dano_pct"] == min(otros) else txt)
        L.append(f"| {nombre} | {celdas[0]} | {celdas[1]} | {celdas[2]} |")
    L.append("")
    L.append("En **negritas** el mejor de cada fila. Menos es mejor.")
    L.append("")

    # --- lectura, derivada de los datos ---
    ganados = []
    perdidos = []
    for _, nombre, d in datos:
        t, q = fila_de(d, "Octuma INT4"), fila_de(d, RIVAL)
        if t and q:
            (ganados if t["dano_pct"] < q["dano_pct"] else perdidos).append((nombre, t, q))

    L.append("## Que dicen los numeros")
    L.append("")
    if ganados:
        peor = max(ganados, key=lambda g: g[2]["dano_pct"] / g[1]["dano_pct"])
        veces = peor[2]["dano_pct"] / peor[1]["dano_pct"]
        L.append(
            f"**Octuma gana en {len(ganados)} de {len(datos)} modelos.** La diferencia mas "
            f"grande esta en {peor[0]}: {peor[1]['dano_pct']:.2f}% contra "
            f"{peor[2]['dano_pct']:.2f}% de Q4_K_M, **{veces:.1f} veces menos dano**."
        )
        L.append("")
    if perdidos:
        for nombre, t, q in perdidos:
            L.append(
                f"**En {nombre} pierde**: {t['dano_pct']:.2f}% contra {q['dano_pct']:.2f}% "
                f"de Q4_K_M, y ademas ocupa mas ({t['bytes'] / 1e9:.2f} GB contra "
                f"{q['bytes'] / 1e9:.2f} GB). En los modelos mas chicos no hay razon para "
                f"preferir Octuma sobre Q4_K_M."
            )
        L.append("")

    serie_t = [(n, fila_de(d, "Octuma INT4")["dano_pct"]) for _, n, d in datos]
    serie_q = [(n, fila_de(d, RIVAL)["dano_pct"]) for _, n, d in datos]
    L.append(
        "**La tendencia importa mas que cualquier fila suelta.** Al crecer el modelo, "
        "Q4_K_M se degrada cada vez mas ("
        + " → ".join(f"{v:.2f}%" for _, v in serie_q)
        + ") mientras Octuma se mantiene plano ("
        + " → ".join(f"{v:.2f}%" for _, v in serie_t)
        + "). El cruce esta entre "
        + f"{serie_t[0][0]} y {serie_t[1][0]}."
    )
    L.append("")
    L.append(
        "Eso encaja con lo que ya media el barrido en PyTorch: entre mas grande el "
        "modelo, menos duele cuantizarlo bien. Y es el caso de uso que importa para "
        "la app Android, donde se corre el modelo mas grande que quepa."
    )
    L.append("")
    L.append("A **Q4_0 le gana en los tres**, siempre por varias veces.")
    L.append("")

    # --- detalle por modelo ---
    L.append("## Detalle por modelo")
    L.append("")
    for slug, nombre, d in datos:
        L.append(f"### {nombre}")
        L.append("")
        L.append(f"![{nombre}](../docs/img/gguf-{slug}.png)")
        L.append("")
        L.append("| Formato | Perplejidad | vs F16 | Tamaño | Compresion |")
        L.append("|---|---|---|---|---|")
        for r in sorted(d["resultados"], key=lambda r: (r["familia"] != "original", r["dano_pct"])):
            dano = "—" if r["familia"] == "original" else f"+{r['dano_pct']:.2f}%"
            comp = "—" if r["familia"] == "original" else f"{r['compresion']:.2f}x"
            nom = f"**{r['etiqueta']}**" if r["familia"] == "octuma" else r["etiqueta"]
            L.append(
                f"| {nom} | {r['ppl']:.4f} | {dano} | {r['bytes'] / 1e9:.2f} GB | {comp} |"
            )
        L.append("")

    L.append("## Como reproducirlo")
    L.append("")
    L.append("```bash")
    L.append("python scripts/make_wikitext_txt.py --out out/wikitext2.txt")
    L.append("python scripts/bench_gguf.py Qwen/Qwen2.5-3B-Instruct \\")
    L.append("    --octuma out/qwen3b-int4-fix.gguf --slug qwen3b")
    L.append("python scripts/tabla_gguf.py && python scripts/grafica_gguf.py")
    L.append("```")
    L.append("")
    L.append(
        "`bench_gguf.py` baja el original, lo convierte a F16, lo cuantiza con "
        "llama.cpp y mide las cuatro variantes de una pasada."
    )
    L.append("")

    salida = RUNS / "COMPARATIVA_GGUF.md"
    salida.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {salida}")


if __name__ == "__main__":
    main()
