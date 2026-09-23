"""Genera las graficas de la comparativa para el README.

Dos figuras, cada una en claro y oscuro (GitHub elige segun el tema del lector):

  docs/img/comparativa-herramientas.png   Octuma contra otros cuantizadores
  docs/img/degradacion-por-tamano.png     cuanto duele cuantizar segun el tamano

Los numeros salen de runs/*.json, no se escriben a mano: si el barrido cambia,
las graficas cambian con el.

    python scripts/grafica_comparativa.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path("runs")
DESTINO = Path("docs/img")

# Paleta validada para daltonismo: un solo azul para lo nuestro y grises para
# el contexto. El color nunca es el unico codigo: cada barra lleva su etiqueta.
TEMAS = {
    "claro": {
        "sufijo": "",
        "fondo": "#fcfcfb",
        "texto": "#0b0b0b",
        "texto2": "#52514e",
        "acento": "#2a78d6",
        "neutro": "#b8b7b2",
        "malo": "#e34948",
        "rejilla": "#e3e2dd",
    },
    "oscuro": {
        "sufijo": "-dark",
        "fondo": "#1a1a19",
        "texto": "#ffffff",
        "texto2": "#c3c2b7",
        "acento": "#3987e5",
        "neutro": "#6b6a65",
        "malo": "#e66767",
        "rejilla": "#333330",
    },
}


def cargar(nombre: str) -> dict | None:
    ruta = RUNS / f"{nombre}.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def perdida(ppl: float, base: float) -> float:
    return (ppl / base - 1) * 100


def datos_herramientas() -> list[tuple[str, float, bool]]:
    """Perdida de calidad de cada herramienta sobre el mismo Qwen 3B."""
    base = cargar("qwen3b__fp16__w20s2048_float16")
    if not base:
        raise SystemExit("falta la corrida FP16 del 3B en runs/")
    ref = base["perplexity"]

    # FP4 se queda fuera a proposito: con su +59.9% aplasta la escala y las
    # diferencias que importan (2.4 contra 6.7) se vuelven invisibles. Va en la
    # nota al pie, que es donde se lee sin deformar el resto.
    candidatos = [
        ("Octuma GPTQ+AWQ", "qwen3b__gptq-awq-int4__w20s2048_float16", True),
        ("Octuma GPTQ", "qwen3b__gptq-int4__w20s2048_float16", True),
        ("bitsandbytes NF4", "baseline__bnb-nf4__qwen3b", False),
    ]
    filas = []
    for etiqueta, archivo, es_nuestro in candidatos:
        d = cargar(archivo)
        if d:
            filas.append((etiqueta, perdida(d["perplexity"], ref), es_nuestro))
    return sorted(filas, key=lambda f: f[1])


def datos_por_tamano() -> list[tuple[str, float, str]]:
    """Cuanto se degrada el mejor metodo disponible en cada tamano.

    Devuelve tambien que metodo se uso: el 7B no tiene corrida con AWQ (se
    quedo sin memoria), asi que su punto no es del todo comparable y la
    grafica tiene que decirlo en vez de disimularlo.
    """
    tamanos = [("0.5B", "qwen0.5b"), ("1.5B", "qwen1.5b"), ("3B", "qwen3b"), ("7B", "qwen7b")]
    metodos = ["gptq-awq-int4", "gptq-int4"]  # el mejor primero
    filas = []
    for etiqueta, corto in tamanos:
        base = cargar(f"{corto}__fp16__w20s2048_float16")
        if not base:
            continue
        for metodo in metodos:
            d = cargar(f"{corto}__{metodo}__w20s2048_float16")
            if d:
                filas.append((etiqueta, perdida(d["perplexity"], base["perplexity"]), metodo))
                break
    return filas


def _base_figura(tema: dict, alto: float):
    fig, ax = plt.subplots(figsize=(8, alto), dpi=200)
    fig.patch.set_facecolor(tema["fondo"])
    ax.set_facecolor(tema["fondo"])
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(tema["rejilla"])
    ax.tick_params(colors=tema["texto2"], labelsize=10, length=0)
    return fig, ax


def grafica_herramientas(tema: dict) -> None:
    filas = datos_herramientas()
    fig, ax = _base_figura(tema, 3.6)

    etiquetas = [f[0] for f in filas]
    valores = [f[1] for f in filas]
    # el color distingue lo nuestro del contexto; la etiqueta lo dice igual
    colores = [
        tema["acento"] if es_nuestro else (tema["malo"] if v > 20 else tema["neutro"])
        for (_, v, es_nuestro) in filas
    ]

    y = range(len(filas))
    ax.barh(list(y), valores, color=colores, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(etiquetas, color=tema["texto"], fontsize=11)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=tema["rejilla"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Calidad perdida frente al modelo sin cuantizar (%)",
                  color=tema["texto2"], fontsize=10, labelpad=10)
    ax.set_xlim(0, max(valores) * 1.18)

    for i, v in enumerate(valores):
        ax.text(v + max(valores) * 0.015, i, f"{v:+.1f}%", va="center",
                color=tema["texto"], fontsize=10, fontweight="bold")

    ax.set_title("Qwen2.5-3B a 4 bits: menos es mejor",
                 color=tema["texto"], fontsize=13, fontweight="bold",
                 loc="left", pad=16)
    # la nota va debajo del eje ya dibujado, no flotando sobre la figura
    ax.text(0, -0.30,
            "Perplejidad en wikitext-2, 20 ventanas de 2048 tokens, mismo evaluador para todos.\n"
            "bitsandbytes FP4 queda fuera de la grafica: pierde +59.9%.",
            transform=ax.transAxes, color=tema["texto2"], fontsize=8.5,
            va="top", linespacing=1.5)

    DESTINO.mkdir(parents=True, exist_ok=True)
    salida = DESTINO / f"comparativa-herramientas{tema['sufijo']}.png"
    fig.savefig(salida, bbox_inches="tight", facecolor=tema["fondo"], pad_inches=0.3)
    plt.close(fig)
    print(f"  {salida}")


def grafica_por_tamano(tema: dict) -> None:
    filas = datos_por_tamano()
    if len(filas) < 2:
        print("  (faltan corridas para la grafica por tamano)")
        return
    fig, ax = _base_figura(tema, 3.4)

    etiquetas = [f[0] for f in filas]
    valores = [f[1] for f in filas]
    # un asterisco en el que no lleva AWQ, explicado en la nota
    etiquetas = [f"{e}*" if m != "gptq-awq-int4" else e for e, _, m in filas]
    hay_asterisco = any(m != "gptq-awq-int4" for *_, m in filas)
    x = range(len(filas))

    ax.plot(list(x), valores, color=tema["acento"], linewidth=2,
            marker="o", markersize=9, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(etiquetas, color=tema["texto"], fontsize=11)
    ax.yaxis.grid(True, color=tema["rejilla"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(valores) * 1.3)
    ax.set_ylabel("Calidad perdida (%)", color=tema["texto2"], fontsize=10)
    ax.set_xlabel("Tamaño del modelo", color=tema["texto2"], fontsize=10, labelpad=10)

    for xi, v in zip(x, valores, strict=True):
        ax.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", color=tema["texto"],
                    fontsize=10, fontweight="bold")

    ax.set_title("Entre más grande el modelo, menos duele cuantizarlo",
                 color=tema["texto"], fontsize=13, fontweight="bold",
                 loc="left", pad=16)
    nota = "Mejor metodo de Octuma frente a FP16, misma evaluacion en los cuatro modelos."
    if hay_asterisco:
        nota += "\n*GPTQ sin AWQ: esa corrida no cabia en memoria."
    ax.text(0, -0.24, nota, transform=ax.transAxes, color=tema["texto2"],
            fontsize=8.5, va="top", linespacing=1.5)

    DESTINO.mkdir(parents=True, exist_ok=True)
    salida = DESTINO / f"degradacion-por-tamano{tema['sufijo']}.png"
    fig.savefig(salida, bbox_inches="tight", facecolor=tema["fondo"], pad_inches=0.3)
    plt.close(fig)
    print(f"  {salida}")


def main() -> None:
    print("Generando graficas:")
    for tema in TEMAS.values():
        grafica_herramientas(tema)
        grafica_por_tamano(tema)


if __name__ == "__main__":
    main()
