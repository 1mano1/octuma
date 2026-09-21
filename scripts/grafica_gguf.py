"""Graficas de TinyQ contra los formatos de llama.cpp, dentro de llama.cpp.

Una figura por modelo (cuanto pierde cada formato frente al original) y una
figura de conjunto que cruza los tres tamanos. Cada una en claro y oscuro.

  docs/img/gguf-<slug>.png        una por modelo
  docs/img/gguf-por-tamano.png    la tendencia: que pasa al crecer el modelo

Los numeros salen de runs/gguf__*.json, que escribe scripts/bench_gguf.py.
No se escriben a mano.

    python scripts/grafica_gguf.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path("runs")
DESTINO = Path("docs/img")

# Misma paleta que grafica_comparativa.py: azul para lo nuestro, gris para el
# contexto, rojo solo para lo que se sale de rango. El color nunca es el unico
# codigo; cada barra lleva su etiqueta y su numero.
TEMAS = {
    "claro": {
        "sufijo": "", "fondo": "#fcfcfb", "texto": "#0b0b0b", "texto2": "#52514e",
        "acento": "#2a78d6", "neutro": "#b8b7b2", "malo": "#e34948", "rejilla": "#e3e2dd",
    },
    "oscuro": {
        "sufijo": "-dark", "fondo": "#1a1a19", "texto": "#ffffff", "texto2": "#c3c2b7",
        "acento": "#3987e5", "neutro": "#6b6a65", "malo": "#e66767", "rejilla": "#333330",
    },
}

MODELOS = [("qwen0.5b", "Qwen2.5-0.5B"), ("qwen1.5b", "Qwen2.5-1.5B"), ("qwen3b", "Qwen2.5-3B")]


def cargar(slug: str) -> dict | None:
    ruta = RUNS / f"gguf__{slug}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def _base_figura(tema: dict, alto: float):
    fig, ax = plt.subplots(figsize=(8, alto), dpi=200)
    fig.patch.set_facecolor(tema["fondo"])
    ax.set_facecolor(tema["fondo"])
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(tema["rejilla"])
    ax.tick_params(colors=tema["texto2"], labelsize=10, length=0)
    return fig, ax


def grafica_modelo(tema: dict, slug: str, titulo: str) -> None:
    d = cargar(slug)
    if not d:
        print(f"  (falta runs/gguf__{slug}.json)")
        return

    # el F16 es la referencia, no un competidor: se queda fuera de las barras
    filas = [r for r in d["resultados"] if r["familia"] != "original"]
    filas.sort(key=lambda r: r["dano_pct"])
    f16 = next(r for r in d["resultados"] if r["familia"] == "original")

    fig, ax = _base_figura(tema, 3.4)
    valores = [r["dano_pct"] for r in filas]
    etiquetas = [f"{r['etiqueta']}\n{r['bytes'] / 1e9:.2f} GB" for r in filas]
    colores = [tema["acento"] if r["familia"] == "tinyq" else tema["neutro"] for r in filas]

    y = range(len(filas))
    ax.barh(list(y), valores, color=colores, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(etiquetas, color=tema["texto"], fontsize=10)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=tema["rejilla"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Calidad perdida frente al original sin cuantizar (%)",
                  color=tema["texto2"], fontsize=10, labelpad=10)
    ax.set_xlim(0, max(valores) * 1.2)

    for i, v in enumerate(valores):
        ax.text(v + max(valores) * 0.015, i, f"{v:+.2f}%", va="center",
                color=tema["texto"], fontsize=10, fontweight="bold")

    ax.set_title(f"{titulo} a 4 bits: menos es mejor",
                 color=tema["texto"], fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.text(0, -0.42,
            f"Perplejidad en wikitext-2 dentro de llama.cpp, {d['eval']['windows']} ventanas "
            f"de {d['eval']['seq_len']} tokens.\n"
            f"El original en F16 da {f16['ppl']:.3f} y ocupa {f16['bytes'] / 1e9:.2f} GB.",
            transform=ax.transAxes, color=tema["texto2"], fontsize=8.5,
            va="top", linespacing=1.5)

    DESTINO.mkdir(parents=True, exist_ok=True)
    salida = DESTINO / f"gguf-{slug}{tema['sufijo']}.png"
    fig.savefig(salida, bbox_inches="tight", facecolor=tema["fondo"], pad_inches=0.3)
    plt.close(fig)
    print(f"  {salida}")


def grafica_por_tamano(tema: dict) -> None:
    datos = [(nombre, cargar(slug)) for slug, nombre in MODELOS]
    datos = [(n, d) for n, d in datos if d]
    if len(datos) < 2:
        print("  (faltan corridas para la grafica de conjunto)")
        return

    series: dict[str, list[float]] = {}
    for _, d in datos:
        for r in d["resultados"]:
            if r["familia"] != "original":
                series.setdefault(r["etiqueta"], []).append(r["dano_pct"])

    fig, ax = _base_figura(tema, 3.8)
    x = range(len(datos))
    estilos = {"TinyQ INT4": (tema["acento"], "o", 2.4, 3), }

    for etiqueta, valores in series.items():
        if len(valores) != len(datos):
            continue
        color, marca, grosor, z = estilos.get(etiqueta, (tema["neutro"], "s", 1.6, 2))
        ax.plot(list(x), valores, color=color, linewidth=grosor, marker=marca,
                markersize=8, zorder=z, label=etiqueta)
        # cada linea se nombra en su punta: se lee sin ir y venir a la leyenda
        ax.annotate(etiqueta, (len(datos) - 1, valores[-1]), textcoords="offset points",
                    xytext=(10, -3), color=color, fontsize=9.5, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels([n for n, _ in datos], color=tema["texto"], fontsize=11)
    ax.yaxis.grid(True, color=tema["rejilla"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(v) for v in series.values() if len(v) == len(datos)) * 1.25)
    ax.set_xlim(-0.25, len(datos) - 0.45)
    ax.set_ylabel("Calidad perdida (%)", color=tema["texto2"], fontsize=10)

    ax.set_title("Al crecer el modelo, TinyQ aguanta y los formatos estandar no",
                 color=tema["texto"], fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.text(0, -0.22,
            "Perplejidad en wikitext-2 dentro de llama.cpp, 20 ventanas de 2048 tokens.\n"
            "En el 0.5B, Q4_K_M pierde menos que TinyQ: el cruce esta entre 0.5B y 1.5B.",
            transform=ax.transAxes, color=tema["texto2"], fontsize=8.5,
            va="top", linespacing=1.5)

    DESTINO.mkdir(parents=True, exist_ok=True)
    salida = DESTINO / f"gguf-por-tamano{tema['sufijo']}.png"
    fig.savefig(salida, bbox_inches="tight", facecolor=tema["fondo"], pad_inches=0.3)
    plt.close(fig)
    print(f"  {salida}")


def main() -> None:
    for tema in TEMAS.values():
        for slug, nombre in MODELOS:
            grafica_modelo(tema, slug, nombre)
        grafica_por_tamano(tema)


if __name__ == "__main__":
    main()
