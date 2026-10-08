"""Genera la animacion de cabecera del README (docs/img/hero*.svg).

Es un SVG animado con CSS: GitHub lo muestra como imagen y la animacion corre
sola, sin JavaScript. Enseña los cuatro pasos de Octuma y como un modelo pasa
de FP16 a INT4.

**Los numeros no se escriben a mano.** Salen de `runs/gguf__qwen3b.json`, que
es la medicion del Qwen2.5-3B dentro de llama.cpp: si se vuelve a medir, se
vuelve a correr este script y la cabecera cambia sola.

    python scripts/hero_svg.py

El logo se incrusta en base64 porque un SVG mostrado como imagen no puede
cargar archivos externos.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
IMG = RAIZ / "docs" / "img"

# Los colores de la app (lib/core/design/tokens.dart en el repo de Octuma App)
FONDO = "#080A14"
PANEL = "#11152A"
BORDE = "#232842"
ACENTO = "#5B7CFF"
ACENTO_SUAVE = "#8FA6FF"
TEXTO = "#F2F4FF"
TENUE = "#9AA3C7"
VERDE = "#4ADE80"

TEXTOS = {
    "en": {
        "lema": "Quantize language models to 4 bits.",
        "lema2": "Run them on modest laptops and Android phones.",
        "pasos": ["Calibrate", "Quantize", "Evaluate", "Export"],
        "detalles": [
            "real text finds what matters",
            "GPTQ + AWQ, groups of 32",
            "perplexity, memory, speed",
            ".gguf · llama.cpp · Android",
        ],
        "antes": "original · FP16",
        "despues": "Octuma · INT4",
        "perdida": "quality lost",
        "menor": "smaller",
        "nota": "{modelo}, measured inside llama.cpp · wikitext-2, 20 windows of 2048 tokens",
        "aria": (
            "Octuma quantizes language models to 4 bits in four steps: calibrate, "
            "quantize, evaluate and export. {modelo} goes from {antes} GB in FP16 to "
            "{despues} GB in INT4, {veces}x smaller, losing {perdida}% of quality."
        ),
    },
    "es": {
        "lema": "Cuantiza modelos de lenguaje a 4 bits.",
        "lema2": "Córrelos en laptops modestas y teléfonos Android.",
        "pasos": ["Calibrar", "Cuantizar", "Evaluar", "Exportar"],
        "detalles": [
            "texto real dice qué importa",
            "GPTQ + AWQ, grupos de 32",
            "perplejidad, memoria, tok/s",
            ".gguf · llama.cpp · Android",
        ],
        "antes": "original · FP16",
        "despues": "Octuma · INT4",
        "perdida": "de calidad perdida",
        "menor": "más chico",
        "nota": "{modelo}, medido dentro de llama.cpp · wikitext-2, 20 ventanas de 2048 tokens",
        "aria": (
            "Octuma cuantiza modelos de lenguaje a 4 bits en cuatro pasos: calibrar, "
            "cuantizar, evaluar y exportar. {modelo} pasa de {antes} GB en FP16 a "
            "{despues} GB en INT4, {veces} veces más chico, perdiendo {perdida}% de calidad."
        ),
    },
}


def medicion() -> dict:
    datos = json.loads((RAIZ / "runs" / "gguf__qwen3b.json").read_text(encoding="utf-8"))
    por_familia = {r["familia"]: r for r in datos["resultados"] if r["familia"] != "llama.cpp"}
    original, octuma = por_familia["original"], por_familia["octuma"]
    return {
        "modelo": datos["modelo"].split("/")[-1].replace("-Instruct", ""),
        "antes": original["bytes"] / 1e9,
        "despues": octuma["bytes"] / 1e9,
        "perdida": octuma["dano_pct"],
        "veces": octuma["compresion"],
    }


def svg(idioma: str, m: dict, logo_b64: str) -> str:
    t = TEXTOS[idioma]
    ancho, alto = 880, 400
    x0, ancho_barra = 60, 760
    fraccion = m["despues"] / m["antes"]
    fmt = {
        "modelo": m["modelo"],
        "antes": f"{m['antes']:.2f}",
        "despues": f"{m['despues']:.2f}",
        "perdida": f"{m['perdida']:.2f}",
        "veces": f"{m['veces']:.2f}",
    }

    # Cuatro pasos, 3 s cada uno, en un ciclo de 12 s.
    ciclo = 12
    paso_ancho, hueco = 175, 20
    pasos = []
    for i, (nombre, detalle) in enumerate(zip(t["pasos"], t["detalles"], strict=True)):
        x = x0 + i * (paso_ancho + hueco)
        pasos.append(
            f'<g class="paso p{i}">'
            f'<rect class="caja" x="{x}" y="150" width="{paso_ancho}" height="62" rx="12"/>'
            f'<text class="num" x="{x + 16}" y="176">{i + 1}</text>'
            f'<text class="nombre" x="{x + 36}" y="176">{nombre}</text>'
            f'<text class="chico" x="{x + 16}" y="198">{_corta(detalle)}</text>'
            f"</g>"
        )
        if i < 3:
            fx = x + paso_ancho + 4
            pasos.append(
                f'<path class="flecha" d="M{fx} 181h{hueco - 9}m-4 -4l4 4l-4 4"/>'
            )

    estilo_pasos = "\n".join(
        f".p{i} .caja{{animation:paso{i} {ciclo}s infinite}}"
        f".p{i} .nombre,.p{i} .num{{animation:letra{i} {ciclo}s infinite}}"
        for i in range(4)
    )
    claves = []
    for i in range(4):
        ini, fin = i * 25, (i + 1) * 25
        antes = f"{max(ini - 2, 0)}%{{stroke:{BORDE};fill:{PANEL}}}" if i else ""
        despues = f"{min(fin + 2, 100)}%{{stroke:{BORDE};fill:{PANEL}}}" if i < 3 else ""
        claves.append(
            f"@keyframes paso{i}{{0%,100%{{stroke:{BORDE};fill:{PANEL}}}{antes}"
            f"{ini + 1}%,{fin - 1}%{{stroke:{ACENTO};fill:#161C3D}}{despues}}}"
        )
        antes_l = f"{max(ini - 2, 0)}%{{fill:{TENUE}}}" if i else ""
        despues_l = f"{min(fin + 2, 100)}%{{fill:{TENUE}}}" if i < 3 else ""
        claves.append(
            f"@keyframes letra{i}{{0%,100%{{fill:{TENUE}}}{antes_l}"
            f"{ini + 1}%,{fin - 1}%{{fill:{TEXTO}}}{despues_l}}}"
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ancho} {alto}" width="{ancho}" height="{alto}" role="img" aria-label="{t['aria'].format(**fmt)}">
  <title>Octuma — {fmt['antes']} GB → {fmt['despues']} GB, +{fmt['perdida']}%</title>
  <!-- Generado por scripts/hero_svg.py desde runs/gguf__qwen3b.json. No editar a mano. -->
  <style>
    text{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
    .marca{{font-size:34px;font-weight:700;fill:{TEXTO}}}
    .lema{{font-size:16px;fill:{TENUE}}}
    .nombre{{font-size:15px;font-weight:600;fill:{TENUE}}}
    .num{{font-size:13px;font-weight:700;fill:{TENUE}}}
    .chico{{font-size:9.5px;fill:{TENUE}}}
    .caja{{fill:{PANEL};stroke:{BORDE};stroke-width:1.5}}
    .flecha{{fill:none;stroke:{BORDE};stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}}
    .etq{{font-size:12px;fill:{TENUE}}}
    .gb{{font-size:22px;font-weight:700;fill:{TEXTO}}}
    .dato{{font-size:15px;font-weight:600;fill:{VERDE}}}
    .nota{{font-size:11px;fill:{TENUE}}}
    /* Sin animacion (un visor que no la soporte) se ve el estado final. */
    .barra{{transform-box:fill-box;transform-origin:left center;transform:scaleX({fraccion:.4f});animation:encoge {ciclo}s infinite}}
    .antes{{opacity:0;animation:sale {ciclo}s infinite}}
    .despues{{animation:entra {ciclo}s infinite}}
    .resultado{{animation:resultado {ciclo}s infinite}}
    .archivo{{animation:archivo {ciclo}s infinite}}
    {estilo_pasos}
    {''.join(claves)}
    @keyframes encoge{{0%,27%{{transform:scaleX(1)}}47%,96%{{transform:scaleX({fraccion:.4f})}}100%{{transform:scaleX(1)}}}}
    @keyframes sale{{0%,30%{{opacity:1}}36%,96%{{opacity:0}}100%{{opacity:1}}}}
    @keyframes entra{{0%,38%{{opacity:0}}45%,96%{{opacity:1}}100%{{opacity:0}}}}
    @keyframes resultado{{0%,52%{{opacity:0}}58%,96%{{opacity:1}}100%{{opacity:0}}}}
    @keyframes archivo{{0%,77%{{opacity:0}}83%,96%{{opacity:1}}100%{{opacity:0}}}}
    @media (prefers-reduced-motion:reduce){{*{{animation:none!important}}}}
  </style>
  <rect width="{ancho}" height="{alto}" rx="16" fill="{FONDO}"/>
  <rect x="0.5" y="0.5" width="{ancho - 1}" height="{alto - 1}" rx="16" fill="none" stroke="{BORDE}"/>

  <image x="{x0}" y="34" width="84" height="84" href="data:image/png;base64,{logo_b64}"/>
  <text class="marca" x="{x0 + 104}" y="68">Octuma</text>
  <text class="lema" x="{x0 + 104}" y="94">{t['lema']}</text>
  <text class="lema" x="{x0 + 104}" y="114">{t['lema2']}</text>

  {''.join(pasos)}

  <rect x="{x0}" y="262" width="{ancho_barra}" height="26" rx="8" fill="{PANEL}" stroke="{BORDE}"/>
  <rect class="barra" x="{x0}" y="262" width="{ancho_barra}" height="26" rx="8" fill="{ACENTO}"/>

  <g class="antes">
    <text class="gb" x="{x0}" y="248">{fmt['antes']} GB</text>
    <text class="etq" x="{x0 + 104}" y="247">{t['antes']}</text>
  </g>
  <g class="despues">
    <text class="gb" x="{x0}" y="248">{fmt['despues']} GB</text>
    <text class="etq" x="{x0 + 104}" y="247">{t['despues']}</text>
  </g>
  <g class="resultado">
    <text class="dato" x="{x0 + ancho_barra}" y="247" text-anchor="end">{fmt['veces']}× {t['menor']} · +{fmt['perdida']}% {t['perdida']}</text>
  </g>
  <g class="archivo">
    <rect x="{x0 + int(ancho_barra * fraccion) + 16}" y="263" width="212" height="24" rx="12" fill="none" stroke="{ACENTO_SUAVE}"/>
    <text x="{x0 + int(ancho_barra * fraccion) + 122}" y="279" text-anchor="middle" font-size="12" fill="{ACENTO_SUAVE}">model-int4.gguf → llama.cpp · Android</text>
  </g>

  <text class="nota" x="{x0}" y="322">{t['nota'].format(**fmt)}</text>
  <text class="nota" x="{x0}" y="342">pip install "octuma[hf,gguf]"   ·   octuma quantize Qwen/{m['modelo']}-Instruct</text>
</svg>
"""


def _corta(texto: str, maximo: int = 29) -> str:
    """El detalle tiene que caber en la caja del paso."""
    if len(texto) > maximo:
        raise ValueError(f"no cabe en la caja ({len(texto)} > {maximo}): {texto!r}")
    return texto


def main() -> None:
    m = medicion()
    logo = base64.b64encode((IMG / "logo-128.png").read_bytes()).decode()
    for idioma, nombre in (("en", "hero.svg"), ("es", "hero-es.svg")):
        destino = IMG / nombre
        destino.write_text(svg(idioma, m, logo), encoding="utf-8")
        print(f"{destino.relative_to(RAIZ)}  {destino.stat().st_size / 1024:.0f} KB")
    print(
        f"{m['modelo']}: {m['antes']:.2f} GB -> {m['despues']:.2f} GB, "
        f"{m['veces']:.2f}x, +{m['perdida']:.2f}%"
    )


if __name__ == "__main__":
    main()
