"""Compara las RESPUESTAS del modelo original y el cuantizado.

La perplejidad puede quedar casi igual y aun asi el modelo cuantizado repetir
palabras, inventar datos o perder el idioma. Esta prueba es la que convence a
quien lee el README: las dos respuestas, lado a lado, sobre las mismas
preguntas y con la misma semilla.

    python scripts/sanity_chat.py --model Qwen/Qwen2.5-3B-Instruct --quant out/qwen3b-int4-g32

Escribe runs/sanity__<short>.md listo para pegar en el README.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PREGUNTAS = [
    "Explica en dos frases que es la cuantizacion de modelos.",
    "¿Cual es la capital de Australia?",
    "Escribe una funcion de Python que invierta una cadena.",
    "¿Cuanto es 17 por 24? Muestra el procedimiento.",
    "Traduce al ingles: 'El gato duerme en la ventana'.",
    "Nombra tres planetas del sistema solar.",
    "¿Que es mas pesado, un kilo de plomo o un kilo de plumas?",
    "Resume en una frase la trama de Don Quijote.",
    "Escribe un haiku sobre la lluvia.",
    "¿En que año llego el hombre a la Luna?",
    "Explica la diferencia entre una lista y una tupla en Python.",
    "¿Cual es el rio mas largo del mundo?",
    "Convierte 100 grados Fahrenheit a Celsius.",
    "Da tres consejos para dormir mejor.",
    "¿Que hace el comando 'git rebase'?",
    "Escribe una consulta SQL que cuente filas por categoria.",
    "¿Por que el cielo es azul?",
    "Corrige: 'Ayer yo fui al tienda y compre pan'.",
    "¿Que significa que un modelo tenga 7 mil millones de parametros?",
    "Termina el refran: 'Mas vale pajaro en mano...'",
]


def responder(model, tok, pregunta: str, max_new: int = 120) -> tuple[str, float]:
    mensajes = [{"role": "user", "content": pregunta}]
    texto = tok.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)
    entrada = tok(texto, return_tensors="pt").to(model.device)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        salida = model.generate(
            **entrada, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.perf_counter() - t0
    nuevos = salida[0][entrada.input_ids.shape[1]:]
    # tokens por segundo: lo unico que le importa a quien lo corre en un telefono
    return tok.decode(nuevos, skip_special_tokens=True).strip(), len(nuevos) / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--quant", required=True, help="carpeta .tq de Octuma")
    ap.add_argument("--short", default="qwen3b")
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from octuma.export.tq import load_quantized

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print("cargando original...")
    original = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": "cuda"},
    ).eval()

    filas = []
    vel_orig = []
    for p in PREGUNTAS:
        r, v = responder(original, tok, p, args.max_new)
        filas.append({"pregunta": p, "original": r})
        vel_orig.append(v)
        print(f"  [orig] {p[:40]}...")
    del original
    torch.cuda.empty_cache()

    print("cargando cuantizado...")
    # load_quantized rellena un esqueleto de HF: se crea vacio y en CPU para no
    # pagar dos veces la memoria del modelo.
    esqueleto = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, low_cpu_mem_usage=True,
    )
    quant = load_quantized(esqueleto, args.quant, device="cuda").to("cuda").eval()

    vel_quant = []
    for fila in filas:
        r, v = responder(quant, tok, fila["pregunta"], args.max_new)
        fila["cuantizado"] = r
        vel_quant.append(v)
        print(f"  [quant] {fila['pregunta'][:40]}...")

    prom_o = sum(vel_orig) / len(vel_orig)
    prom_q = sum(vel_quant) / len(vel_quant)

    out = Path("runs") / f"sanity__{args.short}.md"
    lineas = [
        f"# Original vs cuantizado — {args.model}",
        "",
        "Mismas preguntas, generacion sin muestreo (`do_sample=False`) y la misma",
        "semilla, asi que cualquier diferencia viene de la cuantizacion y no del azar.",
        "",
        f"- Velocidad original: **{prom_o:.1f} tok/s**",
        f"- Velocidad cuantizado: **{prom_q:.1f} tok/s**",
        "",
    ]
    iguales = 0
    for fila in filas:
        if fila["original"].strip() == fila["cuantizado"].strip():
            iguales += 1
        lineas += [
            f"### {fila['pregunta']}",
            "",
            "**Original**",
            "",
            "```", fila["original"], "```",
            "",
            "**Cuantizado**",
            "",
            "```", fila["cuantizado"], "```",
            "",
        ]
    lineas.insert(7, f"- Respuestas identicas palabra por palabra: **{iguales}/{len(filas)}**\n")
    out.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\nescrito {out} · identicas {iguales}/{len(filas)} · {prom_o:.1f} vs {prom_q:.1f} tok/s")


if __name__ == "__main__":
    main()
