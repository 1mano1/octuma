# Comparativa contra otras herramientas — Qwen2.5-3B-Instruct

Fecha: 2026-09-20. Corpus wikitext-2 test, ventanas de 2048 tokens.

## 1. Contra bitsandbytes (motor de PyTorch/Transformers)

Mismo evaluador de TinyQ, 20 ventanas, mismas condiciones.

| Herramienta | Perplejidad | Memoria | Perdida vs FP16 |
|---|---|---|---|
| FP16 | 8.347 | 6.79 GB | — |
| **TinyQ GPTQ+AWQ** | **8.549** | 2.76 GB | **+2.4%** |
| TinyQ GPTQ | 8.578 | 2.76 GB | +2.8% |
| bitsandbytes NF4 | 8.906 | 2.63 GB | +6.7% |
| bitsandbytes FP4 | 13.343 | 2.63 GB | +59.9% |

**TinyQ hace menos de la mitad de dano que bitsandbytes NF4**, el cuantizador
por defecto de Hugging Face y el de QLoRA, con un 5% mas de memoria.

Script: `scripts/compare_baselines.py`.

## 2. Contra llama.cpp (su propio motor)

Los valores absolutos de llama.cpp **no son comparables** con los de arriba:
cada motor trocea y promedia distinto (su FP16 da 7.33 donde TinyQ mide 8.35).
Lo comparable es cuanto se degrada cada uno **respecto a su propio FP16**, y
por eso el GGUF de TinyQ se midio dentro de llama.cpp, no fuera.

| Modelo | Perplejidad | vs su F16 | Tamano |
|---|---|---|---|
| F16 | 7.330 | — | 6.18 GB |
| Q4_K_M | 7.824 | +6.7% | 1.93 GB |
| Q4_0 | 8.163 | +11.4% | 1.82 GB |
| **GGUF de TinyQ** | **15.494** | **+111%** | 2.30 GB |

### El export a GGUF esta roto

El mismo modelo, leido desde `.tq` con el motor de TinyQ, da **8.55**: bien.
Exportado a GGUF y leido por llama.cpp da **15.49**: mas del doble.

Aviso al comparar: el GGUF sale de los modelos **`-int4-g32`** (grupos de 32)
y el `.tq` medido es de **grupos de 64**. No son el mismo modelo. Grupos de 32
deberian dar mejor calidad, no peor, asi que no explica un +111%, pero el
primer paso es medir el `.tq` de g32 para partir el problema en dos.

Ya descartado: el exportador usa **Q4_1 con bloques de 32**, que es correcto, y
convierte bien los parametros (`d = scale`, `m = -zero * scale`). Quedan como
sospechosos el vocabulario y los metadatos (`_write_vocab`, `_write_metadata`):
un tokenizador mal escrito dispara la perplejidad aunque los pesos esten bien,
y encaja con un modelo que responde pero mal.

Es el bug mas caro del proyecto porque GGUF es el formato que usa la app
Android (Lumen). **Los .gguf publicados en Hugging Face estan degradados y no
deben anunciarse hasta arreglar esto.** Las carpetas `.tq` estan bien.

## 3. Calidad de las respuestas

20 preguntas, sin muestreo y con la misma semilla (`runs/sanity__qwen3b.md`):

- Identicas palabra por palabra: **2/20**. Las otras cambian la redaccion pero
  no el contenido: ambos responden "Canberra", ambos descomponen bien 17x24.
- **Velocidad: 14.4 tok/s el original contra 4.2 el cuantizado.**

El cuantizado es **3.4 veces mas lento**: `QuantLinear` desempaqueta los 4 bits
a FP16 en cada multiplicacion con PyTorch normal, sin kernel CUDA dedicado.
El README no puede prometer velocidad; el argumento es memoria y calidad.
