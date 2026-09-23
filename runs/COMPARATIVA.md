# Comparativa contra otras herramientas — Qwen2.5-3B-Instruct

Fecha: 2026-09-20. Corpus wikitext-2 test, ventanas de 2048 tokens.

## 1. Contra bitsandbytes (motor de PyTorch/Transformers)

Mismo evaluador de Octuma, 20 ventanas, mismas condiciones.

| Herramienta | Perplejidad | Memoria | Perdida vs FP16 |
|---|---|---|---|
| FP16 | 8.347 | 6.79 GB | — |
| **Octuma GPTQ+AWQ** | **8.549** | 2.76 GB | **+2.4%** |
| Octuma GPTQ | 8.578 | 2.76 GB | +2.8% |
| bitsandbytes NF4 | 8.906 | 2.63 GB | +6.7% |
| bitsandbytes FP4 | 13.343 | 2.63 GB | +59.9% |

**Octuma hace menos de la mitad de dano que bitsandbytes NF4**, el cuantizador
por defecto de Hugging Face y el de QLoRA, con un 5% mas de memoria.

Script: `scripts/compare_baselines.py`.

## 2. Contra llama.cpp (su propio motor)

Los valores absolutos de llama.cpp **no son comparables** con los de arriba:
cada motor trocea y promedia distinto (su FP16 da 7.33 donde Octuma mide 8.35).
Lo comparable es cuanto se degrada cada uno **respecto a su propio FP16**, y
por eso el GGUF de Octuma se midio dentro de llama.cpp, no fuera.

| Modelo | Perplejidad | vs su F16 | Tamano |
|---|---|---|---|
| F16 | 7.330 | — | 6.18 GB |
| **Octuma int4** | **7.492** | **+2.2%** | 2.30 GB |
| Q4_K_M | 7.824 | +6.7% | 1.93 GB |
| Q4_0 | 8.163 | +11.4% | 1.82 GB |
| Octuma int4 *antes del arreglo* | 15.494 | +111% | 2.30 GB |

**Octuma tambien le gana a llama.cpp**: +2.2% contra el +6.7% de Q4_K_M, que es
el formato mas usado para correr modelos en local. Tres veces menos dano, a
cambio de 0.37 GB mas de archivo.

Importa para la app Android: los modelos que produce Octuma son mejores que los
GGUF estandar que cualquiera bajaria de Hugging Face.

### El bug del export, resuelto

La fila de +111% era un GGUF con tres metadatos mal escritos, no un problema de
cuantizacion. Detalle en `CLAUDE.md`. Se arreglo el 2026-09-20 y el archivo
re-exportado es el que da 7.492.

Los tres modelos publicados se volvieron a medir con este mismo protocolo en
`runs/NOTA_gguf_reexport.md`: el 0.5B pasa de 27.484 a 12.710, el 1.5B de
24.462 a 8.486 y el 3B de 15.453 a 7.512. Esa medicion del 3B reproduce las
dos filas de esta tabla dentro del 0.3%, en otra maquina y otro build de
llama.cpp.

## 3. Calidad de las respuestas

20 preguntas, sin muestreo y con la misma semilla (`runs/sanity__qwen3b.md`):

- Identicas palabra por palabra: **2/20**. Las otras cambian la redaccion pero
  no el contenido: ambos responden "Canberra", ambos descomponen bien 17x24.
- **Velocidad: 14.4 tok/s el original contra 4.2 el cuantizado.**

El cuantizado es **3.4 veces mas lento**: `QuantLinear` desempaqueta los 4 bits
a FP16 en cada multiplicacion con PyTorch normal, sin kernel CUDA dedicado.
El README no puede prometer velocidad; el argumento es memoria y calidad.
