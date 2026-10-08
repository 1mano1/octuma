# Verificación de la 0.1.5 con un modelo real

Fecha: 2026-10-07. Qwen2.5-0.5B-Instruct, solo CPU (laptop de 16 GB de RAM),
con el recorrido "Solo CPU" del README. Sirve para comprobar que la versión
funciona de punta a punta; **no reemplaza las tablas**, que se midieron con la
calibración completa (128 ventanas de 2048 tokens) y esta usa la corta.

## Qué se corrió

```bash
octuma quantize Qwen/Qwen2.5-0.5B-Instruct --samples 32 --seqlen 512 --out corta
octuma compare corta --windows 5 --seqlen 512
octuma try corta -p "What is the capital of Australia?"
octuma export corta --out corta.gguf --name qwen0.5b-int4-Octuma
python scripts/verify_gguf.py corta corta.gguf
llama-perplexity -m corta.gguf -f out/wikitext2.txt -c 2048 --chunks 20 -ngl 0
```

Entorno: Python 3.11.0, torch 2.14.0+cpu, transformers 5.17.0, llama.cpp b11065.

## Resultados

| Paso | Resultado |
|---|---|
| `quantize` | 168 capas a INT4, grupos de 32, GPTQ + AWQ. Compresión 3.37x, error medio 0.1070, 19.5 min |
| `compare` (5 ventanas de 512) | original 18.726, cuantizado 19.362: **+3.4%**, 1.94x más chico en memoria |
| `try` | "The capital of Australia is Canberra." |
| `export` | 0.52 GB (519 160 064 bytes), `general.file_type` = Q4_1, con plantilla de chat |
| `verify_gguf.py` | pesos y metadatos correctos |
| `llama-perplexity` (20 ventanas de 2048) | **12.7583** ± 0.244 |
| `llama-cli` | "The capital of Australia is Canberra." a 53 tok/s (CPU) |

Frente al F16 de `runs/gguf__qwen0.5b.json` (12.2523), 12.7583 es **+4.13%**.
El modelo publicado, con la calibración completa, da 12.7096 (+3.73%): cuatro
veces más texto de calibración recorta la pérdida en 0.4 puntos.

## Lo que demuestra

- **La 0.1.5 cuantiza igual que la 0.1.4.** Un `.gguf` hecho el 2026-09-29 con
  la 0.1.4 y la misma calibración da 12.7583, idéntico hasta el cuarto
  decimal. Los cambios de la 0.1.5 no tocaron la cuantización por defecto.
- **El GGUF nuevo corrige dos metadatos** que los tres modelos publicados
  todavía llevan mal: la etiqueta `F16` y la plantilla de chat ausente.
  `verify_gguf.py` marca los dos en un archivo como los publicados.


## Re-exportación de los tres modelos publicados

Fecha: 2026-10-07, en la PC con RTX 4060 (Python 3.10.0, torch 2.5.1+cu121,
transformers 5.12.1, llama.cpp b11065 con CUDA). Las 129 pruebas pasan ahí.

Se re-exportaron las tres carpetas `.tq` publicadas, **sin recuantizar**:

```bash
octuma export <carpeta-tq> --out <slug>-int4.gguf --name <slug>-int4-Octuma
python scripts/verify_gguf.py <carpeta-tq> <slug>-int4.gguf
llama-perplexity -m <slug>-int4.gguf -f out/wikitext2.txt -c 2048 --chunks 20 -ngl 99
```

| Modelo | Perplejidad | Publicada | `verify_gguf.py` | Bytes | SHA-256 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 12.7096 | 12.7096 | correcto | 519 162 624 | `81a76b5061c454178be00dcbf21bd574a745cc3bcb2c5446893c8bc5b0485a81` |
| Qwen2.5-1.5B | 8.4864 | 8.4864 | correcto | 1 321 322 432 | `696021abd23ff9ff902b3c167ccbd18f432f100950b763e9d426a1224e211a42` |
| Qwen2.5-3B | 7.5118 | 7.5118 | correcto | 2 402 231 616 | `3f502c8531f863463ea192aa2b0597f6fdef9bf6e4f10010bfd27b8c10dbda82` |

La perplejidad es idéntica a la de `runs/gguf__*.json` porque los pesos no
cambian. Cambian tres metadatos: `general.file_type` pasa de F16 a Q4_1,
`general.name` deja de decir TinyQ y aparece `tokenizer.chat_template`. Cada
archivo crece 2 560 bytes, que es la plantilla.

Los tres contestan "The capital of Australia is Canberra." en `llama-cli`, a
189, 156 y 92 tokens por segundo en la RTX 4060.

**Estos archivos todavía no están en Hugging Face.** Al subirlos cambian el
tamaño y el SHA-256 de cada `.gguf`: quien los verifique por huella tiene que
actualizarla a la vez.
