# Grupos de 32 contra grupos de 64

Fecha: 2026-10-07. El barrido principal (`runs/*__w20s2048_float16.json`) se
corrió con **grupos de 64**, que era el valor por defecto entonces. Desde que
existe el exportador a GGUF, Octuma usa **32**. Faltaba saber cuánto cambia
la calidad entre uno y otro.

## Cómo se midió

Octuma 0.1.5 instalada desde PyPI, con la configuración por defecto
(`QuantConfig()`: 4 bits, grupos de 32, GPTQ + AWQ, búsqueda de escala) y el
mismo protocolo del barrido: calibración de 128 ventanas de 2048 tokens de
wikitext-2 train, evaluación en wikitext-2 test con 20 ventanas de 2048, en
float16. PC con RTX 4060 (8 GB), torch 2.14.1+cu126.

El original en FP16 dio lo mismo que en el barrido (13.8180 contra 13.8185 y
9.3715 contra 9.3716), así que las dos series se pueden comparar.

## Resultados

| Modelo | Pérdida con 64 | Pérdida con 32 | Memoria con 64 | Memoria con 32 |
|---|---|---|---|---|
| Qwen2.5-0.5B | +5.23% | **+2.83%** | 0.740 GB | 0.757 GB |
| Qwen2.5-1.5B | +2.31% | **+2.03%** | 1.650 GB | 1.712 GB |

Datos en `runs/qwen0.5b__gptq-awq-int4-g32__w20s2048_float16.json` y
`runs/qwen1.5b__gptq-awq-int4-g32__w20s2048_float16.json`.

## Lo que dice

- **Grupos de 32 pierden menos calidad en los dos modelos.** La diferencia es
  grande en el 0.5B (casi la mitad) y pequeña en el 1.5B.
- **Cuestan poca memoria**: 2.3% más en el 0.5B y 3.7% más en el 1.5B, por
  guardar el doble de escalas.
- Los `.gguf` de esta corrida, medidos dentro de llama.cpp, dan 12.700 (0.5B)
  y 8.499 (1.5B), a 0.1% de los modelos publicados (12.710 y 8.486), que
  también son de grupos de 32.

## Lo que falta

**El 3B y el 7B no se midieron con grupos de 32.** El 3B cabe justo en una
tarjeta de 8 GB y tarda bastante más; se dejó para otra ocasión. Es lo que
convendría medir después, con este mismo protocolo, para saber si la ventaja
de 32 se mantiene al crecer el modelo o se hace tan pequeña como en el 1.5B.
