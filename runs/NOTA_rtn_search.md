# Aviso: `rtn-int4-search` en 0.5B, 1.5B y 3B no es valido

## Que pasa

En esos tres modelos, `rtn-int4` y `rtn-int4-search` dan resultados **identicos
hasta el ultimo decimal**, tanto en perplejidad como en error de reconstruccion:

| Modelo | rtn-int4 | rtn-int4-search | mean_rel_fro |
|---|---|---|---|
| 0.5B | 15.5923 | 15.5923 | identico |
| 1.5B | 10.1731 | 10.1731 | identico |
| 3B | 8.9759 | 8.9759 | identico |
| 7B | 7.4386 | **7.5189** | 0.09503 vs 0.09152 |

Solo el 7B muestra dos corridas realmente distintas.

## Por que

No es un error del codigo: `search_scale` se lee bien del YAML y
`quantize_tensor` enruta bien a `quantize_groupwise_searched`.

El problema es de **proceso**. El commit `0d9ac7e` ("La busqueda de escala
tambien se aplica en RTN") llego al pod en el `git pull` de las 06:05 UTC, pero
el barrido ya llevaba horas corriendo. Python carga los modulos al arrancar:
el proceso viejo siguio ejecutando en memoria la version anterior, en la que
RTN ignoraba el flag. Los modelos chicos corrieron en ese proceso; el 7B corrio
en el proceso relanzado a las 07:35, ya con el codigo nuevo.

Un `git pull` a mitad de un barrido no cambia nada del barrido en curso, y nada
en los resultados lo delataba. Por eso ahora cada JSON guarda `code.commit`,
`code.dirty` y `code.started_at`, sellados al arrancar el proceso.

## Que hay que hacer

1. **Volver a correr** `rtn-int4-search` en 0.5B, 1.5B y 3B con `--force`. Los
   numeros actuales son duplicados de `rtn-int4` y no deben publicarse.
2. **No sacar conclusiones todavia** del unico dato valido (7B). Dice algo
   incomodo y quiza interesante: la busqueda **reduce** el error de pesos
   (0.09503 -> 0.09152) pero **empeora** la perplejidad (7.4386 -> 7.5189).
   Si se confirma en los otros modelos, es un resultado que vale la pena
   reportar: minimizar el error de reconstruccion de pesos no equivale a
   preservar la calidad del modelo, que es justo el supuesto sobre el que
   descansa la busqueda de escala por grupo.
