# Los tres GGUF publicados, medidos antes y despues del arreglo

Fecha: 2026-09-20. Maquina: RTX 4060 (8 GB), `llama-perplexity` build b11065
con CUDA 12.4. Codigo en `0e3a69f`.

Hasta hoy el arreglo del export estaba medido **solo en el 0.5B**, y con
ventanas de 512. Aqui estan los tres modelos publicados, cada uno contra su
propia version re-exportada, **con el mismo protocolo**: wikitext-2 test,
20 ventanas de 2048 tokens, `-ngl 99`.

| Modelo | Publicado en HF | Re-exportado | Mejora |
|---|---|---|---|
| Qwen2.5-0.5B | 27.484 | **12.710** | 2.16x |
| Qwen2.5-1.5B | 24.462 | **8.486** | 2.88x |
| Qwen2.5-3B | 15.453 | **7.512** | 2.06x |

## Lo que los numeros dejan ver

**El bug se comia el beneficio de crecer.** Con los archivos publicados, pasar
del 0.5B al 1.5B —triplicar el modelo— movia la perplejidad de 27.48 a 24.46,
un 11%. Arreglados, la misma comparacion va de 12.71 a 8.49: un 33%. Quien
bajara el 1.5B degradado estaba pagando 2.6x el tamano por casi nada.

**Los pesos siempre estuvieron bien.** `verify_gguf.py` da el mismo peor error
relativo en el archivo roto y en el arreglado (0.00555 en el 0.5B): es el
redondeo del embedding en Q8_0, nada mas. Lo unico que cambio fueron tres
metadatos.

## verify_gguf.py cazaria el bug hoy

Corrido contra el 0.5B **publicado**, devuelve codigo 1 y los nombra:

    rope.freq_base=10000.0 pero el modelo usa 1000000.0
    tokenizer.pre='default' cuando qwen2 necesita 'qwen2'
    falta tokenizer.eos_token_id: el modelo no sabra cuando parar

Contra los tres re-exportados devuelve 0. Por eso puede ir en CI: impide
publicar un GGUF roto, que es exactamente lo que paso.

## El protocolo es reproducible

Las mediciones del 3B de la otra maquina se reprodujeron aqui dentro del 0.3%:

| | otra maquina | esta | diferencia |
|---|---|---|---|
| 3B roto | 15.494 | 15.4525 | 0.27% |
| 3B arreglado | 7.492 | 7.5118 | 0.26% |

Eso valida `scripts/make_wikitext_txt.py`, que genera el `.txt` uniendo el
split con `\n\n` igual que `octuma.evaluate.wikitext2_ids`. Sin ese script la
medicion de GGUF no era repetible: el corpus se armaba a mano.

Con GPU cada medicion tarda **8-18 segundos**. En CPU eran horas, que es la
razon por la que esto llevaba pendiente desde el arreglo.

## Comandos

```bash
python scripts/make_wikitext_txt.py --out out/wikitext2.txt
octuma export out/qwen3b-tq --out out/qwen3b-int4-fix.gguf
python scripts/verify_gguf.py out/qwen3b-tq out/qwen3b-int4-fix.gguf
llama-perplexity -m out/qwen3b-int4-fix.gguf -f out/wikitext2.txt \
    -c 2048 --chunks 20 -ngl 99
```
