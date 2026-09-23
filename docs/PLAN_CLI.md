# Plan: la CLI que no da dolores de cabeza

Objetivo: que alguien instale Octuma y cuantice su primer modelo **sin tomar una
sola decision**. Quien quiera afinar, que pueda; quien no, que no tenga que.

## El problema de hoy

Los valores por defecto de `octuma quantize` **no son la configuracion ganadora
de nuestros propios experimentos**:

| Opcion | Hoy | Lo que gana en el barrido |
|---|---|---|
| `method` | `gptq` | `gptq` **+ awq** |
| `awq` | `False` | `True` |
| `device` | `cpu` | la GPU, si hay |
| `dtype` | `float32` | `float16` |
| `samples` x `seq_len` | 64 x 512 | 128 x 2048 |

Quien corra el comando obvio obtiene **peores resultados que los de nuestra
tabla** y, en CPU con float32, tarda tanto que va a concluir que Octuma es lento.

Regla: **lo que esta medido como mejor es lo que debe pasar sin pedirlo.**

## Los tres comandos

### 1. Cuantizar

```bash
octuma quantize Qwen/Qwen2.5-3B-Instruct
```

- Sin `--out`: se deduce `qwen2.5-3b-instruct-int4/`.
- Sin elegir metodo: usa GPTQ+AWQ, que gana en los cuatro modelos medidos.
- Detecta la GPU solo; en CPU avisa que va a tardar y ofrece `--bits 8`.
- **Antes de empezar** comprueba la memoria y dice "este modelo pide ~13 GB y
  tienes 8 libres", en vez de reventar a los diez minutos.
- Al terminar imprime: pesaba X, ahora pesa Y, perdiste Z% de calidad.

### 2. Comparar

```bash
octuma compare qwen2.5-3b-instruct-int4
```

Mide el cuantizado **contra su original** y saca una sola tabla: perplejidad,
memoria, tokens por segundo y porcentaje de perdida. Es la respuesta a
"¿quedo bien?".

Con `--against bnb-nf4` lo compara tambien contra bitsandbytes, que es lo que
hace hoy `scripts/compare_baselines.py`.

### 3. Probar

```bash
octuma try qwen2.5-3b-instruct-int4                 # chat en la terminal
octuma try qwen2.5-3b-instruct-int4 --side-by-side  # 20 preguntas, dos columnas
```

Es la respuesta a "¿sigue hablando bien?", que la perplejidad no contesta. El
modo `--side-by-side` es `scripts/sanity_chat.py` convertido en comando.

Los comandos avanzados (`analyze`, `export`, y todas las perillas) siguen
existiendo: solo dejan de ser lo primero que ve la gente.

## La documentacion, en cinco piezas

1. **README** — los primeros 30 segundos: que es, `pip install octuma`, los tres
   comandos y **la tabla contra bitsandbytes**. Esa tabla convence: va arriba.
2. **Guia de inicio** — cuantizar el primer modelo de punta a punta, con las
   salidas reales de la terminal, cuanto tarda y cuanta memoria pide.
3. **Referencia de comandos** — generada de la propia CLI para que no se
   desincronice con el codigo.
4. **Como funciona** — calibrar, cuantizar, evaluar, exportar. Ya esta bien
   explicado en el README actual y merece pagina propia.
5. **Cuando algo falla** — la que casi nadie escribe y evita la mitad de los
   issues: "me quede sin memoria" -> `--bits 8` o modelo mas chico; "va
   lentisimo" -> estas en CPU; "dice tonterias" -> group_size muy grande.

## Lo que hay que decir y no ocultar

En Qwen 3B, el modelo cuantizado genera a **4.2 tok/s contra 14.4 del
original**: 3.4 veces mas lento. `QuantLinear` desempaqueta los 4 bits a FP16
en cada multiplicacion con PyTorch normal, sin kernel CUDA dedicado, que es lo
que si traen bitsandbytes y AutoGPTQ.

No invalida el proyecto — el destino real es el telefono via GGUF y llama.cpp,
que si tiene kernels optimizados — pero el README **no puede prometer
velocidad**. El argumento es memoria y calidad. Decirlo antes de que lo diga
otro es lo que da credibilidad, y deja planteado el siguiente trabajo: un
kernel de multiplicacion en 4 bits.
