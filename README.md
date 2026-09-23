# Octuma

Cuantizacion INT4/INT8 de modelos de lenguaje, pensada para que corran en
laptops modestas, servidores chicos y telefonos Android.

Un modelo de 3B en FP16 pide ~6.2 GB de memoria. Octuma lo baja a **2.4 GB
perdiendo 2.4% de calidad**, y el archivo resultante corre en llama.cpp, que es
lo que usa cualquier telefono o laptop sin GPU.

```bash
pip install "octuma[hf,gguf] @ git+https://github.com/1mano1/octuma.git"

octuma quantize Qwen/Qwen2.5-3B-Instruct    # cuantiza, sin elegir nada
octuma compare qwen2.5-3b-instruct-int4     # ¿quedo bien?
octuma try qwen2.5-3b-instruct-int4         # ¿sigue hablando bien?
```

No hay que decidir metodo, bits ni carpeta: los valores por defecto son la
configuracion que gana en nuestras propias mediciones, la GPU se detecta sola y
te avisa **antes de descargar nada** si el modelo no va a caber en tu memoria.

> **El proyecto se llamaba TinyQ** y se renombro a Octuma el 2026-09-22, para
> compartir nombre con la app Android que corre estos modelos. Lo que se
> instala, lo que se importa y el comando de la terminal son los tres `octuma`.
> Los repos de los modelos en Hugging Face todavia llevan el nombre viejo.

## Instalacion

```bash
# lo normal: cuantizar modelos de Hugging Face y exportar a GGUF
pip install "octuma[hf,gguf] @ git+https://github.com/1mano1/octuma.git"

# solo el motor, si ya traes torch y no vas a exportar
pip install "git+https://github.com/1mano1/octuma.git"
```

Necesita Python 3.10+ y PyTorch. Para cuantizar con GPU hace falta una
instalacion de torch con CUDA; en CPU funciona igual, solo mas lento.

| Extra | Para que |
|---|---|
| `hf` | Bajar modelos y datasets de Hugging Face (`transformers`, `datasets`) |
| `gguf` | Exportar a GGUF para llama.cpp y Android |
| `dev` | `pytest` y `ruff`, para desarrollar |

## Modelos listos para usar

Tres modelos de la familia Qwen2.5 ya cuantizados, cada uno con la carpeta
`.tq` (PyTorch) y el `.gguf` (llama.cpp / Android):

| Modelo | Tamaño INT4 | Perplejidad | vs original | Licencia del original |
|---|---|---|---|---|
| [qwen0.5b-int4-TinyQ](https://huggingface.co/Imanol11/qwen0.5b-int4-TinyQ) | 0.52 GB | 12.710 | +3.73% | Apache 2.0 |
| [qwen1.5b-int4-TinyQ](https://huggingface.co/Imanol11/qwen1.5b-int4-TinyQ) | 1.32 GB | 8.486 | +2.11% | Apache 2.0 |
| [qwen3b-int4-TinyQ](https://huggingface.co/Imanol11/qwen3b-int4-TinyQ) | 2.40 GB | 7.512 | +2.43% | **Qwen Research (no comercial)** |

> **El 3B no se puede usar comercialmente.** A diferencia del resto de la
> familia, [`Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
> no es Apache 2.0 sino [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE):
> solo investigacion y evaluacion. El modelo cuantizado hereda esa condicion.
> Si lo que quieres es algo comercial, el 1.5B y el 0.5B son Apache 2.0, y el
> [7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) tambien.

```bash
hf download Imanol11/qwen3b-int4-TinyQ --local-dir qwen3b
llama-cli -m qwen3b/qwen3b-int4.gguf -p "Hola"
```

## Que tan bien queda

### Contra los formatos de llama.cpp, dentro de llama.cpp

Es la comparacion que importa para correr en local: el `.gguf` de Octuma medido
con el mismo motor, el mismo corpus y las mismas ventanas que sus rivales.

![Degradacion por tamaño del modelo](docs/img/gguf-por-tamano.png#gh-light-mode-only)
![Degradacion por tamaño del modelo](docs/img/gguf-por-tamano-dark.png#gh-dark-mode-only)

| Modelo | Octuma INT4 | Q4_K_M | Q4_0 |
|---|---|---|---|
| Qwen2.5-0.5B | +3.73% | **+2.63%** | +13.17% |
| Qwen2.5-1.5B | **+2.11%** | +4.74% | +8.35% |
| Qwen2.5-3B | **+2.43%** | +6.21% | +11.02% |

Calidad perdida frente al mismo original en F16; menos es mejor, y en negritas
el mejor de cada fila.

**Al crecer el modelo, Q4_K_M se degrada cada vez mas (2.63% → 4.74% → 6.21%)
mientras Octuma se mantiene plano (3.73% → 2.11% → 2.43%).** En el 3B, Octuma
hace **2.6 veces menos dano** que Q4_K_M, que es el formato mas usado para
correr modelos en local.

**En el 0.5B, Q4_K_M gana**: pierde menos y ocupa menos (0.40 GB contra 0.52).
El cruce esta entre 0.5B y 1.5B. Si tu modelo es diminuto, usa Q4_K_M; Octuma
rinde cuando el modelo crece, que es justo cuando la memoria aprieta.

Detalle por modelo, con grafica y tabla completa, en
[`runs/COMPARATIVA_GGUF.md`](runs/COMPARATIVA_GGUF.md).

### Contra bitsandbytes, dentro de PyTorch

![Comparativa contra otros cuantizadores](docs/img/comparativa-herramientas.png#gh-light-mode-only)
![Comparativa contra otros cuantizadores](docs/img/comparativa-herramientas-dark.png#gh-dark-mode-only)

Sobre el mismo Qwen2.5-3B, con el mismo evaluador y las mismas ventanas, Octuma
hace **menos de la mitad de dano** que bitsandbytes NF4, el cuantizador por
defecto de Hugging Face y el que usa QLoRA:

| Herramienta | Perplejidad | Memoria | Perdida |
|---|---|---|---|
| FP16 (sin cuantizar) | 8.347 | 6.79 GB | — |
| **Octuma GPTQ+AWQ** | **8.549** | 2.76 GB | **+2.4%** |
| Octuma GPTQ | 8.578 | 2.76 GB | +2.8% |
| bitsandbytes NF4 | 8.906 | 2.63 GB | +6.7% |
| bitsandbytes FP4 | 13.343 | 2.63 GB | +59.9% |

> Las perplejidades de esta tabla y las de la anterior **no se comparan entre
> si**: cada motor trocea y promedia distinto, y el mismo F16 da 8.347 aqui y
> 7.334 en llama.cpp. Lo comparable es siempre la perdida dentro de un motor.

Y el dano baja conforme el modelo crece, tambien en PyTorch:

![Degradacion segun el tamano del modelo](docs/img/degradacion-por-tamano.png#gh-light-mode-only)
![Degradacion segun el tamano del modelo](docs/img/degradacion-por-tamano-dark.png#gh-dark-mode-only)

Los numeros salen de `runs/*.json` y las graficas se regeneran con
`python scripts/grafica_comparativa.py` y `python scripts/grafica_gguf.py`. El
detalle esta en [`runs/COMPARATIVA.md`](runs/COMPARATIVA.md) y
[`runs/COMPARATIVA_GGUF.md`](runs/COMPARATIVA_GGUF.md).

### Lo que Octuma **no** hace

Los modelos cuantizados **generan mas lento** en PyTorch: 4.2 tokens/s contra
14.4 del original en Qwen 3B. `QuantLinear` desempaqueta los 4 bits en cada
multiplicacion sin un kernel dedicado, que es lo que si traen bitsandbytes y
AutoGPTQ. El argumento de Octuma es **memoria y calidad**, no velocidad: para
correr rapido, exporta a GGUF y usa llama.cpp.

Tampoco esta comparado contra AutoGPTQ ni GPTQModel, que son los rivales
tecnicos mas directos: hoy no instalan con setuptools moderno.

## Como funciona

| Paso | Que hace |
|---|---|
| **Calibrar** | Toma ventanas aleatorias de un corpus real (wikitext-2, C4 o tus propios textos) y mide, capa por capa, que direcciones de la entrada importan. Se guarda como `H = 2·XXᵀ`. |
| **Cuantizar** | Agrupa los pesos en bloques de 32 por fila, saca una escala y un punto cero por bloque, y los lleva a 4 bits. El redondeo no es ciego: con GPTQ, el error de cada columna se reparte entre las columnas que faltan usando la Hessiana. |
| **Evaluar** | Perplejidad en wikitext-2 con ventanas sin solape, memoria por tipo de capa y tokens por segundo. |
| **Exportar** | Formato propio `.tq` (safetensors empaquetado) para PyTorch, y **GGUF** para llama.cpp y Android. |

### Precision mixta guiada por datos

No todas las capas sufren igual. `octuma analyze` mide, capa por capa, cuanto
cambia su **salida** al cuantizar, no cuanto cambian sus pesos:

```
||ΔW·X||² = tr(ΔW · H · ΔWᵀ)
```

Con eso ordena las capas por dano real y arma un plan: las mas sensibles suben
a 8 bits y el resto se queda en 4, sin pasarse del promedio de bits que pidas.

```bash
octuma analyze Qwen/Qwen2.5-0.5B-Instruct --target-bits 4.5 -o plan.json
octuma quantize Qwen/Qwen2.5-0.5B-Instruct --out out/qwen-mix --plan plan.json
```

### Por que por grupos y asimetrico

Una sola escala por tensor se arruina con un solo peso atipico. Con grupos de 32
el dano queda contenido en su bloque. Y como los pesos casi nunca estan
centrados en cero, guardar tambien un punto cero (asimetrico) aprovecha los 16
niveles completos en vez de desperdiciar la mitad del rango.

El costo es pequeno: por cada grupo se guarda una escala en FP16 y un punto
cero, o sea ~4.5 bits por peso en vez de 4.

## Android y llama.cpp

El exportador escribe GGUF con los pesos en **Q4_1**, que es exactamente el
mismo formato que un grupo asimetrico de 32 de Octuma:

```
Octuma:  w = (q - z)·s        Q4_1:  w = d·q + m        d = s,  m = -z·s
```

Por eso el exportador exige grupos de 32, que ya es el valor por defecto:

```bash
octuma quantize <modelo>
octuma export <carpeta-int4> --out modelo-int4.gguf
python scripts/verify_gguf.py <carpeta-int4> modelo-int4.gguf
```

**Verifica siempre antes de publicar un GGUF.** Los pesos pueden estar
perfectos y el archivo salir roto por los metadatos: un `rope_theta` mal
escrito **duplica la perplejidad sin tocar un solo peso**, y el modelo sigue
respondiendo frases cortas con normalidad, asi que a simple vista no se nota.
Nos paso, y los tres modelos publicados estuvieron degradados hasta que
`verify_gguf.py` aprendio a revisar los metadatos ademas de los pesos. Devuelve
codigo de error, asi que puede ir en CI.

**Verificado de verdad**: Qwen2.5-0.5B cuantizado con Octuma, exportado a GGUF y
ejecutado en llama.cpp:

```
$ llama-cli -m qwen05b-int4.gguf -p "La capital de Francia es" --temp 0 -n 32 -st -ngl 0
La capital de Francia es París.

prompt: 229 tok/s · generacion: 64 tok/s   (solo CPU, 6 hilos)
```

Tiene que ser `llama-cli` y no `llama-completion`: estos son modelos *Instruct*
y `llama-cli` les aplica su plantilla de chat. Con completado crudo, el 0.5B y
`--temp 0` se quedan repitiendo la pregunta.

## Uso desde Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from octuma.calibrate import load_wikitext2
from octuma.quantizer import QuantConfig, quantize_model
from octuma.export.tq import save_quantized

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

calib = load_wikitext2(tok, n_samples=64, seq_len=512)
report = quantize_model(model, calib, QuantConfig(bits=4, group_size=32))

print(report.summary())
save_quantized(model, "out/qwen-int4")
```

Cargar uno ya cuantizado:

```python
from transformers import AutoConfig, AutoModelForCausalLM
from octuma.export.tq import load_quantized

cfg = AutoConfig.from_pretrained("out/qwen-int4")
model = load_quantized(AutoModelForCausalLM.from_config(cfg), "out/qwen-int4")
```

Precision mixta, para las capas que mas sufren:

```python
cfg = QuantConfig(bits=4, group_size=32, bits_overrides={"mlp.down_proj": 8})
```

## Resultados del barrido

33 corridas sobre Qwen2.5 (0.5B, 1.5B, 3B y 7B), en `runs/*.json`. El orden de
los metodos es **identico en los cuatro modelos**, que es la mejor señal de que
la implementacion hace lo que dice:

```
GPTQ+AWQ  >  GPTQ  >  AWQ-RTN  >  RTN
```

Degradacion del mejor metodo frente a FP16: 5.2% (0.5B), 2.3% (1.5B), 2.4%
(3B), 1.8% (7B con GPTQ solo: sus corridas con AWQ no cabian en memoria).
INT8 es practicamente gratis (+0.03%) pero solo comprime 1.91x; INT4 comprime
3.66x, contando solo los pesos: el modelo entero baja 59% en el 3B, porque los
embeddings se quedan en FP16.

> El barrido se corrio con **grupos de 64**, que era el valor por defecto
> entonces. Hoy la CLI usa 32, que es lo que exige el exportador a GGUF y lo
> que llevan dentro los tres modelos publicados. Grupos mas chicos guardan mas
> escalas y por eso comprimen menos: 3.66x con 64 contra 3.37x con 32 en el
> 0.5B. Las tablas de llama.cpp de mas arriba si son con 32; lo que no esta
> medido es cuanta perplejidad cambia entre un tamaño de grupo y el otro.

### Cuanta calibracion hace falta

GPTQ estima esa misma `H = 2·XXᵀ` por capa. Con menos tokens que dimensiones
tenga la capa, esa matriz es singular y la compensacion de error se vuelve
ruido: el resultado sale **peor** que no usar GPTQ. Lo medimos en el 0.5B:

| Tokens de calibracion | GPTQ | GPTQ + AWQ |
|---|---|---|
| 512 (menos que las 896 dimensiones) | peor que RTN | mucho peor que RTN |
| 8192 | +7.5% | +4.2% |

Por eso `quantize_model` avisa cuando la calibracion no alcanza. Regla practica:
al menos 10 veces la dimension de la capa mas ancha.

### Un resultado negativo que vale la pena

La busqueda de escala (`--search-scale`) ayuda en 0.5B y 3B, y **estorba** en
1.5B y 7B. En el 7B reduce el error de reconstruccion de los pesos (0.09503 →
0.09152) pero **empeora** la perplejidad (7.4386 → 7.5189).

Minimizar el error de los pesos no equivale a preservar la calidad del modelo.
Detalle en [`runs/NOTA_rtn_search.md`](runs/NOTA_rtn_search.md).

## Estado

- [x] Cuantizacion por grupos INT2/INT4/INT8, simetrica y asimetrica
- [x] Empaquetado real de bits (INT4 = medio byte)
- [x] GPTQ con compensacion de error e Hessiana amortiguada
- [x] Cuantizacion secuencial bloque por bloque (el error se propaga como en la vida real)
- [x] Capa `QuantLinear` que dequantiza al vuelo
- [x] Precision mixta por capa (`bits_overrides`)
- [x] Guardar y cargar `.tq`
- [x] Perplejidad, memoria y velocidad
- [x] Exportar a GGUF Q4_1 / Q8_0 (llama.cpp / Android)
- [x] Analisis de sensibilidad por capa y plan de precision mixta
- [x] GGUF validado corriendo en llama.cpp y medido contra sus formatos
- [ ] Cuantizacion del cache KV
- [ ] Kernel rapido para INT4 (hoy se dequantiza al vuelo)
- [ ] App Android de demostracion

## Desarrollo

```bash
git clone https://github.com/1mano1/octuma.git
cd octuma
pip install -e ".[hf,gguf,dev]"
pytest -q          # 60 pruebas, segundos en CPU
ruff check .
```

Las pruebas usan modelos Llama diminutos creados al vuelo, sin descargar nada.
Las dos reglas que ya costaron caro estan en
[`CONTRIBUTING.md`](CONTRIBUTING.md): un default nunca tapa un fallo, y los
defaults son lo que se midio como mejor.

## Licencia

El codigo de Octuma es **MIT** (ver [LICENSE](LICENSE)).

**Los modelos son otra cosa.** Un modelo cuantizado es una obra derivada: se
queda con la licencia del original, y la de Octuma no la afloja. Por eso cada
repo publicado lleva dentro la licencia de su modelo base:

| Base | Licencia | Uso comercial |
|---|---|---|
| Qwen2.5-0.5B / 1.5B / 7B-Instruct | Apache 2.0 | si |
| Qwen2.5-3B-Instruct | [Qwen Research](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) | **no** |

Si cuantizas otro modelo con Octuma, revisa su licencia antes de publicarlo:
varias familias populares (Llama, Gemma) traen condiciones propias que viajan
con los pesos derivados.

Built with Qwen.
