# TinyQ

Cuantizacion INT4/INT8 de modelos de lenguaje, pensada para que corran en
laptops modestas, servidores chicos y telefonos Android.

Un modelo de 8B en FP16 pide ~16 GB de memoria y no cabe en casi ningun equipo
normal. TinyQ lo baja a ~4.3 GB conservando la mayor parte de la calidad, con
cuatro pasos: **calibrar, cuantizar, evaluar y exportar**.

```bash
pip install -e ".[hf,gguf]"

tinyq quantize Qwen/Qwen2.5-3B-Instruct    # cuantiza, sin elegir nada
tinyq compare qwen2.5-3b-instruct-int4     # ¿quedo bien?
tinyq try qwen2.5-3b-instruct-int4         # ¿sigue hablando bien?
```

No hay que decidir metodo, bits ni carpeta: los valores por defecto son la
configuracion que gana en nuestras propias mediciones, la GPU se detecta sola y
te avisa **antes de descargar nada** si el modelo no va a caber en tu memoria.

## Que tan bien queda

![Comparativa contra otros cuantizadores](docs/img/comparativa-herramientas.png#gh-light-mode-only)
![Comparativa contra otros cuantizadores](docs/img/comparativa-herramientas-dark.png#gh-dark-mode-only)

Sobre el mismo Qwen2.5-3B, con el mismo evaluador y las mismas ventanas, TinyQ
hace **menos de la mitad de dano** que bitsandbytes NF4, el cuantizador por
defecto de Hugging Face y el que usa QLoRA:

| Herramienta | Perplejidad | Memoria | Perdida |
|---|---|---|---|
| FP16 (sin cuantizar) | 8.347 | 6.79 GB | — |
| **TinyQ GPTQ+AWQ** | **8.549** | 2.76 GB | **+2.4%** |
| TinyQ GPTQ | 8.578 | 2.76 GB | +2.8% |
| bitsandbytes NF4 | 8.906 | 2.63 GB | +6.7% |
| bitsandbytes FP4 | 13.343 | 2.63 GB | +59.9% |

Y el dano baja conforme el modelo crece, que es justo lo que interesa:

![Degradacion segun el tamano del modelo](docs/img/degradacion-por-tamano.png#gh-light-mode-only)
![Degradacion segun el tamano del modelo](docs/img/degradacion-por-tamano-dark.png#gh-dark-mode-only)

Y dentro de llama.cpp, comparado contra sus propios formatos sobre el mismo
Qwen 3B (20 ventanas de 2048, F16 de referencia 7.330):

| Formato | Perplejidad | Perdida |
|---|---|---|
| **TinyQ int4** | **7.492** | **+2.2%** |
| Q4_K_M | 7.824 | +6.7% |
| Q4_0 | 8.163 | +11.4% |

Q4_K_M es el formato mas usado para correr modelos en local, y TinyQ le hace
**tres veces menos dano** al modelo.

Los numeros salen de `runs/*.json` y las graficas se regeneran con
`python scripts/grafica_comparativa.py`. El detalle completo esta en
[`runs/COMPARATIVA.md`](runs/COMPARATIVA.md).

### Lo que TinyQ **no** hace

Los modelos cuantizados **generan mas lento** en PyTorch: 4.2 tokens/s contra
14.4 del original en Qwen 3B. `QuantLinear` desempaqueta los 4 bits en cada
multiplicacion sin un kernel dedicado, que es lo que si traen bitsandbytes y
AutoGPTQ. El argumento de TinyQ es **memoria y calidad**, no velocidad: para
correr rapido, exporta a GGUF y usa llama.cpp.

## Como funciona

| Paso | Que hace |
|---|---|
| **Calibrar** | Toma ventanas aleatorias de un corpus real (wikitext-2, C4 o tus propios textos) y mide, capa por capa, que direcciones de la entrada importan. Se guarda como `H = 2·XXᵀ`. |
| **Cuantizar** | Agrupa los pesos en bloques de 64 por fila, saca una escala y un punto cero por bloque, y los lleva a 4 bits. El redondeo no es ciego: con GPTQ, el error de cada columna se reparte entre las columnas que faltan usando la Hessiana. |
| **Evaluar** | Perplejidad en wikitext-2 con ventanas sin solape, memoria por tipo de capa y tokens por segundo. |
| **Exportar** | Formato propio `.tq` (safetensors empaquetado) para PyTorch, y **GGUF** para llama.cpp y Android. |

### Precision mixta guiada por datos

No todas las capas sufren igual. `tinyq analyze` mide, capa por capa, cuanto
cambia su **salida** al cuantizar, no cuanto cambian sus pesos:

```
||ΔW·X||² = tr(ΔW · H · ΔWᵀ)
```

Con eso ordena las capas por dano real y arma un plan: las mas sensibles suben
a 8 bits y el resto se queda en 4, sin pasarse del promedio de bits que pidas.

```bash
tinyq analyze Qwen/Qwen2.5-0.5B-Instruct --target-bits 4.5 -o plan.json
tinyq quantize Qwen/Qwen2.5-0.5B-Instruct --out out/qwen-mix --plan plan.json
```

### Android

El exportador escribe GGUF con los pesos en **Q4_1**, que es exactamente el
mismo formato que un grupo asimetrico de 32 de TinyQ:

```
TinyQ:  w = (q - z)·s        Q4_1:  w = d·q + m        d = s,  m = -z·s
```

Por eso el exportador exige grupos de 32, que ya es el valor por defecto:

```bash
tinyq quantize <modelo>
tinyq export <carpeta-int4> --out modelo-int4.gguf
python scripts/verify_gguf.py <carpeta-int4> modelo-int4.gguf
```

**Verifica siempre antes de publicar un GGUF.** Los pesos pueden estar
perfectos y el archivo salir roto por los metadatos: un `rope_theta` mal
escrito duplica la perplejidad sin tocar un solo peso, y el modelo sigue
respondiendo frases cortas con normalidad, asi que a simple vista no se nota.
`verify_gguf.py` revisa las dos cosas y devuelve codigo de error si algo no
cuadra.

El `.gguf` resultante lo lee llama.cpp, y de ahi corre en Android.

**Verificado de verdad**: Qwen2.5-0.5B cuantizado con TinyQ, exportado a GGUF y
ejecutado en llama.cpp (build b11057):

```
$ llama-completion -m qwen05b-int4.gguf -p "La capital de Francia es" --temp 0
La capital de Francia es la ciudad de Paris.

prompt eval: 121 tokens/s · eval: 77 tokens/s   (CPU de laptop, 6 hilos)
```

El archivo pesa 0.52 GB. `scripts/verify_gguf.py` compara tensor por tensor el
GGUF contra el modelo original: 291 de 291 presentes, peor error relativo
0.0056 (los embeddings en Q8_0; las capas cuantizadas quedan en 0.0006).

### Por que por grupos y asimetrico

Una sola escala por tensor se arruina con un solo peso atipico. Con grupos de 64
el dano queda contenido en su bloque. Y como los pesos casi nunca estan
centrados en cero, guardar tambien un punto cero (asimetrico) aprovecha los 16
niveles completos en vez de desperdiciar la mitad del rango.

El costo es pequeno: por cada 64 pesos se guarda una escala en FP16 y un punto
cero de un byte, o sea ~4.4 bits por peso en vez de 4.

## Resultados medidos

Medidos, no estimados. Qwen2.5-0.5B en CPU, INT4 con grupos de 64, calibracion
de 8192 tokens, evaluado en wikitext-2 con 4 ventanas de 512 tokens:

| Metodo | Perplejidad | vs FP16 |
|---|---|---|
| FP16 (sin cuantizar) | 19.218 | - |
| RTN (redondeo directo) | 21.970 | +14.3% |
| AWQ + RTN | 20.952 | +9.0% |
| GPTQ | 20.656 | +7.5% |
| **GPTQ + AWQ** | **20.029** | **+4.2%** |

Cada pieza aporta: el flujo completo baja la perdida de 14.3% a 4.2% en el caso
mas dificil, que es el modelo mas chico. Los pesos de las capas lineales pasan
de 0.72 GB a 0.20 GB (3.66x).

Tres notas honestas:

1. Un modelo de 0.5B es el peor escenario: tiene poca redundancia. Los numeros
   tipicos de 4 bits (1-2% de perdida) son de modelos de 3B para arriba.
2. La calibracion de esta tabla es corta (8192 tokens) porque corrio en una
   laptop. El barrido de `experiments/sweep.yaml` usa 262144.
3. La memoria total baja menos que los pesos porque aqui los embeddings
   (151936 x 896) pesan mas que todas las capas lineales juntas y no se
   cuantizan. En modelos grandes son una fraccion minima.

Reproducirlo:

```bash
python scripts/run_sweep.py experiments/smoke.yaml   # ~35 min de CPU
python scripts/run_sweep.py --table
```

### Cuanta calibracion hace falta

GPTQ estima `H = X·Xᵀ` por capa. Con menos tokens que dimensiones tenga la
capa, esa matriz es singular y la compensacion de error se vuelve ruido: el
resultado sale **peor** que no usar GPTQ. Lo medimos:

| Tokens de calibracion | GPTQ | GPTQ + AWQ |
|---|---|---|
| 512 (menos que las 896 dimensiones) | peor que RTN | mucho peor que RTN |
| 8192 | +7.5% | +4.2% |

Por eso `quantize_model` avisa cuando la calibracion no alcanza. Regla practica:
al menos 10 veces la dimension de la capa mas ancha.

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
- [x] GGUF validado corriendo en llama.cpp (genera texto correcto)
- [ ] Cuantizacion del cache KV
- [ ] Kernel rapido para INT4 (hoy se dequantiza al vuelo)
- [ ] App Android de demostracion

## Uso desde Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinyq.calibrate import load_wikitext2
from tinyq.quantizer import QuantConfig, quantize_model
from tinyq.export.tq import save_quantized

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

calib = load_wikitext2(tok, n_samples=64, seq_len=512)
report = quantize_model(model, calib, QuantConfig(bits=4, group_size=64))

print(report.summary())
save_quantized(model, "out/qwen-int4")
```

Precision mixta, para las capas que mas sufren:

```python
cfg = QuantConfig(bits=4, group_size=64, bits_overrides={"mlp.down_proj": 8})
```

## Desarrollo

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[hf,dev]"
.venv\Scripts\pytest -q
```

Las pruebas corren en CPU en segundos: usan modelos Llama diminutos creados al
vuelo, sin descargar nada.

## Licencia

MIT
