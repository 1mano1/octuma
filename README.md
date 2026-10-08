<div align="center">

<img src="https://raw.githubusercontent.com/1mano1/octuma/main/docs/img/hero-es.svg" alt="Octuma cuantiza modelos de lenguaje a 4 bits en cuatro pasos: calibrar, cuantizar, evaluar y exportar. Qwen2.5-3B pasa de 6.18 GB en FP16 a 2.40 GB en INT4 perdiendo 2.43% de calidad." width="880">

<p>
  <a href="https://github.com/1mano1/octuma/actions/workflows/tests.yml"><img src="https://github.com/1mano1/octuma/actions/workflows/tests.yml/badge.svg" alt="Pruebas"></a>
  <a href="https://pypi.org/project/octuma/"><img src="https://img.shields.io/pypi/v/octuma.svg?color=5B7CFF&labelColor=080A14&label=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/octuma/"><img src="https://img.shields.io/pypi/pyversions/octuma.svg?color=5B7CFF&labelColor=080A14&label=python" alt="Versiones de Python"></a>
  <a href="https://huggingface.co/Imanol11"><img src="https://img.shields.io/badge/modelos-Hugging%20Face-8FA6FF?labelColor=080A14" alt="Modelos en Hugging Face"></a>
  <a href="https://github.com/1mano1/octuma/blob/main/LICENSE"><img src="https://img.shields.io/badge/licencia-MIT-9AA3C7?labelColor=080A14" alt="Licencia MIT"></a>
</p>

<p>
  <b><a href="#empieza-en-un-minuto">Inicio rápido</a></b> ·
  <a href="#cómo-funciona">Cómo funciona</a> ·
  <a href="#qué-tan-bien-queda">Resultados</a> ·
  <a href="#modelos-listos-para-usar">Modelos</a> ·
  <a href="#comandos">Comandos</a> ·
  <a href="#estructura-del-proyecto">Estructura</a> ·
  <a href="#android-y-llamacpp">Android</a> ·
  <a href="https://github.com/1mano1/octuma/blob/main/llms.txt">llms.txt</a>
</p>

<p><b>Español</b> · <a href="https://github.com/1mano1/octuma/blob/main/README.en.md">English</a></p>

<sub><b>Agentes de IA y LLM:</b> lean <a href="https://github.com/1mano1/octuma/blob/main/llms.txt"><code>llms.txt</code></a>: resume el proyecto, los comandos y dónde está cada cosa.</sub>

</div>

Octuma es una librería de Python que cuantiza modelos de lenguaje a 4 y 8 bits
para que corran en laptops modestas, servidores chicos y teléfonos Android. Un
Qwen2.5 de 3B pasa de **6.18 GB a 2.40 GB perdiendo 2.4% de calidad**, y el
archivo que sale corre en llama.cpp, que es lo que usa cualquier equipo sin GPU.

Todo corre en tu máquina. Octuma solo se conecta a internet para bajar de
Hugging Face el modelo y el texto de calibración que tú le pides.

## Qué hace

- **Cuantiza** a INT4 o INT8 con GPTQ y AWQ, por grupos de 32 pesos.
- **Compara** el modelo cuantizado con el original en una sola tabla: perplejidad, memoria y velocidad.
- **Te deja probarlo** en la terminal, o con las mismas diez preguntas al original y al cuantizado.
- **Exporta a GGUF** para llama.cpp y Android, y a su formato propio `.tq` para PyTorch.
- **Analiza qué capas sufren más** y arma un plan de precisión mixta (unas a 8 bits, el resto a 4).
- **Avisa antes de descargar** si el modelo no va a caber en tu memoria.

## Cómo funciona

```
 Modelo de Hugging Face o carpeta local  (Qwen, Llama, Mistral…)
        │
        ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  Octuma   (corre en tu equipo, con GPU o solo CPU)              │
 │  ─────────────────────────────────────────────────────────────  │
 │  1. Calibrar    128 ventanas de texto real → H = 2·XXᵀ por capa │
 │  2. Cuantizar   AWQ escala los canales  →  GPTQ redondea y      │
 │                 reparte el error, bloque por bloque             │
 │  3. Evaluar     perplejidad, memoria y tokens por segundo       │
 │  4. Exportar    .tq (PyTorch)  y  .gguf en Q4_1 (llama.cpp)     │
 └─────────────────────────────────────────────────────────────────┘
        │
        ▼
 carpeta .tq  +  modelo-int4.gguf  →  llama.cpp · Android · PyTorch
```

| Paso | Qué hace | Dónde está |
|---|---|---|
| **Calibrar** | Toma ventanas al azar de un corpus real (wikitext-2, C4 o tus propios `.txt`) y las pasa por el modelo. De ahí sale, capa por capa, qué direcciones de la entrada importan: `H = 2·XXᵀ`. | [`calibrate.py`](https://github.com/1mano1/octuma/blob/main/src/octuma/calibrate.py) |
| **Cuantizar** | Agrupa los pesos de 32 en 32 por fila, con una escala y un punto cero por grupo, y los lleva a 4 bits. El redondeo no es ciego: AWQ le da más resolución a los canales que reciben activaciones grandes, y GPTQ reparte el error de cada columna entre las que faltan usando `H`. Va bloque por bloque, y cada bloque recibe la salida ya cuantizada del anterior. | [`quantizer.py`](https://github.com/1mano1/octuma/blob/main/src/octuma/quantizer.py), [`quant/`](https://github.com/1mano1/octuma/tree/main/src/octuma/quant) |
| **Evaluar** | Perplejidad en wikitext-2 con ventanas sin solape, memoria por tipo de capa y tokens por segundo. | [`evaluate.py`](https://github.com/1mano1/octuma/blob/main/src/octuma/evaluate.py) |
| **Exportar** | Guarda la carpeta `.tq` (safetensors con los bits empaquetados) y escribe GGUF con los pesos en Q4_1. | [`export/`](https://github.com/1mano1/octuma/tree/main/src/octuma/export) |

### Por qué por grupos y asimétrico

Una sola escala para todo el tensor se arruina con un solo peso atípico. Con
grupos de 32 el daño queda dentro de su grupo. Y como los pesos casi nunca
están centrados en cero, guardar también un punto cero (asimétrico) aprovecha
los 16 niveles completos en vez de desperdiciar parte del rango.

El costo: cada grupo guarda una escala en FP16 y un punto cero de un byte. En
la carpeta `.tq` son **4.75 bits por peso** con grupos de 32, no 4. En GGUF el
bloque Q4_1 ocupa 5 bits por peso.

### Precisión mixta guiada por datos

No todas las capas sufren igual. `octuma analyze` mide, capa por capa, cuánto
cambia su **salida** al cuantizar, no cuánto cambian sus pesos:

```
||ΔW·X||² = tr(ΔW · H · ΔWᵀ)
```

Con eso ordena las capas por daño real y arma un plan: las más sensibles suben
a 8 bits y el resto se queda en 4, sin pasarse del promedio de bits que pidas.

```bash
octuma analyze Qwen/Qwen2.5-0.5B-Instruct --target-bits 4.5 -o plan.json
octuma quantize Qwen/Qwen2.5-0.5B-Instruct --out qwen-mixto --plan plan.json
```

> Hasta la versión 0.1.4, `--plan` subía a 8 bits la capa elegida **en todos
> los bloques**, no solo en el que decía el plan, y el promedio de bits se
> pasaba del pedido. Está corregido en la 0.1.5. Un modelo con precisión mixta
> no se puede exportar a GGUF: el exportador solo escribe Q4_1.

## Empieza en un minuto

```bash
pip install "octuma[hf,gguf]"

octuma quantize Qwen/Qwen2.5-3B-Instruct    # cuantiza, sin elegir nada
octuma compare qwen2.5-3b-instruct-int4     # ¿quedó bien?
octuma try qwen2.5-3b-instruct-int4         # ¿sigue hablando bien?
octuma export qwen2.5-3b-instruct-int4 --out qwen3b-int4.gguf
```

No hay que decidir método, bits ni carpeta. Los valores por defecto son la
configuración que gana en nuestras propias mediciones (GPTQ + AWQ, grupos de
32, 128 ventanas de 2048 tokens), la GPU se detecta sola y Octuma te avisa
**antes de descargar nada** si el modelo no va a caber en tu memoria.

Sin GPU también funciona, pero cuantizar un modelo grande es lento. Para
probar en CPU, usa el 0.5B con la calibración corta de
[Pruébalo en tu PC](#pruébalo-en-tu-pc).

## Qué tan bien queda

### Contra los formatos de llama.cpp, dentro de llama.cpp

Es la comparación que importa para correr en local: el `.gguf` de Octuma medido
con el mismo motor, el mismo corpus y las mismas ventanas que sus rivales.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/1mano1/octuma/main/docs/img/gguf-por-tamano-dark.png">
  <img src="https://raw.githubusercontent.com/1mano1/octuma/main/docs/img/gguf-por-tamano.png" alt="Calidad perdida según el tamaño del modelo: Octuma INT4 contra Q4_K_M y Q4_0" width="760">
</picture>

Calidad perdida frente al mismo original en F16, y tamaño del archivo. Menos es
mejor en las dos; en negritas, la menor pérdida de cada fila.

| Modelo | Octuma INT4 | Q4_K_M | Q4_0 |
|---|---|---|---|
| Qwen2.5-0.5B | +3.73% · 0.52 GB | **+2.63%** · 0.40 GB | +13.17% · 0.35 GB |
| Qwen2.5-1.5B | **+2.11%** · 1.32 GB | +4.74% · 0.99 GB | +8.35% · 0.93 GB |
| Qwen2.5-3B | **+2.43%** · 2.40 GB | +6.21% · 1.93 GB | +11.02% · 1.82 GB |

**Al crecer el modelo, Q4_K_M se degrada cada vez más (2.63% → 4.74% → 6.21%)
mientras Octuma se mantiene plano (3.73% → 2.11% → 2.43%).** En el 3B, Octuma
pierde **2.6 veces menos calidad** que Q4_K_M, el formato más usado para correr
modelos en local.

**El archivo de Octuma es más grande en los tres**: entre 24% y 33% más que
Q4_K_M. Es el precio de la escala y el punto cero por cada 32 pesos. Lo que
compras con ese espacio es calidad.

**En el 0.5B, Q4_K_M gana en las dos cosas**: pierde menos y ocupa menos. El
cruce está entre 0.5B y 1.5B. Si tu modelo es diminuto, usa Q4_K_M.

Detalle por modelo, con gráfica y tabla completa, en
[`runs/COMPARATIVA_GGUF.md`](https://github.com/1mano1/octuma/blob/main/runs/COMPARATIVA_GGUF.md).

### Contra bitsandbytes, dentro de PyTorch

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/1mano1/octuma/main/docs/img/comparativa-herramientas-dark.png">
  <img src="https://raw.githubusercontent.com/1mano1/octuma/main/docs/img/comparativa-herramientas.png" alt="Perplejidad de Qwen2.5-3B con Octuma y con bitsandbytes" width="760">
</picture>

Sobre el mismo Qwen2.5-3B, con el mismo evaluador y las mismas ventanas, Octuma
pierde **menos de la mitad** que bitsandbytes NF4, el cuantizador por defecto
de Hugging Face y el que usa QLoRA, a cambio de 5% más de memoria:

| Herramienta | Perplejidad | Memoria | Pérdida |
|---|---|---|---|
| FP16 (sin cuantizar) | 8.347 | 6.79 GB | — |
| **Octuma GPTQ+AWQ** | **8.549** | 2.76 GB | **+2.4%** |
| Octuma GPTQ | 8.578 | 2.76 GB | +2.8% |
| bitsandbytes NF4 | 8.906 | 2.63 GB | +6.7% |
| bitsandbytes FP4 | 13.343 | 2.63 GB | +59.9% |

> Las perplejidades de esta tabla y las de la anterior **no se comparan entre
> sí**: cada motor trocea y promedia distinto, y el mismo original da 8.347
> aquí y 7.334 en llama.cpp. Lo comparable es siempre la pérdida dentro de un
> motor. Esta tabla, además, se midió con grupos de 64; la de llama.cpp, con 32.

Y la pérdida baja conforme el modelo crece, también en PyTorch:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/1mano1/octuma/main/docs/img/degradacion-por-tamano-dark.png">
  <img src="https://raw.githubusercontent.com/1mano1/octuma/main/docs/img/degradacion-por-tamano.png" alt="Pérdida de calidad según el tamaño del modelo, en PyTorch" width="760">
</picture>

Los números salen de `runs/*.json` y las gráficas se regeneran con
`python scripts/grafica_comparativa.py` y `python scripts/grafica_gguf.py`. El
detalle está en [`runs/COMPARATIVA.md`](https://github.com/1mano1/octuma/blob/main/runs/COMPARATIVA.md).

### Lo que Octuma no hace

- **No acelera en PyTorch.** El modelo cuantizado genera más lento: 4.2
  tokens/s contra 14.4 del original en Qwen 3B con GPU. `QuantLinear`
  desempaqueta los 4 bits en cada multiplicación, sin un kernel dedicado. El
  argumento de Octuma es **memoria y calidad**; para correr rápido, exporta a
  GGUF y usa llama.cpp.
- **No está comparado contra AutoGPTQ ni GPTQModel**, los rivales técnicos más
  directos. Cuando se intentó (septiembre de 2026) no instalaban con
  setuptools moderno.
- **El exportador a GGUF solo conoce cuatro arquitecturas**: `llama`, `qwen2`,
  `mistral` y `gemma`. Solo la familia Qwen2.5 está medida de punta a punta.

## Modelos listos para usar

Tres modelos de la familia Qwen2.5 ya cuantizados y públicos. Cada repo trae la
carpeta `.tq` (PyTorch) y el `.gguf` (llama.cpp y Android):

| Modelo | Archivo GGUF | Perplejidad | vs. original | Licencia del original |
|---|---|---|---|---|
| [qwen0.5b-int4-Octuma](https://huggingface.co/Imanol11/qwen0.5b-int4-Octuma) | 0.52 GB | 12.710 | +3.73% | Apache 2.0 |
| [qwen1.5b-int4-Octuma](https://huggingface.co/Imanol11/qwen1.5b-int4-Octuma) | 1.32 GB | 8.486 | +2.11% | Apache 2.0 |
| [qwen3b-int4-Octuma](https://huggingface.co/Imanol11/qwen3b-int4-Octuma) | 2.40 GB | 7.512 | +2.43% | **Qwen Research (no comercial)** |

```bash
hf download Imanol11/qwen1.5b-int4-Octuma --local-dir qwen1.5b
llama-cli -m qwen1.5b/qwen1.5b-int4.gguf -p "Hola"
```

> **El 3B no se puede usar comercialmente.** A diferencia del resto de la
> familia, [`Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
> no es Apache 2.0 sino [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE):
> solo investigación y evaluación. El modelo cuantizado hereda esa condición.
> Para algo comercial, el 0.5B y el 1.5B son Apache 2.0, y el
> [7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) original también.

Dos cosas que conviene saber de esos tres archivos, publicados cuando el
proyecto se llamaba TinyQ:

- Sus metadatos se llaman `tinyq.json`. Octuma los abre igual desde la 0.1.5;
  las versiones anteriores respondían que la carpeta no era de Octuma.
- El `.gguf` se anuncia como `F16` en llama.cpp y en el visor de Hugging Face,
  aunque sus pesos son Q4_1, y no trae la plantilla de chat. La calidad es la
  de la tabla; llama.cpp usa ChatML cuando falta la plantilla, que es la de
  Qwen. Las dos cosas se corrigen al re-exportarlos con la 0.1.5.

## Instalación

```bash
# lo normal: cuantizar modelos de Hugging Face y exportar a GGUF
pip install "octuma[hf,gguf]"

# la versión en desarrollo
pip install "octuma[hf,gguf] @ git+https://github.com/1mano1/octuma.git"
```

Necesita Python 3.10 o más nuevo y PyTorch.

**Instala los extras.** `pip install octuma` a secas deja fuera `transformers`,
y sin él solo funcionan `octuma --version` y `octuma info`. Se llaman
opcionales por cómo los nombra pip, no porque se puedan omitir.

| Extra | Para qué | Sin él |
|---|---|---|
| `hf` | `transformers` y `datasets` | no hay `quantize`, `compare`, `try`, `evaluate`, `analyze` ni `export` |
| `gguf` | Exportar a GGUF para llama.cpp y Android | no hay `export` |
| `dev` | `pytest` y `ruff`, para desarrollar | — |

### Con GPU NVIDIA

PyTorch con CUDA no está en PyPI: hay que instalarlo antes desde su propio
índice. Si se omite este paso, `octuma quantize` corre en CPU y solo lo dice
con un aviso amarillo.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "octuma[hf,gguf]"     # respeta el torch que ya está
```

`cu126` tiene PyTorch para Python 3.10 a 3.14. El error `No matching
distribution found for torch` casi siempre significa que el índice no tiene la
versión de Python instalada; `cu121`, por ejemplo, llega solo hasta la 3.12.

## Pruébalo en tu PC

Todo queda dentro de una carpeta que se borra al final. Los comandos son de
PowerShell; en Linux o macOS el entorno se activa con `source .venv/bin/activate`.
No hace falta descargar un modelo antes: `quantize` baja de Hugging Face el
Qwen2.5 0.5B (cerca de 1 GB, sin cuenta).

### Solo CPU

La calibración y la comparación van recortadas para que terminen en minutos y
no en horas.

```powershell
mkdir C:\prueba-octuma
cd C:\prueba-octuma
Set-ExecutionPolicy -Scope Process Bypass   # permite activar el entorno, solo en esta terminal
py -m venv .venv                            # con varias versiones: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install "octuma[hf,gguf]"
octuma --version

octuma quantize Qwen/Qwen2.5-0.5B-Instruct --samples 32 --seqlen 512 --out qwen05b
octuma compare qwen05b --windows 5 --seqlen 512
octuma try qwen05b -p "What is the capital of Australia?"
octuma export qwen05b --out qwen05b.gguf
```

`compare` pasa varios minutos sin imprimir nada mientras mide; no está trabado.

### Con GPU NVIDIA

Con la configuración completa, la misma de las tablas de arriba.

```powershell
mkdir C:\prueba-octuma
cd C:\prueba-octuma
Set-ExecutionPolicy -Scope Process Bypass
py -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "octuma[hf,gguf]"
python -c "import torch; print(torch.cuda.is_available())"   # debe decir True
octuma --version

octuma quantize Qwen/Qwen2.5-0.5B-Instruct --out qwen05b
octuma compare qwen05b
octuma try qwen05b -p "What is the capital of Australia?"
octuma export qwen05b --out qwen05b.gguf
```

Los números de la prueba en CPU no se comparan con las tablas, que usan 128
ventanas de 2048 tokens; los de GPU sí.

### Para borrar todo

```powershell
deactivate
cd C:\
Remove-Item -Recurse -Force C:\prueba-octuma
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
Remove-Item -Recurse -Force "$env:APPDATA\octuma"   # registro de la última versión usada
```

Los datos de wikitext quedan en `.cache\huggingface\hub\datasets--*`.

## Comandos

Los tres primeros son los de todos los días. `octuma <comando> --help` enseña
todas las opciones; la terminal habla en inglés.

| Comando | Qué hace | Opciones que más se usan |
|---|---|---|
| `octuma quantize <modelo>` | Calibra, cuantiza y guarda la carpeta `.tq`. `<modelo>` es un id de Hugging Face o una carpeta local. | `--out`, `--bits` (2, 3, 4 u 8), `--samples`, `--seqlen`, `--calib` (`wikitext2`, `c4` o una ruta a `.txt`), `--no-awq`, `--method rtn`, `--plan` |
| `octuma compare <carpeta>` | Mide el cuantizado y el original y los pone en una tabla, con la línea que resume: "2.46x smaller for +2.4% perplexity". | `--windows`, `--seqlen`, `--no-speed`, `--original` |
| `octuma try <carpeta>` | Chat en la terminal con el modelo cuantizado. | `-p "pregunta"` para una sola, `--side-by-side` para las diez preguntas contra el original, `--out` para guardarlas en Markdown |
| `octuma export <carpeta> --out <archivo.gguf>` | Convierte la carpeta `.tq` a GGUF. Exige 4 bits y grupos de 32, que son los valores por defecto. | `--name` |
| `octuma info <carpeta>` | Enseña qué hay dentro: modelo de origen, calibración, capas y bits. No carga los pesos. | — |
| `octuma evaluate <modelo>` | Mide perplejidad y memoria de un modelo, cuantizado o no. | `--windows`, `--seqlen`, `--speed`, `--out` para guardar el reporte en JSON |
| `octuma analyze <modelo>` | Ordena las capas por cuánto sufren a 4 bits y propone un plan de precisión mixta. | `--target-bits`, `--group`, `--out` |
| `octuma --version` | La versión instalada. | — |

Todos aceptan `--device auto`, `cpu` o `cuda` donde tiene sentido. `quantize`,
`compare` y `try` eligen la GPU solos; `evaluate` y `analyze` usan CPU salvo
que se les diga otra cosa.

## Uso desde Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from octuma.calibrate import load_wikitext2
from octuma.quantizer import QuantConfig, quantize_model
from octuma.export.tq import save_quantized

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

calib = load_wikitext2(tok, n_samples=64, seq_len=512)
report = quantize_model(model, calib, QuantConfig())

print(report.summary())
save_quantized(model, "out/qwen-int4")
```

`QuantConfig()` sin argumentos hace lo mismo que `octuma quantize`: 4 bits,
grupos de 32, GPTQ y AWQ. Hasta la 0.1.4 no era así: venía con grupos de 64 y
AWQ apagado, y lo que salía no se podía exportar a GGUF.

Cargar uno ya cuantizado:

```python
from transformers import AutoConfig, AutoModelForCausalLM
from octuma.export.tq import load_quantized

cfg = AutoConfig.from_pretrained("out/qwen-int4")
model = load_quantized(AutoModelForCausalLM.from_config(cfg), "out/qwen-int4")
```

Precisión mixta a mano: por tipo de capa en todos los bloques, o una capa
concreta con su número de bloque.

```python
cfg = QuantConfig(bits=4, group_size=32, bits_overrides={"mlp.down_proj": 8})
cfg = QuantConfig(bits=4, group_size=32, bits_overrides={"3.mlp.down_proj": 8})
```

## Estructura del proyecto

```
octuma/
├── src/octuma/            el paquete que se instala
│   ├── cli.py             los comandos de la terminal
│   ├── calibrate.py       arma las ventanas de texto de calibración
│   ├── quantizer.py       recorre el modelo bloque por bloque y lo cuantiza
│   ├── quant/
│   │   ├── core.py        la matemática: grupos, escalas y empaquetado de bits
│   │   ├── gptq.py        redondeo con compensación de error (GPTQ)
│   │   ├── awq.py         escalado de canales según las activaciones (AWQ)
│   │   ├── search.py      búsqueda de la escala que menos error deja por grupo
│   │   └── qlinear.py     QuantLinear, la capa que sustituye a nn.Linear
│   ├── sensitivity.py     qué capas sufren más y el plan de precisión mixta
│   ├── evaluate.py        perplejidad, memoria y velocidad
│   ├── export/
│   │   ├── tq.py          guardar y cargar la carpeta .tq
│   │   └── gguf_export.py escribir GGUF en Q4_1 para llama.cpp
│   └── logo.py            el pulpo que sale en la terminal
├── tests/                 pruebas: sin red, sin GPU y sin modelos reales
├── scripts/               andamiaje de experimentos; no se instala con pip
├── experiments/           las matrices del barrido (YAML)
├── runs/                  los resultados medidos (JSON) y las comparativas
├── docs/                  imágenes del README y el plan de la terminal
├── llms.txt               resumen del proyecto para agentes de IA
├── CHANGELOG.md           qué cambió en cada versión
└── CONTRIBUTING.md        cómo instalar, probar y proponer cambios
```

Lo que hace cada pieza, en el orden en que se usa:

| Pieza | Qué hace |
|---|---|
| `cli.py` | Traduce cada comando a llamadas de las piezas de abajo. Aquí viven los valores por defecto y los avisos de memoria. |
| `calibrate.py` | Junta el corpus, lo tokeniza y recorta ventanas al azar con una semilla fija. Devuelve un `CalibrationSet`. |
| `quantizer.py` | `quantize_model` encuentra los bloques del transformer, captura la entrada del primero y avanza de uno en uno: aplica AWQ, estima `H`, cuantiza cada capa lineal y la reemplaza por una `QuantLinear`. `QuantConfig` es la configuración. |
| `quant/core.py` | Cuantiza un tensor por grupos (`quantize_tensor`), lo reconstruye y empaqueta los valores: dos de 4 bits por byte. |
| `quant/gptq.py` | Cuantiza una matriz columna por columna y reparte el error de cada una entre las que faltan, con la inversa de `H` amortiguada. |
| `quant/awq.py` | Busca, por grupo de capas, el factor `s = mean\|x\|^α` que menos error de salida deja, y lo pliega en la capa anterior para que en inferencia no cueste nada. |
| `quant/search.py` | En vez de tomar el mínimo y el máximo del grupo, prueba varios recortes del rango y se queda con el de menor error. |
| `quant/qlinear.py` | `QuantLinear` guarda los pesos empaquetados y los reconstruye en cada pasada. Por eso ahorra memoria y no tiempo. |
| `sensitivity.py` | Mide el error de **salida** de cada capa a 4 y a 8 bits y elige cuáles subir sin pasarse del promedio pedido. |
| `evaluate.py` | Perplejidad por ventanas sin solape, bytes por tipo de capa y tokens por segundo generando. |
| `export/tq.py` | Escribe `model.tq.safetensors` y `octuma.json`, y reconstruye el modelo sobre un esqueleto de Hugging Face. |
| `export/gguf_export.py` | Convierte cada grupo de 32 en un bloque Q4_1 y escribe los metadatos y el vocabulario que llama.cpp necesita. |

### El formato `.tq`

Una carpeta `.tq` es lo que deja `octuma quantize`:

| Archivo | Qué trae |
|---|---|
| `model.tq.safetensors` | Por cada capa cuantizada: `qweight` (bits empaquetados, `uint8`), `scales` (FP16) y `zeros` (`uint8`). Lo demás, embeddings y normas, en FP16. |
| `octuma.json` | Versión del formato, bits y tamaño de grupo de cada capa, la configuración usada, el modelo de origen y la calibración. |
| `config.json`, `tokenizer.json` y compañía | Los del modelo original, sin cambios. |

`.tq` conserva su nombre de cuando el proyecto se llamaba TinyQ, porque vive
dentro de los modelos ya publicados.

## Android y llama.cpp

El exportador escribe GGUF con los pesos en **Q4_1**, que es exactamente el
mismo formato que un grupo asimétrico de 32 de Octuma:

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
respondiendo frases cortas con normalidad, así que a simple vista no se nota.
Nos pasó, y los tres modelos publicados estuvieron degradados hasta que
`verify_gguf.py` aprendió a revisar los metadatos además de los pesos. Devuelve
código de error, así que puede ir en CI. El script está en el repositorio, no
en el paquete de pip.

Qwen2.5-0.5B cuantizado con Octuma, exportado a GGUF y ejecutado en llama.cpp:

```
$ llama-cli -m qwen05b-int4.gguf -p "La capital de Francia es" --temp 0 -n 32 -st -ngl 0
La capital de Francia es París.

prompt: 229 tok/s · generación: 64 tok/s   (solo CPU, 6 hilos, una PC de escritorio)
```

Tiene que ser `llama-cli` y no `llama-completion`: estos son modelos *Instruct*
y `llama-cli` les aplica su plantilla de chat. Con completado crudo, el 0.5B y
`--temp 0` se quedan repitiendo la pregunta.

En un teléfono, dentro de **Octuma App** (la app Android que corre estos
modelos; en pruebas, todavía sin publicar), con un Xiaomi 14T Pro y Android 16:

| Modelo | Generación |
|---|---|
| Qwen2.5-0.5B INT4 | 28.4 tok/s |
| Qwen2.5-1.5B INT4 | 9.9 tok/s |

Mediana de cinco respuestas cada uno, build de publicación, dos hilos. Es un
teléfono de gama alta: en uno modesto será más lento. Condiciones completas en
[`runs/NOTA_telefono_2026-10-07.md`](https://github.com/1mano1/octuma/blob/main/runs/NOTA_telefono_2026-10-07.md).

## Resultados del barrido

33 corridas sobre Qwen2.5 (0.5B, 1.5B, 3B y 7B), en `runs/*.json`. El orden de
los métodos es **idéntico en los cuatro modelos**, que es la mejor señal de que
la implementación hace lo que dice:

```
GPTQ+AWQ  >  GPTQ  >  AWQ-RTN  >  RTN
```

Pérdida del mejor método frente a FP16: 5.2% (0.5B), 2.3% (1.5B), 2.4% (3B) y
1.8% (7B con GPTQ solo: sus corridas con AWQ no cabían en memoria). INT8 es
prácticamente gratis (+0.03%) pero solo comprime 1.91x; INT4 comprime 3.66x,
contando solo los pesos cuantizados: el modelo entero baja 59% en el 3B, porque
los embeddings se quedan en FP16.

> El barrido se corrió con **grupos de 64**, que era el valor por defecto
> entonces. Hoy la terminal usa 32, que es lo que exige el exportador a GGUF y
> lo que llevan dentro los tres modelos publicados. Grupos más chicos guardan
> más escalas y por eso comprimen menos: 3.66x con 64 contra 3.37x con 32. Las
> tablas de llama.cpp de más arriba sí son con 32; lo que no está medido es
> cuánta perplejidad cambia entre un tamaño de grupo y el otro.

### Cuánta calibración hace falta

GPTQ estima `H = 2·XXᵀ` por capa. Con menos tokens que dimensiones tenga la
capa, esa matriz es singular y la compensación de error se vuelve ruido: el
resultado sale **peor** que no usar GPTQ. Medido en el 0.5B:

| Tokens de calibración | GPTQ | GPTQ + AWQ |
|---|---|---|
| 512 (menos que las 896 dimensiones) | peor que RTN | mucho peor que RTN |
| 8192 | +7.5% | +4.2% |

Por eso `quantize_model` avisa cuando la calibración no alcanza. Regla práctica:
al menos 10 veces la dimensión de la capa más ancha.

### Un resultado negativo que vale la pena

La búsqueda de escala (`--search-scale`), medida sobre RTN, ayuda en 0.5B y 3B
y **estorba** en 1.5B y 7B. En el 7B reduce el error de reconstrucción de los
pesos (0.09503 → 0.09152) pero **empeora** la perplejidad (7.4386 → 7.5189).

Minimizar el error de los pesos no equivale a preservar la calidad del modelo.
Detalle en [`runs/NOTA_rtn_search.md`](https://github.com/1mano1/octuma/blob/main/runs/NOTA_rtn_search.md).

## Estado

- [x] Cuantización por grupos INT2/INT3/INT4/INT8, simétrica y asimétrica
- [x] Empaquetado real de bits (INT4 = medio byte)
- [x] GPTQ con compensación de error y Hessiana amortiguada
- [x] AWQ con búsqueda del exponente por grupo de capas
- [x] Cuantización secuencial bloque por bloque (el error se propaga como en la vida real)
- [x] Capa `QuantLinear` que reconstruye los pesos al vuelo
- [x] Precisión mixta por capa (`bits_overrides`) y plan automático (`octuma analyze`)
- [x] Guardar y cargar `.tq`
- [x] Perplejidad, memoria y velocidad
- [x] Exportar a GGUF Q4_1 (llama.cpp y Android), validado y medido contra sus formatos
- [ ] Octuma App para Android publicada (hoy en pruebas)
- [ ] Kernel rápido para INT4 (hoy se reconstruye al vuelo)
- [ ] Cuantización del caché KV

## Desarrollo

```bash
git clone https://github.com/1mano1/octuma.git
cd octuma
pip install -e ".[hf,gguf,dev]"
pytest -q          # 131 pruebas, menos de un minuto en CPU
ruff check .
```

Las pruebas usan modelos Llama diminutos creados al vuelo, sin descargar nada.
Corren la terminal de punta a punta y comprueban que los comandos, las opciones
y los imports que enseña este README existen de verdad. La CI las corre en
Linux y Windows con Python 3.10, 3.11 y 3.12.

Las dos reglas que ya costaron caro están en
[`CONTRIBUTING.md`](https://github.com/1mano1/octuma/blob/main/CONTRIBUTING.md): un valor por defecto nunca tapa un fallo,
y los valores por defecto son lo que se midió como mejor.

## Licencia

El código de Octuma es **MIT** (ver [LICENSE](https://github.com/1mano1/octuma/blob/main/LICENSE)).

**Los modelos son otra cosa.** Un modelo cuantizado es una obra derivada: se
queda con la licencia del original, y la de Octuma no la afloja. Por eso cada
repo publicado lleva dentro la licencia de su modelo base:

| Base | Licencia | Uso comercial |
|---|---|---|
| Qwen2.5-0.5B / 1.5B / 7B-Instruct | Apache 2.0 | sí |
| Qwen2.5-3B-Instruct | [Qwen Research](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) | **no** |

Si cuantizas otro modelo con Octuma, revisa su licencia antes de publicarlo:
varias familias populares (Llama, Gemma) traen condiciones propias que viajan
con los pesos derivados.

El proyecto se llamaba TinyQ hasta el 22 de septiembre de 2026. Los enlaces
viejos de GitHub y de Hugging Face redirigen solos.

Built with Qwen.
