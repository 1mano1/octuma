# TinyQ

Cuantizacion INT4/INT8 de modelos de lenguaje, pensada para que corran en
laptops modestas, servidores chicos y telefonos Android.

Un modelo de 8B en FP16 pide ~16 GB de memoria y no cabe en casi ningun equipo
normal. TinyQ lo baja a ~4.3 GB conservando la mayor parte de la calidad, con
cuatro pasos: **calibrar, cuantizar, evaluar y exportar**.

```bash
pip install -e ".[hf]"

tinyq quantize Qwen/Qwen2.5-0.5B-Instruct --out out/qwen-int4 --bits 4 --group 64
tinyq evaluate out/qwen-int4 --windows 20
tinyq info out/qwen-int4
```

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

Por eso, para exportar a GGUF hay que cuantizar con `--group 32`:

```bash
tinyq quantize <modelo> --out out/m --bits 4 --group 32
tinyq export out/m --out modelo-int4.gguf
```

El `.gguf` resultante lo lee llama.cpp, y de ahi corre en Android.

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
- [ ] Validar el GGUF corriendo en llama.cpp
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
