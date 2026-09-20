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
