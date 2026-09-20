# Contribuir a TinyQ

Gracias por pasarte. Esta guía es corta a propósito: si algo aquí no funciona
o no se entiende, eso ya es un bug y vale la pena abrir un issue.

## Poner el proyecto a andar

```bash
git clone https://github.com/1mano1/TinyQ
cd TinyQ
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -e ".[hf,gguf,dev]"
pytest -q
```

Los tests **no descargan ningún modelo**: usan redes Llama diminutas creadas al
vuelo, así que corren en menos de un minuto en cualquier laptop y sin GPU. Si
algún test tarda más que eso o pide red, es un error del test.

En Windows, si `python` te manda a la Microsoft Store, usa `py` en su lugar.

## El ciclo normal de trabajo

```bash
ruff check src tests scripts      # estilo y errores comunes
ruff format src tests scripts     # formato
pytest -q                         # pruebas
```

Los tres tienen que pasar antes de abrir un PR. La CI corre exactamente eso en
Linux y Windows, con Python 3.10 a 3.12.

## Cómo está organizado

| Carpeta | Qué hay |
|---|---|
| `src/tinyq/quant/` | El núcleo: GPTQ, AWQ, búsqueda de escala, empaquetado de bits. |
| `src/tinyq/export/` | Escribir el modelo en disco: formato `.tq` propio y GGUF. |
| `src/tinyq/` | Calibración, evaluación, sensibilidad de capas y la CLI. |
| `tests/` | Pruebas. Sin red, sin GPU, sin modelos reales. |
| `scripts/` | Andamiaje de experimentos e infraestructura. **No es parte del paquete**; ver `scripts/README.md`. |
| `experiments/` | Las matrices de experimentos en YAML. |
| `runs/` | Resultados en JSON. Se versionan: son la evidencia de la tabla del README. |

## Dos reglas que nos han costado caro

**1. Un valor por defecto nunca debe tapar un fallo.**

Este bug estuvo publicado:

```python
writer.add_rope_freq_base(getattr(cfg, "rope_theta", 10000.0))
```

`transformers 5` movió `rope_theta` dentro de `rope_parameters`. El `getattr`
se lo tragó en silencio y escribió 10000 donde el modelo usa 1000000: la
perplejidad se duplicó y nadie se enteró, porque el modelo seguía respondiendo
frases cortas con normalidad.

Si un valor es indispensable y no aparece, **lanza un error**. Es preferible
que falle ruidosamente a que produzca un modelo degradado.

**2. Lo que está medido como mejor es lo que debe pasar por defecto.**

Los valores por defecto de `tinyq quantize` son la configuración que gana en
el barrido (`runs/COMPARATIVA.md`), no una más conservadora. Si cambias un
valor por defecto, tienes que poder señalar la corrida que lo respalda.
`tests/test_cli.py` los fija justamente para que no se separen de lo medido.

## Al tocar la cuantización

Cualquier cambio en `src/tinyq/quant/` puede degradar la calidad sin romper un
solo test. Antes de abrir el PR, mide con un modelo pequeño:

```bash
tinyq quantize Qwen/Qwen2.5-0.5B-Instruct
tinyq compare qwen2.5-0.5b-instruct-int4 --windows 20
```

Pon el antes y el después en la descripción del PR. Un cambio que mejora la
perplejidad es bienvenido; uno que la empeora necesita justificar por qué
(velocidad, memoria, simplicidad).

## Al tocar el exportador a GGUF

Los tests no detectan un GGUF corrupto: los pesos pueden estar perfectos y el
archivo salir roto por los metadatos. Verifícalo siempre:

```bash
tinyq export <carpeta-tq> --out modelo.gguf
python scripts/verify_gguf.py <carpeta-tq> modelo.gguf
```

Ese script compara tensor por tensor **y** revisa los metadatos, y devuelve un
código de error si algo no cuadra.

## Estilo

- Código y comentarios en español, como el resto del proyecto.
- Los comentarios explican **por qué**, no qué hace la línea. Si un comentario
  se puede deducir leyendo el código, sobra.
- Nombres descriptivos en español para lo nuestro; los términos técnicos
  establecidos (GPTQ, AWQ, perplexity, scale) se quedan como son.
