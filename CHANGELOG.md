# Cambios

## 0.1.3 — 2026-09-23

- **`octuma quantize` ya no se queda sin memoria cuantizando con AWQ.** El hook
  que recoge activaciones guardaba `awq_samples` filas de *cada* lote de
  calibracion y el `torch.cat` posterior se quedaba solo con las primeras
  `awq_samples`: con los 128 lotes que usa el comando por defecto, eso pedia
  2.55 GB en una sola reserva para el `down_proj` de un Qwen2.5-0.5B y moria
  con `DefaultCPUAllocator: not enough memory` en una maquina de 30 GB. Ahora
  el recorte ocurre al guardar. **El resultado es identico** —los 127 lotes de
  mas se tiraban enteros—, pero el pico baja de gigabytes a decenas de megas.
  Esto tambien desbloquea las corridas AWQ del 7B, que se daban por perdidas
  por falta de RAM.
- **El aviso de memoria media la memoria equivocada.** Con CUDA presente solo
  miraba la VRAM, y ademas imprimia `total_memory` con la palabra "libres":
  anunciaba "la GPU tiene 8.6 GB libres" en una tarjeta de 8.6 GB totales,
  justo antes de morir por falta de RAM. Ahora informa de las dos memorias,
  usa la libre de verdad y avisa aparte si la RAM se queda corta.
- **Pasar una carpeta que no existe se explica.** `info` y `compare` soltaban
  un `FileNotFoundError` crudo, y `try` y `export` algo peor: transformers
  tomaba el nombre por un repo de Hugging Face y devolvia un 401 hablando de
  tokens de autenticacion. Ahora los cuatro dicen que la carpeta no existe y
  que revise si `quantize` llego a terminar.
- **README:** se ofrecia `pip install octuma` como "solo el motor". Con esa
  instalacion no se puede cuantizar: sin `transformers` solo funcionan
  `--version` e `info`. Documentado tambien que el wheel de PyTorch de PyPI es
  solo CPU en Windows y que el de CUDA hay que pedirlo a su propio indice.

## 0.1.2 — 2026-09-22

- **`octuma --version` funciona.** Existia el subcomando `octuma version`, pero
  la bandera —que es lo que se teclea sin pensar— contestaba
  `No such option: --version`. Salio al probar el paquete recien bajado de
  PyPI en una maquina limpia, no al leer el codigo. `octuma version` se queda
  como estaba, para no romper a quien ya lo usara.

## 0.1.1 — 2026-09-22

- La version del paquete vive ahora en un solo sitio, `octuma/__init__.py`.
  Estaba escrita tambien en `pyproject.toml` y nada obligaba a que coincidieran:
  se podia publicar un paquete que dijera una version y un `octuma --version`
  que dijera otra.
- El README —que es lo que PyPI muestra como descripcion— ofrece la instalacion
  normal, `pip install "octuma[hf,gguf]"`. En la 0.1.0 solo aparecia la
  instalacion desde git, porque el paquete todavia no existia en PyPI cuando se
  construyo ese artefacto. La instalacion desde git sigue documentada, para la
  version en desarrollo.
- Enlace `Homepage` en la ficha de PyPI.

No hay cambios de comportamiento: el motor de cuantizacion, la CLI y los
formatos `.tq` y GGUF son identicos a la 0.1.0.

## 0.1.0 — 2026-09-22

Primera version publicada. El proyecto se llamaba **TinyQ** y se renombro a
**Octuma** para compartir nombre con la app Android que corre estos modelos.

- Cuantizacion INT4 e INT8 con RTN, AWQ y GPTQ, y la combinacion GPTQ+AWQ que
  gana en las cuatro tallas medidas de Qwen2.5.
- CLI de tres comandos: `octuma quantize`, `octuma compare`, `octuma try`.
- Export a `.tq` (PyTorch) y a GGUF (llama.cpp y Android).
