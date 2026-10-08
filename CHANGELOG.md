# Cambios

## 0.1.5 — 2026-10-07

Tres fallos que encontraron las pruebas nuevas de la terminal, que hasta ahora
no se ejecutaba en ninguna prueba.

- **`octuma quantize --plan` aplicaba el plan a todos los bloques.** El plan de
  `octuma analyze` nombra capas concretas (`blocks.1.mlp.down_proj`), pero al
  leerlo se le quitaba el numero de bloque y la capa subia a 8 bits *en todos
  los bloques*. Un plan de cuatro capas en un modelo de dos bloques dejaba
  ocho a 8 bits; en un modelo de 24 bloques el promedio de bits se pasaba
  mucho del pedido, sin ningun aviso. Ahora `bits_overrides` distingue una
  capa concreta (`"3.mlp.down_proj"`, solo esa) de un tipo de capa
  (`"mlp.down_proj"`, todos los bloques), y `"1.mlp"` ya no casa con el bloque
  11. **Quien haya cuantizado con `--plan` en una version anterior tiene un
  modelo con mas capas a 8 bits de las que pidio**: mas grande, no peor.
- **Los modelos publicados no se podian abrir.** Los tres de Hugging Face se
  subieron cuando el proyecto se llamaba TinyQ y traen `tinyq.json`; tras el
  renombrado, `info`, `compare`, `try` y `export` exigian `octuma.json` y
  contestaban que la carpeta no era de Octuma. Ahora se aceptan los dos
  nombres. Se sigue escribiendo `octuma.json`.
- **En Windows, `octuma quantize <carpeta local>` sin `--out` guardaba en otro
  sitio.** El nombre de la salida se sacaba partiendo por `/`, y una ruta con
  barra invertida no se partia: el resultado quedaba junto al modelo original,
  con la ruta entera en minusculas, en vez de en la carpeta actual.
- **El GGUF lleva la plantilla de chat.** `tokenizer.chat_template` no se
  escribia: el archivo no decia donde empieza y acaba cada turno, y quien lo
  abria tenia que adivinarlo. Los tres modelos publicados salieron asi.
  `verify_gguf.py` lo comprueba ahora, igual que la etiqueta `F16` en un
  archivo Q4_1. Hace falta re-exportar para corregirlo; no hay que recuantizar.
- **La API de Python hace lo mismo que la terminal.** `QuantConfig()` venia
  con grupos de 64 y AWQ apagado, los valores de antes del barrido, mientras
  `octuma quantize` usaba 32 y AWQ: quien cuantizaba desde Python sacaba un
  modelo peor que el de la terminal y que ademas no se podia exportar a GGUF.
  Ahora todo el paquete usa grupos de 32 por defecto (`QuantConfig`,
  `GPTQConfig`, `QuantLinear`, `quantize_tensor`) y `QuantConfig` trae AWQ
  encendido. **Cambia el resultado de quien dependiera de los valores
  viejos**: para conservarlos hay que pedir `group_size=64, awq=False`.
- **`octuma analyze` mide con grupos de 32**, igual que `quantize`. El plan se
  calculaba con 64 y se aplicaba cuantizando con 32.
- **Pruebas: de 74 a 129.** La terminal se ejecuta de punta a punta con un
  modelo diminuto (quantize, info, evaluate, compare, try, analyze y export),
  la calibracion tiene las suyas, y otras comprueban que los comandos, las
  opciones, los imports y los enlaces del README existen de verdad. La
  cobertura pasa de 74% a 92%.
- **README rehecho**, en español y en ingles: como funciona cada paso, tabla
  de comandos, estructura del proyecto, el formato `.tq`, cabecera animada y
  `llms.txt`. Corrige cuatro datos: la tabla contra llama.cpp ahora da el
  tamaño de cada archivo (el de Octuma es entre 24% y 33% mas grande que
  Q4_K_M en los tres modelos, y antes solo se decia del 0.5B); un grupo de 32
  cuesta 4.75 bits por peso, no "~4.5"; `QuantConfig()` a secas no equivale a
  la terminal; y la orden de "Para borrar todo" tenia dos comandos pegados en
  una linea.

## 0.1.4 — 2026-09-29

- **La terminal habla en ingles.** Todo lo que Octuma muestra al usuario
  (progreso, tablas, avisos, errores y `--help`) paso del español al ingles,
  para que lo entienda cualquiera que lo instale de PyPI. Las diez preguntas de
  `octuma try --side-by-side` tambien. El codigo y sus comentarios siguen en
  español.
- **El pulpo de Octuma sale en la terminal** la primera vez que se usa `octuma`
  despues de instalar o actualizar ("updated: 0.1.3 -> 0.1.4", en ingles), y siempre
  con `octuma --version`. No puede salir durante el `pip install`: pip no
  ejecuta codigo del paquete al instalar un wheel. Solo se dibuja en una
  terminal de verdad, asi que la salida de scripts y CI no cambia, y
  `OCTUMA_NO_LOGO=1` lo apaga. La ultima version vista se anota en
  `%APPDATA%/octuma` (Windows) o `~/.config/octuma`.
- **El GGUF se anunciaba como F16 siendo Q4_1.** `general.file_type` estaba
  fijo en `MOSTLY_F16`, y es lo que muestran llama.cpp (`ftype : F16`) y el
  visor de Hugging Face. No afectaba la calidad, porque cada tensor lleva su
  propio tipo, pero los tres modelos publicados dicen F16. Se corrige al
  re-exportarlos; no hace falta recuantizar.

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
