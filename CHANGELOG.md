# Cambios

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
