# Velocidad en un teléfono: Xiaomi 14T Pro

Fecha: 2026-10-07. Los `.gguf` publicados en Hugging Face (re-exportados con
Octuma 0.1.5), corriendo dentro de **Octuma App** con llama.cpp b11065.

## Condiciones

| | |
|---|---|
| Teléfono | Xiaomi 14T Pro (`2407FPN8EG`), Android 16, 12 GB de RAM |
| App | Octuma App, build `release` (no la de depuración) |
| Hilos | 2. Android le dejaba 3 núcleos a la app y la medición de hilos de la app eligió 2 |
| Contexto | 2048 tokens |
| Batería | cargando, entre 37.9 y 40.3 °C durante las pruebas |

Cada respuesta va en una conversación nueva. La velocidad es la que muestra la
app debajo de cada respuesta: **solo la generación**, sin contar el tiempo de
leer la pregunta.

## Resultados

| Modelo | Velocidad por respuesta (tok/s) | Mediana |
|---|---|---|
| Qwen2.5-0.5B INT4 | 28.4 · 27.7 · 28.1 · 28.7 · 28.7 | **28.4 tok/s** |
| Qwen2.5-1.5B INT4 | 7.2 · 11.1 · 10.3 · 9.9 · 9.8 | **9.9 tok/s** |

Respuestas de 60 a 512 tokens. Las cinco preguntas, iguales para los dos
modelos: fotosíntesis, una receta de enchiladas, Python contra JavaScript, una
historia sobre un faro e inflación.

## Lo que no dice

- Es **un teléfono**, de gama alta. En uno modesto será más lento.
- Android cambia los núcleos que le deja a la app mientras corre. Con más
  núcleos disponibles la app midió antes hasta 39 tok/s con el 0.5B (siete
  hilos, build de depuración); con menos, menos.
- El teléfono estaba cargando, que lo calienta. No se midió con el tiempo qué
  pasa en sesiones largas.
