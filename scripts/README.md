# scripts/

Andamiaje. **Nada de aquí es parte del paquete `tinyq`** ni se instala con
`pip`: son herramientas para correr los experimentos y sostener la
infraestructura. Si buscas la herramienta como usuario, lo que quieres es la
CLI (`tinyq quantize`, `tinyq compare`, `tinyq try`).

## Experimentos

| Script | Para qué |
|---|---|
| `run_sweep.py` | Corre la matriz de `experiments/*.yaml` y escribe un JSON por corrida en `runs/`. Salta lo que ya existe, así que se puede interrumpir y retomar. |
| `compare_baselines.py` | Cuantiza el mismo modelo con otras herramientas (bitsandbytes) y lo mide con **nuestro** evaluador, que es la única forma de que los números entren en la misma tabla. |
| `sanity_chat.py` | Las mismas preguntas al original y al cuantizado, lado a lado. Hoy vive también como `tinyq try --side-by-side`. |
| `export_artifacts.py` | Genera los modelos finales (`.tq` y `.gguf`) que se publican. |
| `benchmark.py` | Velocidad de generación. |
| `make_wikitext_txt.py` | Escribe wikitext-2 test como `.txt` para `llama-perplexity`, armado igual que el evaluador de Python. Sin él, el número de llama.cpp y el de TinyQ no miden el mismo texto. |
| `bench_gguf.py` | La cadena completa por modelo: baja el original, lo convierte a F16, lo cuantiza con llama.cpp (Q4_K_M, Q4_0) y mide las cuatro variantes con `llama-perplexity`. Escribe `runs/gguf__<slug>.json`. |
| `tabla_gguf.py` | Genera `runs/COMPARATIVA_GGUF.md` desde esos JSON. Todo el documento sale de los datos, incluidas las frases que dicen quién gana. |
| `grafica_gguf.py` | Una gráfica por modelo más la de conjunto, en claro y oscuro, para el README. |

## Diagnóstico

| Script | Para qué |
|---|---|
| `verify_gguf.py` | Compara un GGUF contra el `.tq` del que salió, tensor por tensor, **y revisa los metadatos**. Devuelve código 1 si algo no cuadra. Úsalo siempre antes de publicar un GGUF. |
| `diag_awq.py`, `diag_awq_gptq.py` | Inspeccionan el escalado AWQ capa por capa cuando un resultado no cuadra. |

## Infraestructura (RunPod)

Solo sirven para correr barridos largos en una GPU alquilada. Si trabajas en
local, ignóralos por completo.

| Script | Para qué |
|---|---|
| `runpod.py`, `runpod_bootstrap.sh` | Crear el pod y dejarlo listo. |
| `night_run.sh` | Encadena barrido, ablaciones, artefactos y publicación sin supervisión. Cada paso revisa su código de salida. |
| `watchdog.sh` | Apaga el pod al terminar. Reintenta hasta confirmarlo y **no apaga nada** si los modelos no están verificados en Hugging Face. |
| `upload_hf.py` | Sube un modelo a Hugging Face. Privado salvo que se diga lo contrario. |

### Dos cosas que costaron caro

**Un `git pull` a mitad de un barrido no cambia nada.** Python carga los
módulos al arrancar; el proceso en curso sigue con el código viejo y los
resultados no lo delatan. Por eso cada JSON guarda `code.commit` y
`code.started_at`. Si cambias el código, **reinicia el proceso**.

**El pod tiene 50 GB de RAM**, no los 503 que reporta `free` (esos son del
host). Cuando un proceso desaparece sin traceback, mira
`/sys/fs/cgroup/memory.events`. Y exporta siempre `HF_HUB_DISABLE_XET=1`:
el backend Xet de Hugging Face falla al reconstruir archivos grandes con un
error que parece de red y no lo es.
