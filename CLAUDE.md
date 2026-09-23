# CLAUDE.md — Octuma

Notas para retomar el proyecto sin tener que reconstruir el contexto.
Ultima actualizacion: 2026-09-22.

## Que es

Libreria de Python que cuantiza modelos de lenguaje a 4 y 8 bits para que
corran en equipos modestos. Cuatro pasos: calibrar, cuantizar, evaluar,
exportar (`.tq` para PyTorch y GGUF para llama.cpp y Android).

Proyecto hermano: la **app Android Octuma** (`C:/ia-local-android`), que corre
estos modelos en el telefono. La libreria comprime, la app ejecuta.

**Los dos se llaman igual a proposito** (decidido el 2026-09-22). Antes la
libreria era "TinyQ" y la app "Lumen", los dos nombres provisionales y ocupados
por terceros. Un solo nombre para las dos mitades le ahorra al usuario tener
que aprender que una cosa produce lo que la otra consume.

## Estado: que esta medido y que no

**Barrido principal: 33 corridas** en `runs/*.json`, familia Qwen2.5
(0.5B, 1.5B, 3B, 7B), evaluadas con perplejidad en wikitext-2, 20 ventanas
de 2048 tokens sin solape.

El orden de metodos es **identico en los cuatro modelos**, que es la mejor
senal de que la implementacion es correcta:

    GPTQ+AWQ  >  GPTQ  >  AWQ-RTN  >  RTN

Degradacion del mejor metodo frente a FP16: 5.2% (0.5B), 2.3% (1.5B),
2.4% (3B), 1.8% (7B con GPTQ solo). **Entre mas grande el modelo, menos duele
cuantizar**, que es el resultado vendible del proyecto.

INT8 es practicamente gratis (+0.03% maximo) pero solo comprime x1.91.
INT4 comprime x3.66.

### Comparacion contra terceros (lo que faltaba)

Todo lo anterior era Octuma contra si misma. En Qwen 3B, misma prueba:

| Herramienta | Perplejidad | Memoria | Perdida vs FP16 |
|---|---|---|---|
| FP16 | 8.347 | 6.79 GB | — |
| **Octuma GPTQ+AWQ** | **8.549** | 2.76 GB | **+2.4%** |
| Octuma GPTQ | 8.578 | 2.76 GB | +2.8% |
| bitsandbytes NF4 | 8.906 | 2.63 GB | +6.7% |
| bitsandbytes FP4 | 13.343 | 2.63 GB | +59.9% |

Octuma hace **menos de la mitad de dano** que bitsandbytes NF4, que es el
cuantizador por defecto de Hugging Face y el de QLoRA. Script:
`scripts/compare_baselines.py`.

**Sigue pendiente**: comparar contra otra implementacion de GPTQ. AutoGPTQ y
su sucesor GPTQModel **no instalan** (su `pyproject.toml` es invalido para
setuptools moderno, y no se arregla desde fuera). Es el rival tecnico mas
directo y hoy no hay numero contra el.

## RESUELTO: el export a GGUF (2026-09-20)

Eran **tres bugs de metadatos, ninguno de cuantizacion**. Los pesos siempre
estuvieron bien: dequantizados desde el `.tq` y desde el GGUF coinciden con
norma relativa 0.0004, que es solo redondeo de float16.

En Qwen2.5-0.5B, 5 ventanas de 512 medidas con `llama-perplexity`:

| Version | Perplejidad |
|---|---|
| GGUF original | 44.36 |
| solo con rope corregido | 31.51 |
| **con los tres arreglos** | **15.31** |

1. **`rope_theta` caia a un default.** transformers 5 lo movio dentro de
   `cfg.rope_parameters`, y `getattr(cfg, "rope_theta", 10000.0)` escribia
   10000 donde Qwen2.5 usa 1000000. Ahora `_rope_theta()` lo busca en ambos
   sitios y **lanza error si no aparece**, en vez de inventarse un valor.
2. **El pre-tokenizador decia `default`** en lugar de `qwen2`: el texto se
   partia distinto a como el modelo aprendio. Ahora sale de `_PRE_POR_ARCH`.
3. **El `eos_token` no se escribia**: `<|im_end|>` vive en `added_tokens`, no
   en `model.vocab`, y se buscaba solo en vocab.

Tres tests de regresion en `tests/test_gguf.py`. **55 tests pasan.**

El patron comun, otra vez: **un default que tapa el fallo en silencio**. Lo
mismo que hacia `night_run.sh` al anunciar "todo listo" con el barrido muerto.

### Verificado sobre los tres modelos publicados

Los tres `.gguf` de Hugging Face, cada uno contra su re-exportado, con el
mismo protocolo: 20 ventanas de 2048 (detalle en `runs/NOTA_gguf_reexport.md`).

| Modelo | el publicado | re-exportado | Mejora |
|---|---|---|---|
| 0.5B | 27.484 | **12.710** | 2.16x |
| 1.5B | 24.462 | **8.486** | 2.88x |
| 3B | 15.453 | **7.512** | 2.06x |

Los archivos publicados estaban degradados de verdad; no era cosa del modelo de
prueba. **El bug se comia el beneficio de crecer**: entre el 0.5B y el 1.5B
publicados habia un 11% de diferencia, y entre los arreglados hay un 33%.

Los tres re-exportados estan en `out/*-int4-fix.gguf`.

### Pendiente de esto

- ~~Re-exportar los `.gguf` del 1.5B y el 3B~~ **HECHO**: los tres estan
  re-exportados y verificados (`verify_gguf.py` da 0 en los tres).
- ~~Rehacer la comparativa del 3B contra Q4_K_M~~ **HECHO**: 7.512 aqui contra
  7.824 de Q4_K_M, reproduciendo el 7.492 de la otra maquina.
- ~~Resubir los tres a Hugging Face~~ **HECHO (2026-09-20)**: los tres repos
  tienen el `.gguf` arreglado, comprobado por tamaño contra el archivo local.
  La ficha de cada uno la genera `scripts/publicar_gguf.py` desde
  `runs/gguf__<slug>.json`, asi que la tabla no se puede desfasar a mano.

## La comparativa completa contra llama.cpp (2026-09-20)

Los tres modelos medidos **dentro de llama.cpp** contra sus propios formatos,
mismo corpus y mismas ventanas (`runs/COMPARATIVA_GGUF.md`, generado por
`scripts/tabla_gguf.py`; graficas con `scripts/grafica_gguf.py`):

| Modelo | Octuma INT4 | Q4_K_M | Q4_0 |
|---|---|---|---|
| 0.5B | +3.73% | **+2.63%** | +13.17% |
| 1.5B | **+2.11%** | +4.74% | +8.35% |
| 3B | **+2.43%** | +6.21% | +11.02% |

**Octuma pierde en el 0.5B**, y hay que decirlo: Q4_K_M hace menos daño y ocupa
menos (0.40 GB contra 0.52). El argumento honesto no es "siempre gana", es que
**Q4_K_M se degrada al crecer el modelo (2.63 → 4.74 → 6.21) y Octuma no**
(3.73 → 2.11 → 2.43). El cruce esta entre 0.5B y 1.5B, y el lado bueno del
cruce es justo el caso de uso que importa: el modelo mas grande que quepa.

`scripts/bench_gguf.py` hace la cadena completa por modelo (baja el original,
convierte a F16, cuantiza con llama.cpp y mide las cuatro variantes). Reproduce
las cuatro filas documentadas del 3B dentro del 0.5%.

### El proyecto se llama `octuma` (renombrado 2026-09-22)

Se llamaba **TinyQ**. El nombre cambio para que la libreria y la app Android
sean la misma marca: quien ve "Octuma" en el telefono encuentra "Octuma" en
PyPI, y no tiene que aprender que una cosa produce lo que la otra consume.

De paso desaparecio una costura fea. `tinyq` en PyPI **es otro proyecto sin
relacion** (un gestor de colas de trabajo, v0.3.0, de mozillazg), asi que habia
que distribuir como `tiny-q` e importar como `tinyq`: dos nombres para lo
mismo. `octuma` estaba libre el 2026-09-22 (PyPI responde 404), asi que ahora
**lo que se instala, lo que se importa y el comando se llaman igual**.

Esto **no reserva el nombre**: nadie lo tiene hasta que se suba a PyPI. Si el
proyecto se va a abrir, conviene registrarlo antes de anunciarlo.

**Lo que NO se renombro, a proposito:**

- Los tres repos de Hugging Face (`Imanol11/qwen*-int4-TinyQ`) y el repo de
  GitHub (`1mano1/TinyQ`). Son direcciones reales que hoy funcionan; cambiarlas
  es una decision del autor, no del renombrado.
- El formato **`.tq`** (`model.tq.safetensors`). Vive dentro de los tres
  modelos ya publicados: renombrarlo los rompe a cambio de nada que el usuario
  vea. El modulo sigue siendo `src/octuma/export/tq.py`.

Tambien: **`huggingface-cli` ya no existe**, el comando es `hf`.

## Licencias: el 3B no es Apache 2.0 (2026-09-20)

**`Qwen2.5-3B-Instruct` esta bajo la Qwen RESEARCH LICENSE**, que define
"Non-Commercial" como "research or evaluation purposes only". El resto de la
familia que usamos (0.5B, 1.5B, 7B) si es Apache 2.0. Los tres repos de
Hugging Face declaraban `apache-2.0`, o sea que el del 3B **tergiversaba la
licencia de Alibaba**. Corregido.

Un modelo cuantizado es obra derivada: se queda con la licencia del original.
Que Octuma sea MIT no afloja nada. Lo que pide la Qwen Research y ya se cumple:

- **§3a** copia del acuerdo para quien reciba los pesos -> `LICENSE` en el repo
- **§3b** avisar que los archivos estan modificados -> lo dice la ficha
- **§3c** aviso de atribucion en un `NOTICE` -> subido, texto literal
- **§4b** "Built with Qwen" visible -> en la ficha y en el README

`publicar_gguf.py` lee la licencia del Hub al publicar y **revienta si
encuentra una que no sabe describir**, en vez de asumir Apache. Tambien baja el
`LICENSE` del original y lo sube con los pesos (Apache 2.0 §4(a) pide lo mismo).

**Esto le pega a la app**: si Octuma para Android llega a ser comercial, no puede distribuir
el 3B. El 1.5B es el modelo mas grande que puede usar sin pedirle permiso a
Alibaba. Conviene decidirlo antes de construir encima.

wikitext-2 (CC BY-SA 3.0) se usa para calibrar y evaluar, y **no se
redistribuye**: `make_wikitext_txt.py` lo genera en `out/`, que esta ignorado.

## La CLI rediseñada (2026-09-20)

Tres comandos de entrada, segun `docs/PLAN_CLI.md`:

- `octuma quantize <modelo>` — sin `--out` (se deduce), sin elegir metodo
  (GPTQ+AWQ, grupos de 32, 128x2048: **la configuracion que gana el barrido**),
  con `--device auto` y `--dtype auto`. Antes venia con AWQ apagado y 64x512,
  asi que el comando obvio daba peores resultados que la tabla del README.
- `octuma compare <carpeta>` — cuantizado contra original en una sola tabla, y
  la linea que resume: "1.94x mas chico por +5.9% de perplejidad".
- `octuma try <carpeta>` — chat en la terminal, `-p` para una sola pregunta, y
  `--side-by-side` para las 10 preguntas contra el original.

Ademas, **avisa de la memoria antes de descargar nada**: "Este modelo pide
~8.6 GB y la RAM tiene 7.3 GB libres". Reventar a los diez minutos tras bajar
15 GB era la peor primera impresion posible.

`tests/test_cli.py` fija los defaults para que no vuelvan a divergir de lo
medido. **60 tests pasan.**

## Velocidad: el cuantizado es 3.4x mas lento

14.4 tok/s el original contra 4.2 el cuantizado (Qwen 3B, GPU). `QuantLinear`
desempaqueta los 4 bits a FP16 en cada multiplicacion con PyTorch normal, sin
kernel CUDA. El README **no puede prometer velocidad**: el argumento es
memoria y calidad. Siguiente trabajo natural: un kernel de 4 bits.

## Hallazgos que hay que respetar

### 1. La busqueda de escala no funciona de forma consistente

`rtn-int4-search` ayuda en 0.5B y 3B, y **estorba** en 1.5B y 7B. En el 7B
reduce el error de reconstruccion de los pesos (0.09503 -> 0.09152) pero
**empeora** la perplejidad (7.4386 -> 7.5189).

Eso contradice el supuesto en el que descansa la tecnica: minimizar el error
de los pesos no equivale a preservar la calidad del modelo. Es un resultado
negativo que vale la pena reportar, no un bug que esconder.

Detalle completo en `runs/NOTA_rtn_search.md`. Las corridas originales eran
invalidas (duplicados de `rtn-int4`) y estan guardadas en `runs/invalidos/`.

### 2. Un `git pull` a mitad del barrido no cambia nada

Python carga los modulos al arrancar. El barrido de la noche del 19 siguio
usando codigo viejo horas despues del pull, y **nada en los resultados lo
delataba**. Por eso cada JSON guarda ahora `code.commit`, `code.dirty` y
`code.started_at`, sellados al importar el modulo.

Si se cambia el codigo, hay que **reiniciar el proceso**, no solo hacer pull.

### 3. El 7B con AWQ no cabe en 50 GB de RAM

El contenedor de RunPod tiene 50 GB (lo que reporta `free` son los 503 GB del
host, no sirve). Las corridas con AWQ del 7B mueren por OOM justo despues de
cargar los pesos, **sin traceback**: el log corta y ya. Si un proceso
desaparece en silencio, revisar `/sys/fs/cgroup/memory.events`.

Las 4 corridas del 7B que si funcionaron no usan AWQ. Faltan `awq-rtn-int4` y
`gptq-awq-int4`. Opciones: bajar `calib_samples` solo para el 7B (rompe la
comparabilidad) o liberar las activaciones capturadas por bloque (correcto,
pero hay que tocar el motor y reprobarlo en un modelo chico).

## Trampas de infraestructura ya resueltas

- **`HF_HUB_DISABLE_XET=1`** es obligatorio. El backend Xet de Hugging Face
  revienta al reconstruir shards grandes con un `CAS Client Error` que parece
  un fallo de red y no lo es.
- **`HF_HOME` debe apuntar a `/workspace`**. El disco de root son 30 GB y un
  7B en FP16 pide 15.
- **El vigilante fallaba en silencio.** `terminate()` hacia `exit 0` sin mirar
  el resultado; un 403 pasajero de RunPod dejo el pod encendido 6 horas. Ahora
  reintenta 8 veces con espera creciente y **verifica que el pod desaparecio**.
- **`night_run.sh` mentia.** Ningun paso miraba su codigo de salida, asi que
  anunciaba "todo listo" con el barrido muerto por OOM. Ahora cada paso pasa
  por `paso()`, distingue el 137 del OOM killer y termina con "CON N fallos".
- **El pod no se apaga sin verificar los modelos.** El centinela
  `/workspace/.modelos_a_salvo` solo se escribe tras consultar Hugging Face y
  comprobar que cada repo existe y trae archivos.

## Modelos publicados

Privados en Hugging Face, cada uno con carpeta `.tq`, `.gguf` y el `LICENSE`
del modelo base:

- `Imanol11/qwen0.5b-int4-TinyQ` (1.02 GB) — Apache 2.0
- `Imanol11/qwen1.5b-int4-TinyQ` (2.58 GB) — Apache 2.0
- `Imanol11/qwen3b-int4-TinyQ` (4.68 GB) — **Qwen Research, no comercial**
  (lleva ademas `NOTICE`)

No hay 7B publicado. Los publica el autor cuando decida, no antes.

Cada repo trae su ficha generada por `scripts/publicar_gguf.py`: tabla de
calidad medida dentro de llama.cpp, como usarlo con `llama-cli` y con PyTorch,
y una nota de que el `.gguf` anterior al 2026-09-20 estaba degradado.

**Ojo con la visibilidad:** el repo de GitHub es publico y su README enlaza a
los tres, pero **los tres siguen privados**, asi que esos enlaces dan 404 para
cualquiera que no sea el autor. Hacerlos publicos es una decision suya; abrir
pesos no se deshace.

## Pendientes

1. Repetir `awq-rtn-int4` y `gptq-awq-int4` del 7B (ver hallazgo 3).
2. Comparar contra una implementacion real de GPTQ.
3. ~~Arreglar el export a GGUF~~ **HECHO** (ver arriba).
4. ~~Resubir los tres .gguf~~ **HECHO** (ver arriba). La app ya puede bajar de
   Hugging Face modelos que no estan degradados.
5. Rediseñar la CLI y la documentacion segun `docs/PLAN_CLI.md`.
6. ~~Decidir nombre de la app~~ **HECHO (2026-09-22)**: se llama **Octuma**,
   igual que esta libreria. Falta registrar `octuma` en PyPI si se abre.

## Entorno local (Windows)

Python **si** esta instalado, en
`C:/Users/brosm/AppData/Local/Programs/Python/Python311/python.exe` (3.11.0).
Lo que hay en el PATH es el stub de la Microsoft Store, que responde
"Python was not found": usar `py` o la ruta completa, nunca `python` a secas.

Con `pip install -e . --no-deps` los tests corren en local: **47 pasan, 1 se
salta**. No hace falta el pod para verificar cambios de codigo. El torch local
es la version CPU.

**En la PC de la RTX 4060** (2026-09-20) el Python es otro: ahi no existe el
3.11, hay **3.10.0 y 3.14.2**, y el que tiene todo instalado es el **3.10**
(`AppData/Local/Programs/Python/Python310/python.exe`), con
**torch 2.5.1+cu121 y CUDA disponible**. Con `pip install -e ".[hf,gguf,dev]"`
pasan los **60 tests**.

## Pendientes y donde correrlos

Hay una **RTX 4060 con 8 GB de VRAM** en otra PC del autor. Eso cambia que
conviene correr donde:

| Tarea | Donde | Por que |
|---|---|---|
| Medir GGUF con `llama-perplexity` | **RTX 4060** | En CPU cada medicion del 3B tarda ~1 h; con GPU son minutos. |
| Re-exportar y verificar los `.gguf` del 1.5B y el 3B | cualquiera | Solo pide RAM, no GPU. |
| Las 2 corridas AWQ del 7B | **ni ahi** | El 7B en FP16 son 15.2 GB: no cabe en 8 GB de VRAM. Necesita GPU alquilada o el arreglo de memoria (liberar activaciones por bloque). |
| Cuantizar hasta 1.5B | RTX 4060 | Comodo. |
| Cuantizar el 3B | RTX 4060, justo | FP16 son 6.79 GB de 8: cabe para evaluar, apretado para cuantizar con AWQ. |

### Comandos para la RTX 4060

```bash
# medir un GGUF con GPU (llama.cpp compilado con CUDA)
llama-perplexity -m modelo.gguf -f wikitext2.txt -c 2048 --chunks 20 -ngl 99

# re-exportar y verificar (no necesita GPU)
octuma export <carpeta-tq> --out modelo-int4.gguf
python scripts/verify_gguf.py <carpeta-tq> modelo-int4.gguf
```

Los binarios de llama.cpp para Windows se bajan ya compilados de
`github.com/ggml-org/llama.cpp/releases` (el zip `bin-win-cuda-12.4-x64`, o
`bin-win-cpu-x64` si no se quiere CUDA). No hace falta compilar nada.

Ya instalados en esta maquina en `C:/llamacpp/bin` (build b11065). Ojo con la
version de CUDA: el driver **591.86 llega hasta CUDA 13.1**, asi que el zip
`cuda-13.4` no arranca y hay que quedarse en el `cuda-12.4`. Tambien hace falta
el zip `cudart-` del mismo build, que trae las DLL del runtime.

El `wikitext2.txt` lo genera `scripts/make_wikitext_txt.py`, que lo arma igual
que el evaluador de Python. Sin el, el numero de llama.cpp no es comparable.

### Lo que falta medir

1. ~~Medir el GGUF del 3B arreglado~~ **HECHO**: da **7.492** con 20 ventanas,
   contra 7.824 de Q4_K_M y 8.163 de Q4_0 (su F16 es 7.330). **Octuma tambien
   le gana a llama.cpp**: +2.2% de dano contra +6.7% del formato mas usado
   para correr modelos en local. Es el argumento de por que la app Android usa
   estos modelos y no unos cualquiera.
2. ~~Re-exportar el `.gguf` del 1.5B~~ **HECHO**: da **8.486** contra 24.462
   del publicado. Los tres estan en `out/*-int4-fix.gguf`, verificados y
   **ya subidos** a Hugging Face.
3. **Tokens por segundo en un telefono real**, cuando la app corra. El
   portafolio tenia una columna "Pixel 7 (tok/s)" **inventada** que hubo que
   quitar: ese hueco se llena con mediciones reales del celular del autor, y es
   un dato que casi nadie publica.

## El portafolio de Figma (actualizado 2026-09-20)

Archivo `nc1HVKdJK4VZsoAyxsOyuz`, pagina **Portafolio V3 — Desktop**, frame
`octuma — Desktop` (95:5999).

**Todos los benchmarks que tenia eran inventados.** Hablaban de Llama-3 8B,
Mistral 7B y Phi-3 — modelos que nunca se midieron — y la nota al pie decia
"valores de ejemplo", pero a simple vista parecian mediciones reales.

Corregido: la tabla y el grafico de memoria ahora son Qwen2.5 con los numeros
de `runs/`; la columna "Pixel 7 (tok/s)" se cambio por el metodo; "GGUF · ONNX
· TFLite" paso a "GGUF · .tq" (ONNX y TFLite no existen en el codigo); la
version "v0.3.0" a 0.1.0; el chip "16 GB → 4.3 GB · 98.6% precision" a los
datos reales del 7B; "Star 1.2k" quitado; y el bloque de codigo animado (13
variantes) usaba `from octuma import Quantizer`, una **API que no existe**, y
ahora usa la real.

Si se vuelven a tocar esos numeros, salen de `runs/COMPARATIVA.md`.

## Estructura y calidad (2026-09-20)

- **`CONTRIBUTING.md`**: instalacion, ciclo `ruff` + `pytest`, mapa de
  carpetas y las dos reglas que ya costaron caro (un default nunca tapa un
  fallo; los defaults son lo medido como mejor).
- **`ruff` configurado y el repo en verde.** 46 avisos arreglados. Los ignores
  estan justificados en `pyproject.toml`: `typer.Option` en los defaults es el
  modo normal de typer (B008), y los hooks capturan la variable del bucle con
  `_name=name`, que B023 no reconoce.
- **CI ampliada**: Linux y **Windows**, Python 3.10/3.11/3.12, mas un job de
  estilo. Windows importa porque es donde aparecen los problemas de rutas y
  codificacion, y el proyecto apunta a equipos modestos.
- **`scripts/README.md`**: separa experimentos, diagnostico e infraestructura,
  y deja claro que nada de eso es parte del paquete.
- **`verify_gguf.py` completado**: antes solo comparaba pesos — por eso no
  cazo el bug de los metadatos. Ahora los revisa y **devuelve codigo de
  error**, asi que puede ir en CI para impedir publicar un GGUF roto.
- **`scripts/grafica_comparativa.py`**: genera las cuatro imagenes del README
  (claro/oscuro) leyendo `runs/*.json`. Los numeros no se escriben a mano.

**Pendiente de estructura**: `cli.py` va por 600 lineas con 7 comandos y pide
partirse en un paquete `cli/`. No se hizo para no mezclarlo con trabajo a
medias.

## Al clonar en otra maquina

```bash
git clone https://github.com/1mano1/TinyQ.git
cd Octuma
pip install -e ".[hf,gguf,dev]"
pytest -q          # deben pasar 60
```

**Los modelos NO estan en el repo.** `.gguf`, `.tq`, `.safetensors` y `out/`
estan en `.gitignore` a proposito: pesan gigabytes. Un clon recien hecho tiene
el codigo y los resultados en `runs/*.json`, pero **`out/` viene vacio**.

Para recuperar un modelo hay dos caminos:

```bash
# 1) bajarlo de Hugging Face (privados: hace falta el token del autor)
hf download Imanol11/qwen3b-int4-TinyQ --local-dir out/qwen3b   # huggingface-cli ya no existe

# 2) volver a cuantizarlo desde cero (mas lento, pero no depende de nada)
octuma quantize Qwen/Qwen2.5-3B-Instruct
```

Los `.gguf` de Hugging Face **ya son los arreglados** (resubidos el
2026-09-20). En esta maquina quedan ademas los `out/*-int4-PUBLICADO.gguf`:
son copias de los degradados que estaban publicados antes, guardadas solo para
poder repetir la comparacion antes/despues. No son los buenos.

## Proyecto hermano

`https://github.com/1mano1/ia-local-android` — la app Android que ejecuta
estos modelos, **tambien llamada Octuma**. En local: `C:/ia-local-android`.
Ahi solo hay diseño todavia, ningun codigo.

## Como trabaja el autor

Imanol, de Colima, habla espanol informal. Prefiere explicaciones directas y
sin rodeos: si algo fallo, decirlo de frente con el numero que lo prueba.
Quiere resultados honestos antes que resultados bonitos — un hallazgo negativo
bien medido le sirve mas que una tabla sin asteriscos.
