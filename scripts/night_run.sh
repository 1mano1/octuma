#!/usr/bin/env bash
# Encadena todo el trabajo sin supervision: espera al barrido, corre las
# ablaciones, genera los modelos finales y, si hay token, los publica.
#
#   (nohup bash scripts/night_run.sh > /workspace/night.log 2>&1 &)
#
# El vigilante (watchdog.sh) apaga el pod cuando esto termina, pero solo si
# encuentra el centinela /workspace/.modelos_a_salvo que se escribe abajo.
set -uo pipefail

cd /workspace/TinyQ
# La cache va en /workspace: el disco de root son 30 GB y un modelo de 7B en
# FP16 pide 15. Cuando no cabe, la descarga revienta a media transferencia y
# el error llega disfrazado de fallo de red del CAS de Hugging Face.
export HF_HOME=/workspace/hf
# El backend Xet de Hugging Face revienta al reconstruir shards grandes
# ("CAS Client Error"): con esto se baja por HTTP normal, mas lento pero fiable.
export HF_HUB_DISABLE_XET=1
LOG=/workspace/night.log
SENTINELA=/workspace/.modelos_a_salvo
rm -f "$SENTINELA"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

# Cada paso pasa por aqui. Antes nada miraba el codigo de salida: el OOM killer
# mato el barrido a mitad y la cadena siguio hasta anunciar "todo listo".
FALLOS=0
paso() {
  local nombre="$1"; shift
  say "$nombre"
  if "$@" >> "$LOG" 2>&1; then
    say "$nombre: ok"
    return 0
  fi
  local code=$?
  # 137 = SIGKILL, casi siempre el OOM killer del sistema
  if [ "$code" -eq 137 ]; then
    say "FALLO en '$nombre': el sistema lo mato por falta de memoria (137)"
  else
    say "FALLO en '$nombre': termino con codigo $code"
  fi
  FALLOS=$(( FALLOS + 1 ))
  return "$code"
}

espacio_libre_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }

say "espacio libre: / = $(espacio_libre_gb /) GB · /workspace = $(espacio_libre_gb /workspace) GB"
say "commit en uso: $(git rev-parse --short HEAD)"

say "esperando a que termine el barrido principal"
# El patron excluye a este propio script para que no se espere a si mismo
while pgrep -f "[r]un_sweep.py" > /dev/null; do sleep 60; done
say "barrido principal terminado"

paso "ablaciones (bits, tamano de grupo, calibracion)" \
  python scripts/run_sweep.py experiments/sweep.yaml --device cuda --ablations

paso "generando modelos finales y GGUF" \
  python scripts/export_artifacts.py --device cuda

python scripts/run_sweep.py --table > runs/TABLA.md 2>>"$LOG" \
  && say "tabla guardada en runs/TABLA.md" \
  || say "no se pudo generar la tabla"

# La subida solo ocurre si hay token: sin el, los modelos se pierden al apagar
TOKEN="${HF_TOKEN:-}"
if [ -f /workspace/.hf_token ]; then TOKEN=$(cat /workspace/.hf_token); fi
if [ -z "$TOKEN" ] && [ -f /root/.cache/huggingface/token ]; then
  TOKEN=$(cat /root/.cache/huggingface/token)
fi

SUBIDOS=()
if [ -n "$TOKEN" ]; then
  export HF_TOKEN="$TOKEN"
  # el usuario de Hugging Face no tiene por que coincidir con el de GitHub
  USER_HF="${HF_USER:-}"
  if [ -z "$USER_HF" ] && [ -f /workspace/.hf_user ]; then
    USER_HF=$(cat /workspace/.hf_user)
  fi
  USER_HF="${USER_HF:-Imanol11}"
  for dir in out/*-int4-g32; do
    [ -d "$dir" ] || continue
    short=$(basename "$dir" | cut -d- -f1)
    gguf="out/${short}-int4.gguf"
    repo="${USER_HF}/$(basename "$dir" | sed 's/-g32//')-TinyQ"
    # privados a proposito: se publican cuando el autor decida
    say "subiendo $dir -> $repo (privado)"
    if python scripts/upload_hf.py "$dir" --repo "$repo" --private \
         ${gguf:+--gguf "$gguf"} >> "$LOG" 2>&1; then
      SUBIDOS+=("$repo")
    else
      say "fallo la subida de $short"
      FALLOS=$(( FALLOS + 1 ))
    fi
  done
  say "subidas terminadas"
else
  say "SIN TOKEN: los modelos quedan solo en el pod y se perderian al apagarlo"
  FALLOS=$(( FALLOS + 1 ))
fi

# El centinela no se escribe por haber llamado a la subida, sino por haber
# comprobado contra Hugging Face que cada repo existe y trae archivos. Es la
# unica prueba de que apagar el pod no borra el trabajo de la noche.
verificar_subidas() {
  [ "${#SUBIDOS[@]}" -gt 0 ] || return 1
  HF_REPOS="${SUBIDOS[*]}" python - <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN") or None)
faltan = []
for repo in os.environ["HF_REPOS"].split():
    try:
        info = api.repo_info(repo)
        n = len(info.siblings or [])
        print(f"  verificado {repo}: privado={info.private} archivos={n}")
        if n == 0:
            faltan.append(repo)
    except Exception as exc:
        print(f"  NO verificado {repo}: {type(exc).__name__}")
        faltan.append(repo)
sys.exit(1 if faltan else 0)
PY
}

if verificar_subidas >> "$LOG" 2>&1; then
  esperados=$(ls -d out/*-int4-g32 2>/dev/null | wc -l)
  if [ "${#SUBIDOS[@]}" -eq "$esperados" ]; then
    : > "$SENTINELA"
    say "modelos verificados en Hugging Face (${#SUBIDOS[@]}/$esperados); el pod puede apagarse"
  else
    say "solo ${#SUBIDOS[@]} de $esperados modelos llegaron a Hugging Face; el pod NO se apagara"
  fi
else
  say "no se pudo verificar la subida; el pod NO se apagara y seguira cobrando"
fi

if [ "$FALLOS" -gt 0 ]; then
  say "terminado CON $FALLOS fallo(s); revisa el log antes de confiar en los resultados"
  exit 1
fi
say "todo listo y verificado; el vigilante apagara el pod tras la gracia"
