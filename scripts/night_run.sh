#!/usr/bin/env bash
# Encadena todo el trabajo sin supervision: espera al barrido, corre las
# ablaciones, genera los modelos finales y, si hay HF_TOKEN, los publica.
#
#   (nohup bash scripts/night_run.sh > /workspace/night.log 2>&1 &)
#
# El vigilante (watchdog.sh) apaga el pod cuando esto termina.
set -uo pipefail

cd /workspace/TinyQ
export HF_HOME=/workspace/hf
LOG=/workspace/night.log
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

say "esperando a que termine el barrido principal"
while pgrep -f "run_sweep.py" > /dev/null; do sleep 60; done
say "barrido principal terminado"

say "ablaciones (bits, tamano de grupo, calibracion)"
python scripts/run_sweep.py experiments/sweep.yaml --device cuda --ablations >> "$LOG" 2>&1
say "ablaciones terminadas"

say "generando modelos finales y GGUF"
python scripts/export_artifacts.py --device cuda >> "$LOG" 2>&1
say "artefactos listos"

python scripts/run_sweep.py --table > runs/TABLA.md 2>/dev/null
say "tabla guardada en runs/TABLA.md"

# La subida solo ocurre si hay token: sin el, los modelos se pierden al apagar
TOKEN="${HF_TOKEN:-}"
if [ -f /workspace/.hf_token ]; then TOKEN=$(cat /workspace/.hf_token); fi
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
    python scripts/upload_hf.py "$dir" --repo "$repo" --private \
      ${gguf:+--gguf "$gguf"} >> "$LOG" 2>&1 || say "fallo la subida de $short"
  done
  say "subidas terminadas"
else
  say "SIN HF_TOKEN: los modelos quedan solo en el pod y se perderan al apagarlo"
fi

say "todo listo; el vigilante apagara el pod tras la gracia"
