#!/usr/bin/env bash
# Vigilante que corre DENTRO del pod: cuando el barrido termina, apaga el pod
# para no seguir gastando creditos. Tiene ademas un tope absoluto de horas por
# si algo se cuelga.
#
#   nohup bash scripts/watchdog.sh <POD_ID> <GRACIA_MIN> <TOPE_HORAS> &
#
# La clave de RunPod se lee de /workspace/.runpod_key (permisos 600).
set -uo pipefail

POD_ID="${1:?falta el id del pod}"
GRACE_MIN="${2:-45}"
MAX_HOURS="${3:-8}"
KEY_FILE="/workspace/.runpod_key"
LOG=/workspace/watchdog.log

started=$(date +%s)
echo "$(date -Is) vigilante activo · pod=$POD_ID gracia=${GRACE_MIN}min tope=${MAX_HOURS}h" >> "$LOG"

terminate() {
  local motivo="$1"
  echo "$(date -Is) apagando el pod: $motivo" >> "$LOG"
  python - "$POD_ID" <<'PY' >> "$LOG" 2>&1
import sys, json, urllib.request
pod_id = sys.argv[1]
key = open("/workspace/.runpod_key").read().strip()
q = {"query": "mutation($id: String!){ podTerminate(input:{podId:$id}) }",
     "variables": {"id": pod_id}}
req = urllib.request.Request(
    f"https://api.runpod.io/graphql?api_key={key}",
    data=json.dumps(q).encode(), headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req, timeout=60).read().decode())
PY
  exit 0
}

while true; do
  ahora=$(date +%s)
  horas=$(( (ahora - started) / 3600 ))
  if [ "$horas" -ge "$MAX_HOURS" ]; then
    terminate "se alcanzo el tope de ${MAX_HOURS} horas"
  fi

  # se vigilan las dos fases: el barrido y la generacion de artefactos
  if ! pgrep -f "run_sweep.py|export_artifacts.py" > /dev/null; then
    echo "$(date -Is) el barrido termino; esperando ${GRACE_MIN} min de gracia" >> "$LOG"
    for _ in $(seq 1 "$GRACE_MIN"); do
      sleep 60
      # si alguien relanza el barrido durante la gracia, se cancela el apagado
      if pgrep -f "run_sweep.py|export_artifacts.py" > /dev/null; then
        echo "$(date -Is) el barrido volvio a correr; se cancela el apagado" >> "$LOG"
        continue 2
      fi
    done
    terminate "el barrido termino y paso la gracia"
  fi
  sleep 60
done
