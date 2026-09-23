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


# Regla del autor: primero se comprueba que los modelos esten a salvo, y solo
# despues se apaga. night_run.sh escribe este centinela unicamente cuando ya
# verifico contra Hugging Face que cada repo existe y tiene archivos.
SENTINELA=/workspace/.modelos_a_salvo

modelos_a_salvo() {
  [ -f "$SENTINELA" ]
}

# Si hay artefactos generados pero nunca se confirmo la subida, apagar
# significa perderlos. En ese caso el vigilante prefiere seguir cobrando y
# dejarlo dicho en el log antes que borrar trabajo de toda una noche.
apagado_seguro() {
  if modelos_a_salvo; then return 0; fi
  if [ -n "$(ls -A /workspace/Octuma/out 2>/dev/null)" ]; then
    echo "$(date -Is) NO se apaga: hay modelos en out/ y no se confirmo la subida a Hugging Face" >> "$LOG"
    return 1
  fi
  return 0
}
# Baja el pod y NO se da por satisfecho hasta confirmarlo. Un 403 pasajero de
# la API dejo el pod encendido seis horas: por eso aqui se reintenta con
# espera creciente y se verifica que el pod de verdad haya desaparecido.
apagar_pod() {
  local key; key=$(cat "$KEY_FILE")
  local code
  # API REST actual; si no responde, se intenta el GraphQL viejo
  code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE \
    -H "Authorization: Bearer $key" \
    "https://rest.runpod.io/v1/pods/$POD_ID" || echo 000)
  if [ "$code" != "200" ] && [ "$code" != "204" ]; then
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
      -H 'Content-Type: application/json' -H "Authorization: Bearer $key" \
      -d "{\"query\":\"mutation{ podTerminate(input:{podId:\\\"$POD_ID\\\"}) }\"}" \
      https://api.runpod.io/graphql || echo 000)
  fi
  echo "$code"
}

pod_sigue_vivo() {
  local key; key=$(cat "$KEY_FILE")
  local http; http=$(curl -s -o /tmp/pod_estado -w '%{http_code}' \
    -H "Authorization: Bearer $key" \
    "https://rest.runpod.io/v1/pods/$POD_ID" || echo 000)
  # 404 = ya no existe; 000/500 = no se sabe, se asume vivo por prudencia
  [ "$http" = "404" ] && return 1
  return 0
}

terminate() {
  local motivo="$1"
  echo "$(date -Is) apagando el pod: $motivo" >> "$LOG"
  local espera=30
  for intento in 1 2 3 4 5 6 7 8; do
    local code; code=$(apagar_pod)
    echo "$(date -Is) intento $intento de apagado -> HTTP $code" >> "$LOG"
    sleep 20
    if ! pod_sigue_vivo; then
      echo "$(date -Is) confirmado: el pod ya no existe" >> "$LOG"
      exit 0
    fi
    echo "$(date -Is) el pod sigue vivo; reintento en ${espera}s" >> "$LOG"
    sleep "$espera"
    espera=$(( espera * 2 ))
  done
  # Ocho intentos fallidos: el vigilante NO se muere en silencio como la vez
  # pasada. Sigue vivo y vuelve a intentarlo en el siguiente ciclo.
  echo "$(date -Is) ALERTA: no se pudo apagar el pod tras 8 intentos; sigue cobrando" >> "$LOG"
  return 1
}

while true; do
  ahora=$(date +%s)
  horas=$(( (ahora - started) / 3600 ))
  if [ "$horas" -ge "$MAX_HOURS" ]; then
    if apagado_seguro; then terminate "se alcanzo el tope de ${MAX_HOURS} horas"; fi
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
    if apagado_seguro; then terminate "el barrido termino y paso la gracia"; fi
  fi
  sleep 60
done
