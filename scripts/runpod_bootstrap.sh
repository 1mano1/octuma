#!/usr/bin/env bash
# Se ejecuta DENTRO del pod de RunPod. Deja todo listo y lanza el barrido.
#
#   bash scripts/runpod_bootstrap.sh            # barrido principal
#   bash scripts/runpod_bootstrap.sh --ablations
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /workspace
if [ ! -d TinyQ ]; then
  git clone "${TINYQ_REPO:-https://github.com/imanolr/tinyq.git}" TinyQ
fi
cd TinyQ

python -m pip install -q --upgrade pip
python -m pip install -q -e ".[hf,gguf,dev]" pyyaml

# los pesos descargados viven en el volumen persistente: no se re-descargan
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME" runs

echo "== pruebas rapidas =="
python -m pytest -q

echo "== barrido =="
python scripts/run_sweep.py experiments/sweep.yaml --device cuda "$@" \
  2>&1 | tee -a runs/sweep.log

echo "== tabla =="
python scripts/run_sweep.py --table | tee runs/TABLA.md
echo "Listo. Descarga la carpeta runs/ antes de apagar el pod."
