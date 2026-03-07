#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "══════════════════════════════════════"
echo "Fase 1: 02 (GPU 0) + 03 (GPU 1) em paralelo"
echo "Diretório: $DIR"
echo "Início: $(date)"
echo "══════════════════════════════════════"

GPU_ID=0 jupyter nbconvert --to notebook --execute \
  --output 02_dl_raw_experiments_out.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  "$DIR/02_dl_raw_experiments.ipynb" &
PID_02=$!

GPU_ID=1 jupyter nbconvert --to notebook --execute \
  --output 03_dl_wavelet_experiments_out.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  "$DIR/03_dl_wavelet_experiments.ipynb" &
PID_03=$!

echo "PIDs: 02=$PID_02 (GPU0)  03=$PID_03 (GPU1)"
echo "Aguardando fase 1..."

wait $PID_02; R02=$?; echo "NB 02 finished (exit=$R02) at $(date)"
wait $PID_03; R03=$?; echo "NB 03 finished (exit=$R03) at $(date)"

echo ""
echo "══════════════════════════════════════"
echo "Fase 2: 04 (GPU 0)"
echo "══════════════════════════════════════"

GPU_ID=0 jupyter nbconvert --to notebook --execute \
  --output 04_learned_wavelet_experiments_out.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  "$DIR/04_learned_wavelet_experiments.ipynb"
R04=$?
echo "NB 04 finished (exit=$R04) at $(date)"

echo ""
echo "══════════════════════════════════════"
echo "RESULTADO FINAL:"
echo "  02=$R02  03=$R03  04=$R04"
echo "Fim: $(date)"
echo "══════════════════════════════════════"
