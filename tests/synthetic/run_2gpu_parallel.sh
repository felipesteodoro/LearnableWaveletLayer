#!/bin/bash
# =============================================================================
# Execução paralela dos notebooks DL em 2 GPUs
#
# Estratégia:
#   GPU 0  →  02_dl_raw_experiments.ipynb
#   GPU 1  →  03_dl_wavelet_experiments.ipynb
#   (em paralelo)
#   Depois, notebook 04 na GPU que estiver livre primeiro.
#
# Uso:
#   ./run_2gpu_parallel.sh                 # execução completa
#   EPOCHS_OVERRIDE=1 MAX_GRID_CONFIGS=2 ./run_2gpu_parallel.sh   # smoke test
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

LOG="$DIR/notebook_execution.log"

# Propagar variáveis de ambiente opcionais
export EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-0}"
export MAX_GRID_CONFIGS="${MAX_GRID_CONFIGS:-0}"

echo "========================================" | tee "$LOG"
echo "Parallel 2-GPU execution: $(date)"       | tee -a "$LOG"
echo "  EPOCHS_OVERRIDE=${EPOCHS_OVERRIDE}"     | tee -a "$LOG"
echo "  MAX_GRID_CONFIGS=${MAX_GRID_CONFIGS}"   | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

run_notebook() {
    local gpu_id=$1
    local nb=$2
    local out="${nb%.ipynb}_gpu${gpu_id}.ipynb"
    local start_ts=$(date +%s)

    echo "[GPU $gpu_id] Starting: $nb at $(date)" | tee -a "$LOG"

    if GPU_ID="$gpu_id" jupyter nbconvert \
        --to notebook --execute \
        --output "$out" \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$nb" >> "$LOG" 2>&1; then
        local end_ts=$(date +%s)
        echo "[GPU $gpu_id] SUCCESS: $nb ($(( end_ts - start_ts ))s)" | tee -a "$LOG"
    else
        local end_ts=$(date +%s)
        echo "[GPU $gpu_id] FAILED:  $nb ($(( end_ts - start_ts ))s)" | tee -a "$LOG"
        return 1
    fi
}

# ── Fase 1: notebooks 02 e 03 em paralelo ──
echo ""                                         | tee -a "$LOG"
echo "── Fase 1: 02 (GPU 0) + 03 (GPU 1) ──"   | tee -a "$LOG"

run_notebook 0 "02_dl_raw_experiments.ipynb"     &
PID_GPU0=$!

run_notebook 1 "03_dl_wavelet_experiments.ipynb" &
PID_GPU1=$!

# Aguardar ambos
wait $PID_GPU0 || true
wait $PID_GPU1 || true

# ── Fase 2: notebook 04 (usa GPU 0) ──
echo ""                                         | tee -a "$LOG"
echo "── Fase 2: 04 (GPU 0) ──"                 | tee -a "$LOG"

run_notebook 0 "04_learned_wavelet_experiments.ipynb"

echo ""                                         | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "All done: $(date)"                        | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
