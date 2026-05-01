#!/bin/bash
# =============================================================================
# Execução dos experimentos sintéticos com paralelização configurável por GPU.
#
# Notebooks GPU  (paralelizáveis): 02, 03, 04
# Notebooks CPU  (sequenciais)   : 01, 05, 06
#
# Estratégia por número de GPUs:
#   1 GPU  →  02 → 03 → 04  (sequencial na GPU 0)
#   2 GPUs →  02 (GPU0) + 03 (GPU1) em paralelo → 04 (GPU0)
#   3 GPUs →  02 (GPU0) + 03 (GPU1) + 04 (GPU2) em paralelo
#
# Uso:
#   ./run_experiments.sh                      # 2 GPUs (padrão)
#   ./run_experiments.sh --gpus 1             # 1 GPU
#   ./run_experiments.sh --gpus 3             # 3 GPUs
#   ./run_experiments.sh --gpus 2 --cpu       # GPUs + notebooks CPU (01,05,06)
#   ./run_experiments.sh --gpus 2 --smoke     # smoke test (epochs=1, configs=2)
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# ── Garantir novo process group para poder matar tudo com kill -- -$$ ─────────
# Rodar em sub-shell com setsid para que cada execução forme seu próprio grupo
[[ "$(ps -o pgid= -p $$)" == "$(ps -o pid= -p $$)" ]] || exec setsid bash "$0" "$@"

# ── Cleanup: mata papermill + kernels Jupyter ao receber sinal ────────────────
_cleanup() {
    echo "" | tee -a "${LOG:-/dev/null}"
    echo "[cleanup] Encerrando todos os subprocessos do grupo $$..." | tee -a "${LOG:-/dev/null}"
    # Mata o grupo inteiro (papermill + ipykernel subprocessos)
    pkill -9 -s $$ 2>/dev/null || true
    pkill -9 -f "ipykernel_launcher" 2>/dev/null || true
    echo "[cleanup] Concluído." | tee -a "${LOG:-/dev/null}"
}
trap '_cleanup' EXIT INT TERM

# ── Defaults ─────────────────────────────────────────────────────────────────
N_GPUS=2
RUN_CPU=0
SMOKE=0

# ── Argparse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            N_GPUS="$2"
            shift 2
            ;;
        --cpu)
            RUN_CPU=1
            shift
            ;;
        --smoke)
            SMOKE=1
            shift
            ;;
        *)
            echo "Uso: $0 [--gpus 1|2|3] [--cpu] [--smoke]"
            exit 1
            ;;
    esac
done

if [[ "$N_GPUS" != "1" && "$N_GPUS" != "2" && "$N_GPUS" != "3" ]]; then
    echo "Erro: --gpus deve ser 1, 2 ou 3 (recebido: $N_GPUS)"
    exit 1
fi

# ── Variáveis de ambiente para os notebooks ───────────────────────────────────
export EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-0}"
export MAX_GRID_CONFIGS="${MAX_GRID_CONFIGS:-0}"
if [[ "$SMOKE" -eq 1 ]]; then
    export EPOCHS_OVERRIDE=1
    export MAX_GRID_CONFIGS=2
fi

# ── Logs ──────────────────────────────────────────────────────────────────────
LOGDIR="$DIR/logs"
mkdir -p "$LOGDIR"
LOG="$DIR/run_experiments.log"

echo "════════════════════════════════════════════" | tee "$LOG"
echo "Synthetic experiments: $(date)"               | tee -a "$LOG"
echo "  N_GPUS=${N_GPUS}  RUN_CPU=${RUN_CPU}  SMOKE=${SMOKE}" | tee -a "$LOG"
echo "  EPOCHS_OVERRIDE=${EPOCHS_OVERRIDE}"         | tee -a "$LOG"
echo "  MAX_GRID_CONFIGS=${MAX_GRID_CONFIGS}"        | tee -a "$LOG"
echo "  Logs individuais: $LOGDIR/"                  | tee -a "$LOG"
echo "════════════════════════════════════════════" | tee -a "$LOG"

# ── Helper: executa um notebook numa GPU ──────────────────────────────────────
run_nb() {
    local gpu_id=$1
    local nb=$2
    local suffix="${3:-gpu${gpu_id}}"
    local out="${nb%.ipynb}_${suffix}.ipynb"
    local nb_log="$LOGDIR/${nb%.ipynb}_gpu${gpu_id}.log"
    local t0=$(date +%s)

    echo "[GPU ${gpu_id}] START: $nb  →  $out  ($(date))" | tee -a "$LOG"
    echo "              log: $nb_log" | tee -a "$LOG"

    if GPU_ID="$gpu_id" papermill \
            --kernel python3 \
            --no-progress-bar \
            --log-output \
            "$nb" "$out" > "$nb_log" 2>&1; then
        echo "[GPU ${gpu_id}] OK:    $nb  ($(( $(date +%s) - t0 ))s)" | tee -a "$LOG"
    else
        echo "[GPU ${gpu_id}] FAIL:  $nb  ($(( $(date +%s) - t0 ))s)" | tee -a "$LOG"
        echo "  Últimas linhas do log:" | tee -a "$LOG"
        tail -20 "$nb_log" | tee -a "$LOG"
        return 1
    fi
}

# ── Helper: executa um notebook sem GPU ──────────────────────────────────────
run_nb_cpu() {
    local nb=$1
    local out="${nb%.ipynb}_out.ipynb"
    local nb_log="$LOGDIR/${nb%.ipynb}_cpu.log"
    local t0=$(date +%s)

    echo "[CPU] START: $nb  →  $out  ($(date))" | tee -a "$LOG"
    echo "      log: $nb_log" | tee -a "$LOG"

    if papermill \
            --kernel python3 \
            --no-progress-bar \
            --log-output \
            "$nb" "$out" > "$nb_log" 2>&1; then
        echo "[CPU] OK:    $nb  ($(( $(date +%s) - t0 ))s)" | tee -a "$LOG"
    else
        echo "[CPU] FAIL:  $nb  ($(( $(date +%s) - t0 ))s)" | tee -a "$LOG"
        echo "  Últimas linhas do log:" | tee -a "$LOG"
        tail -20 "$nb_log" | tee -a "$LOG"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# FASE GPU
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$N_GPUS" -eq 1 ]]; then
    # ── 1 GPU: sequencial ────────────────────────────────────────────────────
    echo "" | tee -a "$LOG"
    echo "── Fase GPU: 02 → 03 → 04 (GPU 0, sequencial) ──" | tee -a "$LOG"

    run_nb 0 "02_dl_raw_experiments.ipynb"
    run_nb 0 "03_dl_wavelet_experiments.ipynb"
    run_nb 0 "04_learned_wavelet_experiments.ipynb"

elif [[ "$N_GPUS" -eq 2 ]]; then
    # ── 2 GPUs: 02+03 paralelo, depois 04 ────────────────────────────────────
    echo "" | tee -a "$LOG"
    echo "── Fase GPU 1/2: 02 (GPU0) + 03 (GPU1) em paralelo ──" | tee -a "$LOG"

    run_nb 0 "02_dl_raw_experiments.ipynb" & PID_02=$!
    run_nb 1 "03_dl_wavelet_experiments.ipynb" & PID_03=$!

    RC_02=0; RC_03=0
    wait $PID_02 || RC_02=$?
    wait $PID_03 || RC_03=$?

    echo "" | tee -a "$LOG"
    echo "── Fase GPU 2/2: 04 (GPU0) ──" | tee -a "$LOG"
    run_nb 0 "04_learned_wavelet_experiments.ipynb"

elif [[ "$N_GPUS" -eq 3 ]]; then
    # ── 3 GPUs: 02+03+04 todos em paralelo ───────────────────────────────────
    echo "" | tee -a "$LOG"
    echo "── Fase GPU: 02 (GPU0) + 03 (GPU1) + 04 (GPU2) em paralelo ──" | tee -a "$LOG"

    run_nb 0 "02_dl_raw_experiments.ipynb" & PID_02=$!
    run_nb 1 "03_dl_wavelet_experiments.ipynb" & PID_03=$!
    run_nb 2 "04_learned_wavelet_experiments.ipynb" & PID_04=$!

    RC_02=0; RC_03=0; RC_04=0
    wait $PID_02 || RC_02=$?
    wait $PID_03 || RC_03=$?
    wait $PID_04 || RC_04=$?
fi

# ─────────────────────────────────────────────────────────────────────────────
# FASE CPU (opcional)
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$RUN_CPU" -eq 1 ]]; then
    echo "" | tee -a "$LOG"
    echo "── Fase CPU: 01 → 05 → 06 ──" | tee -a "$LOG"

    run_nb_cpu "01_ml_experiments.ipynb"
    run_nb_cpu "05_comparison_analysis.ipynb"
    run_nb_cpu "06_learned_filter_analysis.ipynb"
fi

# ── Resumo ────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════" | tee -a "$LOG"
echo "Concluído: $(date)" | tee -a "$LOG"
echo "Log completo: $LOG" | tee -a "$LOG"
echo "════════════════════════════════════════════" | tee -a "$LOG"
