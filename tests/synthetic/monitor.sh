#!/bin/bash
# =============================================================================
# Monitor dos experimentos sintéticos.
# Verifica a cada INTERVAL segundos se os notebooks ainda estão rodando.
# Se um processo morrer inesperadamente (OOM, kernel crash), reinicia-o
# numa GPU livre.
# =============================================================================
INTERVAL=${INTERVAL:-300}   # 5 minutos por padrão
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/../../.venv/bin/activate"
LOGDIR="$DIR/logs"
MLOG="$LOGDIR/monitor.log"
mkdir -p "$LOGDIR"

source "$VENV"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(_ts)] $*" | tee -a "$MLOG"; }

# Mapeamento: notebook → GPU preferencial
declare -A NB_GPU=(
    ["02_dl_raw_experiments.ipynb"]=0
    ["03_dl_wavelet_experiments.ipynb"]=1
    ["04_learned_wavelet_experiments.ipynb"]=2
)
declare -A NB_OUT=(
    ["02_dl_raw_experiments.ipynb"]="02_dl_raw_experiments_gpu0.ipynb"
    ["03_dl_wavelet_experiments.ipynb"]="03_dl_wavelet_experiments_gpu1.ipynb"
    ["04_learned_wavelet_experiments.ipynb"]="04_learned_wavelet_experiments_gpu2.ipynb"
)
declare -A NB_LOG=(
    ["02_dl_raw_experiments.ipynb"]="02_dl_raw_experiments_gpu0.log"
    ["03_dl_wavelet_experiments.ipynb"]="03_dl_wavelet_experiments_gpu1.log"
    ["04_learned_wavelet_experiments.ipynb"]="04_learned_wavelet_experiments_gpu2.log"
)

# PIDs ativos por notebook
declare -A NB_PID

# Verifica se um notebook já terminou com sucesso (arquivo de saída existe e é
# maior que 100 KB — evita reprocessar notebooks já concluídos)
already_done() {
    local nb=$1
    local out="$DIR/${NB_OUT[$nb]}"
    [[ -f "$out" ]] && [[ $(stat -c%s "$out") -gt 102400 ]]
}

# GPU com mais memória livre
free_gpu() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader \
        | sort -t',' -k2 -rn \
        | head -1 \
        | cut -d',' -f1 \
        | tr -d ' '
}

# Inicia um notebook numa GPU
start_nb() {
    local nb=$1
    local gpu=${2:-${NB_GPU[$nb]}}
    local out="$DIR/${NB_OUT[$nb]}"
    local nblog="$LOGDIR/${NB_LOG[$nb]}"

    log "STARTING $nb na GPU $gpu"
    GPU_ID="$gpu" papermill \
        --kernel python3 \
        --no-progress-bar \
        --log-output \
        "$DIR/$nb" "$out" >> "$nblog" 2>&1 &
    NB_PID[$nb]=$!
    log "  PID=${NB_PID[$nb]}"
}

# ── Descobrir processos papermill já em execução ──────────────────────────────
log "════════════════════════════════════════════"
log "Monitor iniciado (interval=${INTERVAL}s)"
log "════════════════════════════════════════════"

for nb in "${!NB_GPU[@]}"; do
    pid=$(pgrep -f "papermill.*${nb}" 2>/dev/null | head -1)
    if [[ -n "$pid" ]]; then
        NB_PID[$nb]=$pid
        log "Adotando processo existente: $nb PID=$pid"
    elif already_done "$nb"; then
        log "Já concluído (pulando): $nb"
    else
        log "Não encontrado em execução, iniciando: $nb"
        start_nb "$nb"
    fi
done

# ── Loop de monitoramento ─────────────────────────────────────────────────────
while true; do
    sleep "$INTERVAL"
    log "──── check ────────────────────────────────"

    # Mosta uso de GPU
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader \
        | awk -F',' '{printf "  GPU%s: used=%s free=%s\n", $1, $2, $3}' \
        | tee -a "$MLOG"

    all_done=true
    for nb in "${!NB_GPU[@]}"; do
        pid=${NB_PID[$nb]:-""}

        # Já concluído com sucesso?
        if already_done "$nb" && { [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; }; then
            log "  ✓ $nb — concluído"
            unset NB_PID[$nb]
            continue
        fi

        # Processo ainda vivo?
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "  ↻ $nb — rodando (PID=$pid)"
            all_done=false
            continue
        fi

        # Processo morreu mas notebook não está completo → reiniciar
        if ! already_done "$nb"; then
            all_done=false
            gpu=$(free_gpu)
            log "  ✗ $nb — FALHOU ou não iniciou. Reiniciando na GPU $gpu..."
            # Mata kernels orfãos relacionados a este notebook
            pkill -9 -f "ipykernel_launcher" 2>/dev/null || true
            sleep 3
            start_nb "$nb" "$gpu"
        fi
    done

    if $all_done; then
        log "Todos os notebooks concluídos. Monitor encerrando."
        break
    fi
done

log "════════════════════════════════════════════"
log "Monitor finalizado: $(_ts)"
log "════════════════════════════════════════════"
