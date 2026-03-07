#!/bin/bash
# Script to execute notebooks in sequence: 02, 03, 04, 01, 05, 06
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

LOG="$DIR/notebook_execution.log"
echo "========================================" > "$LOG"
echo "Notebook execution started: $(date)" >> "$LOG"
echo "========================================" >> "$LOG"

NOTEBOOKS=(
    "02_dl_raw_experiments.ipynb"
    "03_dl_wavelet_experiments.ipynb"
    "04_learned_wavelet_experiments.ipynb"
#    "01_ml_experiments.ipynb"
#    "05_comparison_analysis.ipynb"
#    "06_learned_filter_analysis.ipynb"
)

TOTAL=${#NOTEBOOKS[@]}
COMPLETED=0

for nb in "${NOTEBOOKS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    echo "" >> "$LOG"
    echo "[$COMPLETED/$TOTAL] Starting: $nb at $(date)" >> "$LOG"
    
    if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$nb" >> "$LOG" 2>&1; then
        echo "[$COMPLETED/$TOTAL] SUCCESS: $nb at $(date)" >> "$LOG"
    else
        echo "[$COMPLETED/$TOTAL] FAILED: $nb at $(date)" >> "$LOG"
        echo "Continuing to next notebook..." >> "$LOG"
    fi
done

echo "" >> "$LOG"
echo "========================================" >> "$LOG"
echo "All notebooks finished: $(date)" >> "$LOG"
echo "========================================" >> "$LOG"
