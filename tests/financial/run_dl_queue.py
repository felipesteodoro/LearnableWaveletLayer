"""
Launch all DL financial experiments across 7 GPUs.

Auto-resumes from checkpoint if queue_status.json exists — safe to re-run
after any crash or interruption.

Usage:
    # Terminal 1 — start (or resume) the queue
    cd tests/financial
    python run_dl_queue.py

    # Terminal 2 — monitor dashboard
    python queue/dashboard.py

    # Force fresh start (ignores checkpoint):
    python run_dl_queue.py --fresh

Environment variables:
    N_GPUS      number of GPUs (default: 7)
    GPU_IDS     comma-separated list, e.g. "0,1,2" (overrides N_GPUS)
    RETRY_WAIT  seconds before retrying a failed job (default: 60)
    TICKERS     comma-separated subset, e.g. "PETR4.SA,VALE3.SA"
    DL_MODELS   comma-separated subset, e.g. "CNN,LSTM"
    MODES       comma-separated subset, e.g. "raw,learned_wavelet"
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from gpu_queue import GPUJobQueueManager, ExperimentJob  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_TICKERS = [
    "ABEV3.SA", "B3SA3.SA",  "BBAS3.SA", "BBDC4.SA", "BRKM5.SA",
    "COGN3.SA", "CSNA3.SA",  "CYRE3.SA", "EZTC3.SA", "GGBR4.SA",
    "HYPE3.SA", "ITUB4.SA",  "LREN3.SA", "MGLU3.SA", "MRVE3.SA",
    "MULT3.SA", "PETR4.SA",  "RADL3.SA", "RENT3.SA", "SUZB3.SA",
    "UGPA3.SA", "USIM5.SA",  "VALE3.SA", "VIVT3.SA", "WEGE3.SA",
]
ALL_MODELS = ["CNN", "LSTM", "CNN_LSTM", "Transformer"]
ALL_MODES  = ["raw", "db4", "learned_wavelet"]

BASE_CONFIG = {
    "sequence_length": 30,
    "n_folds": 5,
    "embargo_days": 10,
    "epochs": 100,
    "batch_size": 64,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 7,
    "wavelet_levels": 2,
    "kernel_size": 32,
    "reg_energy": 1e-2,
    "reg_high_dc": 1e-2,
    "reg_smooth": 1e-3,
    "transaction_cost": 0.001,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _from_env(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key, "").strip()
    return [v.strip() for v in raw.split(",") if v.strip()] if raw else default


def _gpu_ids() -> list[int]:
    raw = os.environ.get("GPU_IDS", "").strip()
    if raw:
        return [int(x) for x in raw.split(",") if x.strip()]
    return list(range(int(os.environ.get("N_GPUS", "7"))))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GPU DL experiment queue")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoint and start from scratch",
    )
    args = parser.parse_args()

    tickers    = _from_env("TICKERS", ALL_TICKERS)
    models     = _from_env("DL_MODELS", ALL_MODELS)
    modes      = _from_env("MODES", ALL_MODES)
    gpu_ids    = _gpu_ids()
    retry_wait = int(os.environ.get("RETRY_WAIT", "60"))

    all_jobs = [
        ExperimentJob(
            ticker=ticker,
            model_name=model,
            mode=mode,
            config=BASE_CONFIG.copy(),
        )
        for ticker, model, mode in itertools.product(tickers, models, modes)
    ]

    print(
        f"\n{'='*62}\n"
        f"  GPU Job Queue {'(FRESH START)' if args.fresh else '(auto-resume)'}\n"
        f"  Total jobs : {len(all_jobs)}\n"
        f"  GPUs       : {gpu_ids}\n"
        f"  Tickers    : {len(tickers)}\n"
        f"  Models     : {models}\n"
        f"  Modes      : {modes}\n"
        f"  Retry wait : {retry_wait}s  |  max retries: 2\n"
        f"{'='*62}\n"
        f"  Monitor    : python queue/dashboard.py\n"
        f"{'='*62}\n"
    )

    if args.fresh:
        manager = GPUJobQueueManager(gpu_ids=gpu_ids, retry_wait=retry_wait)
        manager.add_many(all_jobs)
    else:
        manager = GPUJobQueueManager.resume_or_create(
            all_jobs=all_jobs,
            gpu_ids=gpu_ids,
            retry_wait=retry_wait,
        )

    manager.run()
    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
