"""
Launch all DL financial experiments across 7 GPUs.

Cada execução cria uma subpasta datada dentro de results/, por exemplo:
    results/2026-05-01/

Isso permite múltiplas simulações do mesmo experimento sem sobrescrever resultados.

Usage:
    # Terminal 1 — nova execução (cria pasta nova com data atual)
    cd tests/financial
    python run_dl_queue.py --fresh

    # Terminal 1 — retoma a execução mais recente (padrão)
    python run_dl_queue.py

    # Terminal 1 — retoma uma execução específica pelo run_id
    python run_dl_queue.py --run-id 2026-05-01

    # Terminal 2 — monitor dashboard (detecta automaticamente a run mais recente)
    python gpu_queue/dashboard.py

    # Terminal 2 — dashboard de uma run específica
    python gpu_queue/dashboard.py --status results/2026-05-01/queue_status.json

Environment variables:
    N_GPUS          number of GPUs (default: 7)
    GPU_IDS         comma-separated list, e.g. "0,1,2" (overrides N_GPUS)
    RETRY_WAIT      seconds before retrying a failed job (default: 60)
    TICKERS         comma-separated subset, e.g. "PETR4.SA,VALE3.SA"
    DL_MODELS       comma-separated subset, e.g. "CNN,LSTM"
    MODES           comma-separated subset, e.g. "raw,learned_wavelet"
    EPOCHS_OVERRIDE override epoch count for all jobs, e.g. "3" for smoke test
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from gpu_queue import GPUJobQueueManager, ExperimentJob  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

_BASE = Path(__file__).parent

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

# BASE_CONFIG is intentionally empty: all hyperparameters are defined in
# config/experiment_config.py and loaded by FinancialExperimentPipeline.__init__().
BASE_CONFIG: dict = {}

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


def _latest_run_id() -> str | None:
    """Retorna o run_id mais recente que possui queue_status.json, ou None."""
    results = _BASE / "results"
    if not results.exists():
        return None
    # Pastas no formato YYYY-MM-DD, ordenadas da mais recente
    dated = sorted(results.glob("????-??-??"), reverse=True)
    for folder in dated:
        if (folder / "queue_status.json").exists():
            return folder.name
    return None


def _new_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GPU DL experiment queue")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Cria uma nova pasta de resultados (nova run com data atual)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="YYYY-MM-DD",
        help="Retoma uma run específica pelo seu run_id (ex: 2026-05-01)",
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

    # Determina o run_id e results_dir
    if args.fresh:
        run_id      = _new_run_id()
        results_dir = f"results/{run_id}"
        mode_label  = f"NOVA RUN  →  {run_id}"
    elif args.run_id:
        run_id      = args.run_id
        results_dir = f"results/{run_id}"
        mode_label  = f"RESUMINDO RUN  →  {run_id}"
    else:
        # Auto-resume: usa o run mais recente; se não existir, cria um novo
        run_id = _latest_run_id() or _new_run_id()
        results_dir = f"results/{run_id}"
        mode_label  = f"AUTO-RESUME  →  {run_id}"

    print(
        f"\n{'='*62}\n"
        f"  GPU Job Queue  |  {mode_label}\n"
        f"{'='*62}\n"
        f"  Results dir : results/{run_id}/\n"
        f"  Total jobs  : {len(all_jobs)}\n"
        f"  GPUs        : {gpu_ids}\n"
        f"  Tickers     : {len(tickers)}\n"
        f"  Models      : {models}\n"
        f"  Modes       : {modes}\n"
        f"  Retry wait  : {retry_wait}s  |  max retries: 2\n"
        f"{'='*62}\n"
        f"  Dashboard   : python gpu_queue/dashboard.py\n"
        f"  Run status  : results/{run_id}/queue_status.json\n"
        f"{'='*62}\n"
    )

    if args.fresh:
        manager = GPUJobQueueManager(
            gpu_ids=gpu_ids,
            retry_wait=retry_wait,
            results_dir=results_dir,
        )
        manager.add_many(all_jobs)
    else:
        manager = GPUJobQueueManager.resume_or_create(
            all_jobs=all_jobs,
            gpu_ids=gpu_ids,
            retry_wait=retry_wait,
            results_dir=results_dir,
        )

    manager.run()
    print(f"\nAll experiments completed.  Results: results/{run_id}/")


if __name__ == "__main__":
    main()
