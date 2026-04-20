"""
Subprocess entry point — one process per GPU worker.

Environment variables injected by GPUWorker:
  EXP_CONFIG_FILE       path to JSON file with ExperimentJob fields
  CUDA_VISIBLE_DEVICES  GPU index (already set)

Exit codes:
  0  success (including skip-already-done)
  1  experiment error (logged to the job's .log file)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Results are stored relative to the tests/financial directory
FINANCIAL_DIR = Path(__file__).parent.parent
RESULTS_DIR = FINANCIAL_DIR / "results"


def _results_exist(ticker: str, model_name: str, mode: str) -> bool:
    """
    Second line of defense against re-running finished jobs.
    Checks for a complete metrics.json in the expected results directory.
    This protects against crashes that corrupted queue_status.json.
    """
    metrics_file = RESULTS_DIR / ticker / f"{model_name}_{mode}" / "metrics.json"
    return metrics_file.exists()


def main() -> int:
    config_file = os.environ.get("EXP_CONFIG_FILE")
    if not config_file:
        logger.error("EXP_CONFIG_FILE env var not set.")
        return 1

    with open(config_file) as f:
        job = json.load(f)

    ticker: str = job["ticker"]
    model_name: str = job["model_name"]
    mode: str = job["mode"]
    config: dict = job.get("config", {})
    gpu: str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # Skip-if-done: results already on disk from a previous successful run
    if _results_exist(ticker, model_name, mode):
        logger.info(
            "SKIP (results exist): %s / %s_%s — marking done without re-running.",
            ticker, model_name, mode,
        )
        return 0

    logger.info(
        "Job start | ticker=%-10s  model=%-12s  mode=%-16s  GPU=%s",
        ticker, model_name, mode, gpu,
    )

    try:
        _run_experiment(ticker, model_name, mode, config)
    except Exception:
        logger.error("Experiment failed:\n%s", traceback.format_exc())
        return 1

    logger.info(
        "Job done  | ticker=%-10s  model=%-12s  mode=%-16s",
        ticker, model_name, mode,
    )
    return 0


def _run_experiment(ticker: str, model_name: str, mode: str, config: dict) -> None:
    # Add tests/financial to path so src/ is importable
    sys.path.insert(0, str(FINANCIAL_DIR))

    from src.pipeline import FinancialExperimentPipeline

    pipeline = FinancialExperimentPipeline(
        ticker=ticker,
        model_name=model_name,
        mode=mode,
        config=config,
        results_dir=str(RESULTS_DIR),
    )
    pipeline.run()


if __name__ == "__main__":
    sys.exit(main())
