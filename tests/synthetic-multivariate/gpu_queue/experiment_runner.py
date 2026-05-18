"""
Subprocess entry point — one process per GPU worker.

Environment variables injected by GPUWorker:
  EXP_CONFIG_FILE       path to JSON file with ExperimentJob fields
  CUDA_VISIBLE_DEVICES  GPU index (already set)
  RESULTS_DIR           dated results directory (e.g. results/2026-05-09)

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

SYNTHETIC_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(SYNTHETIC_DIR / "results")))


def _results_exist(model_name: str, mode: str, config_idx: int) -> bool:
    """Guard against re-running a job whose metrics.json already exists."""
    p = RESULTS_DIR / f"{model_name}_{mode}" / f"cfg{config_idx:03d}" / "metrics.json"
    return p.exists()


def main() -> int:
    config_file = os.environ.get("EXP_CONFIG_FILE")
    if not config_file:
        logger.error("EXP_CONFIG_FILE env var not set.")
        return 1

    with open(config_file) as f:
        job = json.load(f)

    model_name: str  = job["model_name"]
    mode: str        = job["mode"]
    config: dict     = job.get("config", {})
    config_idx: int  = job.get("config_idx", 0)
    gpu: str         = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    if _results_exist(model_name, mode, config_idx):
        logger.info(
            "SKIP (results exist): %s/%s/cfg%03d — marking done.",
            model_name, mode, config_idx,
        )
        return 0

    logger.info(
        "Job start | model=%-12s  mode=%-26s  cfg=%03d  GPU=%s",
        model_name, mode, config_idx, gpu,
    )

    try:
        _run_experiment(model_name, mode, config, config_idx)
    except Exception:
        logger.error("Experiment failed:\n%s", traceback.format_exc())
        return 1

    logger.info(
        "Job done  | model=%-12s  mode=%-26s  cfg=%03d",
        model_name, mode, config_idx,
    )
    return 0


def _run_experiment(model_name: str, mode: str, config: dict, config_idx: int) -> None:
    sys.path.insert(0, str(SYNTHETIC_DIR))

    from src.pipeline_queue import SyntheticExperimentPipeline

    pipeline = SyntheticExperimentPipeline(
        model_name=model_name,
        mode=mode,
        config=config,
        config_idx=config_idx,
        results_dir=str(RESULTS_DIR),
        data_dir=str(SYNTHETIC_DIR / "data"),
    )
    pipeline.run()


if __name__ == "__main__":
    sys.exit(main())
