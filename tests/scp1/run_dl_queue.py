"""
Launch all DL SelfRegulationSCP1 experiments across multiple GPUs.

Usage:
    cd tests/scp1
    python run_dl_queue.py --fresh       # nova run
    python run_dl_queue.py               # auto-resume
    python run_dl_queue.py --run-id 2026-05-30

Environment variables:
    N_GPUS, GPU_IDS, RETRY_WAIT, DL_MODELS, MODES, EPOCHS_OVERRIDE, MAX_GRID_CONFIGS
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "config"))

from gpu_queue import GPUJobQueueManager, ExperimentJob  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

_BASE = Path(__file__).parent

ALL_MODELS = ["CNN", "LSTM", "CNN_LSTM", "Transformer", "MLP"]
ALL_MODES  = ["raw", "db4", "learned_wavelet_no_warmup", "learned_wavelet"]


def _from_env(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key, "").strip()
    return [v.strip() for v in raw.split(",") if v.strip()] if raw else default


def _gpu_ids() -> list[int]:
    raw = os.environ.get("GPU_IDS", "").strip()
    if raw:
        return [int(x) for x in raw.split(",") if x.strip()]
    return list(range(int(os.environ.get("N_GPUS", "7"))))


def _latest_run_id() -> str | None:
    results = _BASE / "results"
    if not results.exists():
        return None
    dated = sorted(results.glob("????-??-??*"), reverse=True)
    for folder in dated:
        if (folder / "queue_status.json").exists():
            return folder.name
    return None


def _new_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _build_jobs(models: list[str], modes: list[str]) -> list[ExperimentJob]:
    from experiment_config import (
        DL_MODELS_CONFIG, DL_TRAINING_CONFIG,
        LEARNED_WAVELET_MODELS_CONFIG, LEARNED_WAVELET_CONFIG,
        generate_dl_grid, generate_learned_wavelet_grid,
    )

    jobs: list[ExperimentJob] = []

    for model in models:
        for mode in modes:
            if mode in ("raw", "db4"):
                base_cfg = {**DL_MODELS_CONFIG.get(model, {}), **DL_TRAINING_CONFIG}
                if mode == "db4":
                    base_cfg.update({k: LEARNED_WAVELET_CONFIG[k]
                                     for k in ("levels", "align") if k in LEARNED_WAVELET_CONFIG})
                grid = generate_dl_grid(model)
            else:
                base_cfg = {
                    **LEARNED_WAVELET_MODELS_CONFIG.get(model, {}),
                    **DL_TRAINING_CONFIG,
                    **LEARNED_WAVELET_CONFIG,
                }
                if mode == "learned_wavelet":
                    base_cfg["warm_start_db4"] = True
                grid = generate_learned_wavelet_grid(model)

            for config_idx, variation in enumerate(grid):
                config = {**base_cfg, **variation}
                jobs.append(ExperimentJob(
                    model_name=model,
                    mode=mode,
                    config=config,
                    config_idx=config_idx,
                ))

    return jobs


def main():
    parser = argparse.ArgumentParser(description="SelfRegulationSCP1 GPU DL experiment queue")
    parser.add_argument("--fresh", action="store_true", help="New results folder")
    parser.add_argument("--run-id", default=None, metavar="YYYY-MM-DD")
    args = parser.parse_args()

    models     = _from_env("DL_MODELS", ALL_MODELS)
    modes      = _from_env("MODES", ALL_MODES)
    gpu_ids    = _gpu_ids()
    retry_wait = int(os.environ.get("RETRY_WAIT", "60"))

    all_jobs = _build_jobs(models, modes)

    if args.fresh:
        run_id = _new_run_id()
        results_dir = f"results/{run_id}"
        mode_label = f"NOVA RUN → {run_id}"
    elif args.run_id:
        run_id = args.run_id
        results_dir = f"results/{run_id}"
        mode_label = f"RESUMINDO RUN → {run_id}"
    else:
        run_id = _latest_run_id() or _new_run_id()
        results_dir = f"results/{run_id}"
        mode_label = f"AUTO-RESUME → {run_id}"

    print(
        f"\n{'='*62}\n"
        f"  SelfRegulationSCP1 GPU Job Queue  |  {mode_label}\n"
        f"{'='*62}\n"
        f"  Results dir : results/{run_id}/\n"
        f"  Total jobs  : {len(all_jobs)}\n"
        f"  GPUs        : {gpu_ids}\n"
        f"  Models      : {models}\n"
        f"  Modes       : {modes}\n"
        f"  Retry wait  : {retry_wait}s  |  max retries: 2\n"
        f"{'='*62}\n"
    )

    if args.fresh:
        manager = GPUJobQueueManager(gpu_ids=gpu_ids, retry_wait=retry_wait, results_dir=results_dir)
        manager.add_many(all_jobs)
    else:
        manager = GPUJobQueueManager.resume_or_create(
            all_jobs=all_jobs, gpu_ids=gpu_ids, retry_wait=retry_wait, results_dir=results_dir,
        )

    manager.run()
    print(f"\nAll experiments completed.  Results: results/{run_id}/")


if __name__ == "__main__":
    main()
