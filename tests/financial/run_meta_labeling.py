"""
Post-processing: Meta-Labeling on top of DL predictions.

For each completed experiment (ticker / model / mode), applies leave-one-fold-out
meta-labeling using LightGBM and produces a 3×2 comparison table:

    mode         │ sem_meta  │ com_meta
    ─────────────┼───────────┼──────────
    raw          │  ...      │  ...
    db4          │  ...      │  ...
    learned_wavelet │ ...   │  ...

Meta-features (per sample, derived from primary model's softmax output):
  p_sell, p_hold, p_buy   — raw class probabilities
  max_p                   — confidence = max(softmax)
  margin                  — gap between top-2 probabilities
  entropy                 — -sum(p * log(p))

Meta-label: 1 if primary prediction == true label, else 0.
Fractional sizing: position = direction × meta_proba  ([-1, +1] weighted)

Usage:
    # Latest run (auto-detected)
    python run_meta_labeling.py

    # Specific run
    python run_meta_labeling.py --run-id 2026-04-27_151403

    # Specific run + subset
    python run_meta_labeling.py --run-id 2026-04-27_151403 --tickers PETR4.SA VALE3.SA

    # Write per-experiment meta_metrics.json alongside metrics.json
    python run_meta_labeling.py --save

Output:
    Prints 3×2 table per model; optionally writes meta_metrics.json.
    Summary CSV: results/<run_id>/meta_summary.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_BASE        = Path(__file__).parent
_RESULTS_BASE = _BASE / "results"

MODES   = ["raw", "db4", "learned_wavelet"]
MODELS  = ["CNN", "LSTM", "CNN_LSTM", "Transformer"]

TRANSACTION_COST = 0.001  # must match pipeline.py


# ---------------------------------------------------------------------------
# Meta-feature engineering
# ---------------------------------------------------------------------------

def _meta_features(y_proba: np.ndarray) -> np.ndarray:
    """(N, 3) softmax → (N, 6) meta-features."""
    p = np.clip(y_proba, 1e-7, 1.0)
    max_p   = p.max(axis=1)
    sorted_p = np.sort(p, axis=1)
    margin  = sorted_p[:, 2] - sorted_p[:, 1]
    entropy = -(p * np.log(p)).sum(axis=1)
    return np.column_stack([p, max_p, margin, entropy])


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------

def _backtest_primary(y_pred: np.ndarray, ret_test: np.ndarray) -> dict:
    """Position ∈ {-1, 0, +1} from class label (0=sell, 1=hold, 2=buy)."""
    pos = np.where(y_pred == 2, 1.0, np.where(y_pred == 0, -1.0, 0.0))
    return _compute_metrics(pos, ret_test)


def _backtest_meta(
    y_pred: np.ndarray,
    meta_proba: np.ndarray,
    ret_test: np.ndarray,
) -> dict:
    """Fractional position: direction × meta_proba (confidence-weighted)."""
    direction = np.where(y_pred == 2, 1.0, np.where(y_pred == 0, -1.0, 0.0))
    pos = direction * meta_proba
    return _compute_metrics(pos, ret_test)


def _compute_metrics(pos: np.ndarray, ret: np.ndarray) -> dict:
    cost    = TRANSACTION_COST * np.abs(np.diff(pos, prepend=0))
    strat   = pos * ret - cost
    cum     = np.cumprod(1 + strat)
    bh_cum  = np.cumprod(1 + ret)

    total_ret = float(cum[-1] - 1) if len(cum) else float("nan")
    bh_ret    = float(bh_cum[-1] - 1) if len(bh_cum) else float("nan")

    dd        = cum / np.maximum.accumulate(cum) - 1
    max_dd    = float(dd.min()) if len(dd) else float("nan")

    sr        = float(strat.mean() / strat.std() * np.sqrt(252)) if strat.std() > 0 else float("nan")
    return {
        "total_return": total_ret,
        "bh_return": bh_ret,
        "sharpe": sr,
        "max_drawdown": max_dd,
    }


# ---------------------------------------------------------------------------
# Per-experiment meta-labeling
# ---------------------------------------------------------------------------

def _run_meta_experiment(exp_dir: Path) -> Optional[dict]:
    """
    Load all predictions_fold*.npz for one experiment and run leave-one-fold-out
    meta-labeling. Returns dict with primary and meta metrics, or None if no data.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("lightgbm not installed — pip install lightgbm")

    fold_files = sorted(exp_dir.glob("predictions_fold*.npz"))
    if not fold_files:
        return None

    folds: list[dict] = []
    for fp in fold_files:
        data = np.load(fp, allow_pickle=False)
        # Skip folds without val data (experiments before meta-labeling support)
        if "X_val" not in data.files:
            return None
        folds.append({k: data[k] for k in data.files})

    n_folds = len(folds)
    meta_preds: list[np.ndarray] = [None] * n_folds  # type: ignore[list-item]

    for i in range(n_folds):
        # Build training set from all folds except i
        X_meta_tr_parts = []
        y_meta_tr_parts = []
        for j in range(n_folds):
            if j == i:
                continue
            feat = _meta_features(folds[j]["y_proba_val"])
            lbl  = (folds[j]["y_val"] == np.argmax(folds[j]["y_proba_val"], axis=1)).astype(int)
            X_meta_tr_parts.append(feat)
            y_meta_tr_parts.append(lbl)

        if not X_meta_tr_parts:
            meta_preds[i] = np.full(len(folds[i]["y_pred"]), 0.5)
            continue

        X_tr = np.vstack(X_meta_tr_parts)
        y_tr = np.concatenate(y_meta_tr_parts)

        X_te_meta = _meta_features(folds[i]["y_proba"])

        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            verbose=-1,
            n_jobs=1,
        )
        clf.fit(X_tr, y_tr)
        meta_preds[i] = clf.predict_proba(X_te_meta)[:, 1]  # P(correct)

    # Aggregate metrics across folds
    primary_metrics_per_fold = []
    meta_metrics_per_fold    = []

    for i in range(n_folds):
        ret = folds[i].get("ret_test")
        if ret is None or len(ret) == 0:
            continue
        primary_metrics_per_fold.append(_backtest_primary(folds[i]["y_pred"], ret))
        meta_metrics_per_fold.append(_backtest_meta(folds[i]["y_pred"], meta_preds[i], ret))

    if not primary_metrics_per_fold:
        return None

    def _mean(lst: list[dict], key: str) -> float:
        vals = [d[key] for d in lst if not np.isnan(d[key])]
        return float(np.mean(vals)) if vals else float("nan")

    keys = ["total_return", "bh_return", "sharpe", "max_drawdown"]
    return {
        "primary": {k: _mean(primary_metrics_per_fold, k) for k in keys},
        "meta":    {k: _mean(meta_metrics_per_fold, k) for k in keys},
    }


# ---------------------------------------------------------------------------
# Discovery and orchestration
# ---------------------------------------------------------------------------

def _find_run_dir(run_id: str | None) -> Path:
    if run_id:
        d = _RESULTS_BASE / run_id
        if not d.exists():
            raise FileNotFoundError(f"Run not found: {d}")
        return d
    dated = sorted(_RESULTS_BASE.glob("????-??-??"), reverse=True)
    for folder in dated:
        if (folder / "queue_status.json").exists():
            return folder
    raise FileNotFoundError("No completed run found in results/")


def _collect_experiments(
    run_dir: Path,
    tickers: list[str] | None,
    models: list[str] | None,
    modes: list[str] | None,
) -> list[tuple[str, str, str, Path]]:
    """Returns list of (ticker, model, mode, exp_dir) for completed experiments."""
    results = []
    for ticker_dir in sorted(run_dir.iterdir()):
        if not ticker_dir.is_dir() or ticker_dir.name.startswith("."):
            continue
        ticker = ticker_dir.name
        if tickers and ticker not in tickers:
            continue
        for exp_dir in sorted(ticker_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            # exp_dir.name == "{model}_{mode}"
            parts = exp_dir.name.split("_", 1)
            if len(parts) != 2:
                continue
            model_name, mode = parts[0], parts[1]
            if models and model_name not in models:
                continue
            if modes and mode not in modes:
                continue
            if not (exp_dir / "metrics.json").exists():
                continue
            results.append((ticker, model_name, mode, exp_dir))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_comparison_table(
    summary: pd.DataFrame,
    model_name: str,
    metric: str = "sharpe",
):
    sub = summary[summary["model"] == model_name]
    if sub.empty:
        return

    print(f"\n{'='*62}")
    print(f"  Model: {model_name}  |  Metric: {metric}")
    print(f"{'='*62}")
    header = f"  {'mode':<20}  {'sem_meta':>10}  {'com_meta':>10}  {'delta':>10}"
    print(header)
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
    for mode in MODES:
        row = sub[sub["mode"] == mode]
        if row.empty:
            print(f"  {mode:<20}  {'—':>10}  {'—':>10}  {'—':>10}")
            continue
        sem  = row[f"primary_{metric}"].mean()
        com  = row[f"meta_{metric}"].mean()
        delta = com - sem
        sign = "+" if delta >= 0 else ""
        print(
            f"  {mode:<20}  {sem:>10.4f}  {com:>10.4f}  {sign}{delta:>9.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Meta-labeling post-processor")
    parser.add_argument("--run-id", default=None, help="Run ID (default: latest)")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--models",  nargs="*", default=None)
    parser.add_argument("--modes",   nargs="*", default=None)
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write meta_metrics.json per experiment and meta_summary.csv",
    )
    parser.add_argument(
        "--metric",
        default="sharpe",
        choices=["sharpe", "total_return", "max_drawdown"],
        help="Primary metric for comparison table (default: sharpe)",
    )
    args = parser.parse_args()

    run_dir = _find_run_dir(args.run_id)
    logger.info("Run dir: %s", run_dir)

    experiments = _collect_experiments(
        run_dir,
        tickers=args.tickers,
        models=args.models,
        modes=args.modes,
    )
    logger.info("Found %d completed experiments", len(experiments))

    rows = []
    for i, (ticker, model_name, mode, exp_dir) in enumerate(experiments, 1):
        logger.info("[%d/%d] %s / %s_%s", i, len(experiments), ticker, model_name, mode)
        try:
            result = _run_meta_experiment(exp_dir)
        except Exception as exc:
            logger.warning("  SKIP — %s", exc)
            continue
        if result is None:
            logger.debug("  No val data, skipping.")
            continue

        row = {
            "ticker": ticker,
            "model": model_name,
            "mode": mode,
        }
        for key, val in result["primary"].items():
            row[f"primary_{key}"] = val
        for key, val in result["meta"].items():
            row[f"meta_{key}"] = val
        rows.append(row)

        if args.save:
            with open(exp_dir / "meta_metrics.json", "w") as f:
                json.dump(result, f, indent=2)

    if not rows:
        print("No experiments with meta-labeling data found.")
        print("Re-run the queue to generate predictions with X_val saved.")
        return

    summary = pd.DataFrame(rows)

    for model_name in MODELS:
        if model_name in summary["model"].values:
            _print_comparison_table(summary, model_name, metric=args.metric)

    if args.save:
        out_path = run_dir / "meta_summary.csv"
        summary.to_csv(out_path, index=False)
        logger.info("Saved summary: %s", out_path)


if __name__ == "__main__":
    main()
