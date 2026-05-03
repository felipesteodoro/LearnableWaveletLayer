"""
Post-processing: Meta-Labeling on top of DL predictions.

Cross-asset walk-forward meta-labeling with return-weighted labels:

  - Cross-asset: meta-model trains on ALL 25 assets' earlier fold data,
    giving ~25× more training samples than the single-asset version.
    Ticker ID is added as a feature so the model can learn per-asset patterns.

  - Return-weighted labels: meta-training samples are weighted by |ret_test|
    so the meta-model focuses on getting the high-impact predictions right.

  - Walk-forward: for fold i, only folds j < i are used for meta-training
    (across all assets). No future data leaks into the training set.

  - Cold start: fold 0 of every asset has no prior folds → neutral sizing (0.5).

Meta-features (per sample, C = number of output classes):
  p_0 … p_{C-1}  — raw softmax probabilities
  max_p           — confidence = max(softmax)
  margin          — gap between top-2 probabilities
  entropy         — -sum(p * log(p))
  ticker_id       — label-encoded asset identifier

Meta-label: 1 if y_pred == y_true (binary), weighted by |ret_test|.
Fractional sizing: position = direction × meta_proba  ([-1, +1] weighted)

Usage:
    python run_meta_labeling.py
    python run_meta_labeling.py --run-id 2026-05-03
    python run_meta_labeling.py --tickers PETR4.SA VALE3.SA --save

Output:
    Prints comparison table per model/feature_mode.
    Summary CSV: results/<run_id>/meta_summary.csv  (with --save)
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

MODES   = ["raw", "db4", "learned_wavelet", "learned_wavelet_no_warmup"]
MODELS  = ["CNN", "LSTM", "CNN_LSTM", "Transformer", "MLP"]
# Ordered longest-first so prefix matching doesn't confuse CNN with CNN_LSTM
_MODEL_PREFIXES = ["CNN_LSTM", "CNN", "LSTM", "Transformer", "MLP"]

TRANSACTION_COST = 0.0005  # must match pipeline.py


# ---------------------------------------------------------------------------
# Meta-feature engineering
# ---------------------------------------------------------------------------

def _meta_features(y_proba: np.ndarray) -> np.ndarray:
    """(N, C) softmax → (N, C+3) meta-features. Works for C=2 or C=3."""
    p = np.clip(y_proba, 1e-7, 1.0)
    max_p    = p.max(axis=1)
    sorted_p = np.sort(p, axis=1)
    margin   = sorted_p[:, -1] - sorted_p[:, -2]   # gap top-2, C-agnostic
    entropy  = -(p * np.log(p)).sum(axis=1)
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


def _backtest_abstention(
    y_pred: np.ndarray,
    meta_proba: np.ndarray,
    ret_test: np.ndarray,
    tau: float,
) -> dict:
    """Full ±1 position if meta_proba >= tau, else abstain (position=0)."""
    direction = np.where(y_pred == 2, 1.0, np.where(y_pred == 0, -1.0, 0.0))
    pos = np.where(meta_proba >= tau, direction, 0.0)
    return _compute_metrics(pos, ret_test)


def _strat_returns(pos: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """Net strategy returns after transaction costs."""
    cost = TRANSACTION_COST * np.abs(np.diff(pos, prepend=0))
    return pos * ret - cost


def _metrics_from_series(strat: np.ndarray, bh: np.ndarray | None = None) -> dict:
    """Compute metrics from a single (possibly concatenated) return series."""
    cum    = np.cumprod(1 + strat) if len(strat) else np.array([1.0])
    dd     = cum / np.maximum.accumulate(cum) - 1
    sr     = float(strat.mean() / strat.std() * np.sqrt(252)) if strat.std() > 0 else float("nan")
    result = {
        "total_return": float(cum[-1] - 1),
        "sharpe":       sr,
        "max_drawdown": float(dd.min()),
    }
    if bh is not None and len(bh):
        bh_cum = np.cumprod(1 + bh)
        result["bh_return"] = float(bh_cum[-1] - 1)
    return result


def _compute_metrics(pos: np.ndarray, ret: np.ndarray) -> dict:
    strat   = _strat_returns(pos, ret)
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
# Threshold optimisation helpers
# ---------------------------------------------------------------------------

def _optimize_threshold(
    is_results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    tau_grid: np.ndarray,
    min_is_folds: int = 3,
) -> float | None:
    """
    Find τ maximising IS Sharpe across concatenated IS folds.
    Returns None when fewer than min_is_folds are available — caller should
    fall back to median τ to avoid overfitting on tiny samples.
    """
    if len(is_results) < min_is_folds:
        return None
    best_sharpe = float("-inf")
    best_tau    = None
    for tau in tau_grid:
        strats = []
        for y_pred, meta_proba, ret in is_results:
            direction = np.where(y_pred == 2, 1.0, np.where(y_pred == 0, -1.0, 0.0))
            pos = np.where(meta_proba >= tau, direction, 0.0)
            strats.append(_strat_returns(pos, ret))
        all_ret = np.concatenate(strats)
        if all_ret.std() > 0:
            sharpe = float(all_ret.mean() / all_ret.std() * np.sqrt(252))
            if sharpe > best_sharpe:
                best_sharpe, best_tau = sharpe, float(tau)
    return best_tau


def _select_best_per_ticker(
    fold_data: dict,
    min_is_sharpe: float = 0.0,
) -> dict[str, tuple[str, str, str]]:
    """
    Returns {ticker: (fmode, model, mode)} — combo with highest IS Sharpe
    from metrics.json["fin_metrics"]["sharpe"] per asset.

    Uses IS (in-sample) Sharpe to avoid look-ahead bias: we must not rank
    models by OOS performance before deciding which one to meta-label.
    Only combos with IS Sharpe > min_is_sharpe are considered.
    """
    from collections import defaultdict
    candidates: dict[str, list] = defaultdict(list)
    for (fmode, ticker, model, mode), (_folds, exp_dir) in fold_data.items():
        try:
            with open(exp_dir / "metrics.json") as f:
                m = json.load(f)
            is_sharpe = m.get("fin_metrics", {}).get("sharpe")
            if (
                is_sharpe is not None
                and not np.isnan(float(is_sharpe))
                and float(is_sharpe) > min_is_sharpe
            ):
                candidates[ticker].append(((fmode, model, mode), float(is_sharpe)))
        except Exception:
            pass
    return {
        ticker: max(opts, key=lambda x: x[1])[0]
        for ticker, opts in candidates.items()
        if opts
    }


# ---------------------------------------------------------------------------
# Cross-asset walk-forward meta-labeling
# ---------------------------------------------------------------------------

def _load_fold_data(exp_dir: Path) -> list[dict] | None:
    """Load predictions_fold*.npz for one experiment. Returns None if missing."""
    fold_files = sorted(exp_dir.glob("predictions_fold*.npz"))
    if not fold_files:
        return None
    folds = []
    for fp in fold_files:
        data = np.load(fp, allow_pickle=False)
        if "y_proba" not in data.files or "ret_test" not in data.files:
            return None
        folds.append({k: data[k] for k in data.files})
    return folds


def _run_cross_asset_meta(
    target_folds: list[dict],
    peer_data: list[tuple[str, list[dict]]],
    target_ticker: str,
    ticker_to_id: dict[str, int],
) -> dict | None:
    """
    Walk-forward cross-asset meta-labeling with return-weighted labels.

    For each fold i of the target asset:
      - Meta-train: test-fold predictions of ALL assets for folds j < i.
        Using test folds (not val) gives more samples and provides ret_test
        for return-weighting.
      - Meta-label: y_pred == y_true (binary).
      - Sample weight: |ret_test| — focus on high-impact predictions.
      - Meta-features: softmax stats + ticker_id (learns per-asset reliability).
      - Cold start (fold 0): no history → neutral sizing (0.5).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("lightgbm not installed — pip install lightgbm")

    n_folds = len(target_folds)
    meta_preds: list[np.ndarray] = []

    for i in range(n_folds):
        X_parts, y_parts, w_parts = [], [], []

        for ticker, folds in peer_data:
            if len(folds) != n_folds:
                continue
            tid = ticker_to_id.get(ticker, 0)
            for j in range(i):                          # strictly walk-forward
                feat    = _meta_features(folds[j]["y_proba"])
                tid_col = np.full(len(feat), tid, dtype=np.float32)
                X_parts.append(np.column_stack([feat, tid_col]))
                y_parts.append((folds[j]["y_pred"] == folds[j]["y_true"]).astype(int))
                w_parts.append(np.abs(folds[j]["ret_test"]) + 1e-6)

        if not X_parts:
            meta_preds.append(np.full(len(target_folds[i]["y_pred"]), 0.5))
            continue

        X_tr = np.vstack(X_parts)
        y_tr = np.concatenate(y_parts)
        w_tr = np.concatenate(w_parts)

        feat_te = _meta_features(target_folds[i]["y_proba"])
        tid_col = np.full(len(feat_te), ticker_to_id.get(target_ticker, 0), dtype=np.float32)
        X_te    = np.column_stack([feat_te, tid_col])

        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            verbose=-1,
            n_jobs=1,
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        meta_preds.append(clf.predict_proba(X_te)[:, 1])

    primary_strats, meta_strats, bh_rets = [], [], []
    for i in range(n_folds):
        ret = target_folds[i].get("ret_test")
        if ret is None or len(ret) == 0:
            continue
        y_pred   = target_folds[i]["y_pred"]
        direction = np.where(y_pred == 2, 1.0, np.where(y_pred == 0, -1.0, 0.0))
        primary_strats.append(_strat_returns(direction, ret))
        meta_strats.append(_strat_returns(direction * meta_preds[i], ret))
        bh_rets.append(ret)

    if not primary_strats:
        return None

    bh_all = np.concatenate(bh_rets)
    return {
        "primary": _metrics_from_series(np.concatenate(primary_strats), bh_all),
        "meta":    _metrics_from_series(np.concatenate(meta_strats),    bh_all),
    }


def _run_cross_asset_abstention(
    target_folds: list[dict],
    peer_data: list[tuple[str, list[dict]]],
    target_ticker: str,
    ticker_to_id: dict[str, int],
    tau_grid: np.ndarray | None = None,
) -> dict | None:
    """
    Cross-asset walk-forward meta-labeling with per-fold abstention threshold.

    Pass 1 — same as _run_cross_asset_meta: compute meta_preds[i] using
    data from folds j < i across all assets (walk-forward, return-weighted).

    Pass 2 — τ optimisation: for fold i, sweep τ over IS folds 0..i-1 using
    their already-computed meta_preds. τ is never touched by OOS data.
    Cold start (fold 0): τ = median of grid.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("lightgbm not installed — pip install lightgbm")

    if tau_grid is None:
        tau_grid = np.round(np.arange(0.30, 0.80, 0.05), 2)

    n_folds = len(target_folds)

    # ── Pass 1: compute meta_preds for ALL folds ─────────────────────────────
    meta_preds: list[np.ndarray] = []
    for i in range(n_folds):
        X_parts, y_parts, w_parts = [], [], []
        for ticker, folds in peer_data:
            if len(folds) != n_folds:
                continue
            tid = ticker_to_id.get(ticker, 0)
            for j in range(i):
                feat    = _meta_features(folds[j]["y_proba"])
                tid_col = np.full(len(feat), tid, dtype=np.float32)
                X_parts.append(np.column_stack([feat, tid_col]))
                y_parts.append((folds[j]["y_pred"] == folds[j]["y_true"]).astype(int))
                w_parts.append(np.abs(folds[j]["ret_test"]) + 1e-6)

        if not X_parts:
            meta_preds.append(np.full(len(target_folds[i]["y_pred"]), 0.5))
            continue

        X_tr = np.vstack(X_parts)
        y_tr = np.concatenate(y_parts)
        w_tr = np.concatenate(w_parts)

        feat_te = _meta_features(target_folds[i]["y_proba"])
        tid_col = np.full(len(feat_te), ticker_to_id.get(target_ticker, 0), dtype=np.float32)
        X_te    = np.column_stack([feat_te, tid_col])

        clf = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05,
            num_leaves=15, min_child_samples=10,
            verbose=-1, n_jobs=1,
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        meta_preds.append(clf.predict_proba(X_te)[:, 1])

    # ── Pass 2: optimise τ on IS folds (≥3), apply abstention to OOS fold ────
    # τ uses median for folds 0-1 (cold-start) to avoid overfitting tiny samples.
    primary_strats:    list[np.ndarray] = []
    meta_strats:       list[np.ndarray] = []
    abstention_strats: list[np.ndarray] = []
    bh_rets:           list[np.ndarray] = []
    optimal_taus:      list[float] = []
    coverage_list:     list[float] = []

    tau_median = float(np.median(tau_grid))

    for i in range(n_folds):
        ret = target_folds[i].get("ret_test")
        if ret is None or len(ret) == 0:
            continue
        y_pred    = target_folds[i]["y_pred"]
        direction = np.where(y_pred == 2, 1.0, np.where(y_pred == 0, -1.0, 0.0))

        is_results = [
            (target_folds[j]["y_pred"], meta_preds[j], target_folds[j]["ret_test"])
            for j in range(i)
            if target_folds[j].get("ret_test") is not None
            and len(target_folds[j]["ret_test"]) > 0
        ]
        tau_opt = _optimize_threshold(is_results, tau_grid, min_is_folds=3)
        tau     = tau_opt if tau_opt is not None else tau_median

        primary_active    = int((direction != 0).sum())
        abstention_active = int(((meta_preds[i] >= tau) & (direction != 0)).sum())
        coverage_i = (abstention_active / primary_active) if primary_active > 0 else float("nan")

        pos_abstention = np.where(meta_preds[i] >= tau, direction, 0.0)

        optimal_taus.append(tau)
        coverage_list.append(coverage_i)
        primary_strats.append(_strat_returns(direction, ret))
        meta_strats.append(_strat_returns(direction * meta_preds[i], ret))
        abstention_strats.append(_strat_returns(pos_abstention, ret))
        bh_rets.append(ret)

    if not primary_strats:
        return None

    bh_all = np.concatenate(bh_rets)
    return {
        "primary":        _metrics_from_series(np.concatenate(primary_strats),    bh_all),
        "meta":           _metrics_from_series(np.concatenate(meta_strats),       bh_all),
        "abstention":     _metrics_from_series(np.concatenate(abstention_strats), bh_all),
        "mean_tau":       float(np.mean(optimal_taus)) if optimal_taus else float("nan"),
        "trade_coverage": float(np.nanmean(coverage_list)) if coverage_list else float("nan"),
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


def _parse_exp_name(name: str) -> tuple[str, str] | None:
    """Parse '{model}_{mode}' respecting multi-word model names like CNN_LSTM."""
    for model in _MODEL_PREFIXES:
        if name.startswith(model + "_"):
            return model, name[len(model) + 1:]
    return None


def _collect_experiments(
    run_dir: Path,
    tickers: list[str] | None,
    models: list[str] | None,
    modes: list[str] | None,
) -> list[tuple[str, str, str, str, Path]]:
    """Returns list of (feature_mode, ticker, model, mode, exp_dir)."""
    results = []
    # Structure: run_dir / feature_mode / ticker / model_mode /
    for fmode_dir in sorted(run_dir.iterdir()):
        if not fmode_dir.is_dir() or fmode_dir.name in ("logs", "saved_models"):
            continue
        for ticker_dir in sorted(fmode_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            if tickers and ticker not in tickers:
                continue
            for exp_dir in sorted(ticker_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                parsed = _parse_exp_name(exp_dir.name)
                if parsed is None:
                    continue
                model_name, mode = parsed
                if models and model_name not in models:
                    continue
                if modes and mode not in modes:
                    continue
                if not (exp_dir / "metrics.json").exists():
                    continue
                results.append((fmode_dir.name, ticker, model_name, mode, exp_dir))
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
    parser = argparse.ArgumentParser(description="Cross-asset walk-forward meta-labeling")
    parser.add_argument("--run-id", default=None, help="Run ID (default: latest)")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--models",  nargs="*", default=None)
    parser.add_argument("--modes",   nargs="*", default=None)
    parser.add_argument("--save", action="store_true",
                        help="Write meta_metrics.json per experiment and meta_summary.csv")
    parser.add_argument("--metric", default="sharpe",
                        choices=["sharpe", "total_return", "max_drawdown"])
    parser.add_argument("--abstention", action="store_true",
                        help="Best-per-ticker abstention: select best OOS model per asset, "
                             "then run cross-asset meta with walk-forward τ optimisation")
    parser.add_argument("--min-sharpe", type=float, default=0.0,
                        help="Minimum IS Sharpe for a model to be considered in --abstention (default: 0.0)")
    args = parser.parse_args()

    run_dir = _find_run_dir(args.run_id)
    logger.info("Run dir: %s", run_dir)

    # Collect ALL experiments (no filter yet) to build the full cross-asset pool.
    all_experiments = _collect_experiments(run_dir, tickers=None, models=None, modes=None)
    logger.info("Total experiments in run: %d", len(all_experiments))

    # ── Step 1: load fold data for every experiment ──────────────────────────
    logger.info("Loading fold data...")
    fold_data: dict[tuple, tuple[list[dict], Path]] = {}
    for fmode, ticker, model, mode, exp_dir in all_experiments:
        folds = _load_fold_data(exp_dir)
        if folds is not None:
            fold_data[(fmode, ticker, model, mode)] = (folds, exp_dir)
    logger.info("Loaded %d / %d experiments with fold data.", len(fold_data), len(all_experiments))

    # ── Step 2: build ticker encoder from full pool ──────────────────────────
    all_tickers  = sorted({k[1] for k in fold_data})
    ticker_to_id = {t: i for i, t in enumerate(all_tickers)}
    logger.info("Tickers in pool: %d", len(all_tickers))

    # ═══════════════════════════════════════════════════════════════════════
    # ABSTENTION MODE: best model per ticker + walk-forward τ optimisation
    # ═══════════════════════════════════════════════════════════════════════
    if args.abstention:
        best_map = _select_best_per_ticker(fold_data, min_is_sharpe=args.min_sharpe)
        n_all_tickers = len({k[1] for k in fold_data})
        logger.info(
            "Tickers with IS Sharpe > %.2f: %d / %d (excluded: %d)",
            args.min_sharpe, len(best_map), n_all_tickers, n_all_tickers - len(best_map),
        )
        if not best_map:
            print(f"No tickers with IS Sharpe > {args.min_sharpe:.2f}. "
                  "Lower --min-sharpe or check that experiments finished.")
            return

        # Apply optional ticker filter
        if args.tickers:
            best_map = {t: v for t, v in best_map.items() if t in args.tickers}

        tau_grid = np.round(np.arange(0.30, 0.80, 0.05), 2)
        rows = []
        total = len(best_map)

        for idx, (ticker, (fmode, model, mode)) in enumerate(sorted(best_map.items()), 1):
            key = (fmode, ticker, model, mode)
            if key not in fold_data:
                logger.warning("[%d/%d] %s — no fold data, skipping", idx, total, ticker)
                continue

            logger.info("[%d/%d] %s — best: %s/%s_%s", idx, total, ticker, fmode, model, mode)
            target_folds, exp_dir = fold_data[key]

            peer_data = [
                (k[1], v[0])
                for k, v in fold_data.items()
                if k[0] == fmode and k[2] == model and k[3] == mode
            ]

            try:
                result = _run_cross_asset_abstention(
                    target_folds, peer_data, ticker, ticker_to_id, tau_grid
                )
            except Exception as exc:
                logger.warning("  SKIP — %s", exc)
                continue

            if result is None:
                continue

            rows.append({
                "ticker":         ticker,
                "feature_mode":   fmode,
                "model":          model,
                "mode":           mode,
                "primary_sharpe": result["primary"]["sharpe"],
                "meta_sharpe":    result["meta"]["sharpe"],
                "abstention_sharpe": result["abstention"]["sharpe"],
                "mean_tau":       result["mean_tau"],
                "trade_coverage": result["trade_coverage"],
                "primary_return": result["primary"]["total_return"],
                "abstention_return": result["abstention"]["total_return"],
            })

            if args.save:
                with open(exp_dir / "abstention_metrics.json", "w") as f:
                    json.dump(result, f, indent=2)

        if not rows:
            print("No experiments produced results.")
            return

        df = pd.DataFrame(rows).sort_values("abstention_sharpe", ascending=False)

        print(f"\n{'='*90}")
        print(f"  Abstention Meta-Labeling — Best Model per Ticker  (τ grid: {tau_grid[0]:.2f}–{tau_grid[-1]:.2f})")
        print(f"{'='*90}")
        print(
            f"  {'ticker':<12}  {'combo':<30}  {'prim_SR':>8}  {'frac_SR':>8}"
            f"  {'abst_SR':>8}  {'Δ(abst-prim)':>13}  {'τ':>6}  {'cov%':>6}"
        )
        print(f"  {'-'*12}  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*13}  {'-'*6}  {'-'*6}")
        for _, row in df.iterrows():
            combo = f"{row['feature_mode']}/{row['model']}_{row['mode']}"
            delta = row["abstention_sharpe"] - row["primary_sharpe"]
            sign  = "+" if delta >= 0 else ""
            cov   = f"{row['trade_coverage']*100:.0f}%" if not np.isnan(row["trade_coverage"]) else "—"
            tau_s = f"{row['mean_tau']:.2f}" if not np.isnan(row["mean_tau"]) else "—"
            print(
                f"  {row['ticker']:<12}  {combo:<30}  {row['primary_sharpe']:>8.3f}"
                f"  {row['meta_sharpe']:>8.3f}  {row['abstention_sharpe']:>8.3f}"
                f"  {sign}{delta:>12.3f}  {tau_s:>6}  {cov:>6}"
            )

        # Aggregate summary
        valid = df.dropna(subset=["primary_sharpe", "abstention_sharpe"])
        wins  = (valid["abstention_sharpe"] > valid["primary_sharpe"]).sum()
        print(f"\n  Abstention wins vs primary: {wins}/{len(valid)}")
        print(
            f"  Mean primary SR:    {valid['primary_sharpe'].mean():.4f}"
            f"  →  abstention SR: {valid['abstention_sharpe'].mean():.4f}"
            f"  (Δ {valid['abstention_sharpe'].mean()-valid['primary_sharpe'].mean():+.4f})"
        )
        print(f"  Mean fractional SR: {valid['meta_sharpe'].mean():.4f}")
        print(f"{'='*90}\n")

        if args.save:
            out_path = run_dir / "abstention_summary.csv"
            df.to_csv(out_path, index=False)
            logger.info("Saved: %s", out_path)
        return

    # ═══════════════════════════════════════════════════════════════════════
    # STANDARD MODE: fractional cross-asset meta-labeling
    # ═══════════════════════════════════════════════════════════════════════

    # ── Step 3: apply optional filters to select target experiments ──────────
    target_keys = [
        k for k in fold_data
        if (args.tickers is None or k[1] in args.tickers)
        and (args.models  is None or k[2] in args.models)
        and (args.modes   is None or k[3] in args.modes)
    ]
    logger.info("Target experiments (after filters): %d", len(target_keys))

    # ── Step 4: run cross-asset meta-labeling for each target ────────────────
    rows = []
    total = len(target_keys)
    for idx, (fmode, ticker, model, mode) in enumerate(target_keys, 1):
        logger.info("[%d/%d] %s / %s / %s_%s", idx, total, fmode, ticker, model, mode)

        target_folds, exp_dir = fold_data[(fmode, ticker, model, mode)]

        peer_data = [
            (k[1], v[0])
            for k, v in fold_data.items()
            if k[0] == fmode and k[2] == model and k[3] == mode
        ]

        try:
            result = _run_cross_asset_meta(
                target_folds, peer_data, ticker, ticker_to_id
            )
        except Exception as exc:
            logger.warning("  SKIP — %s", exc)
            continue

        if result is None:
            continue

        row = {"feature_mode": fmode, "ticker": ticker, "model": model, "mode": mode}
        for key, val in result["primary"].items():
            row[f"primary_{key}"] = val
        for key, val in result["meta"].items():
            row[f"meta_{key}"] = val
        rows.append(row)

        if args.save:
            with open(exp_dir / "meta_metrics.json", "w") as f:
                json.dump(result, f, indent=2)

    if not rows:
        print("No experiments produced results.")
        return

    summary = pd.DataFrame(rows)

    for fmode in summary["feature_mode"].unique():
        for model_name in MODELS:
            sub = summary[(summary["feature_mode"] == fmode) & (summary["model"] == model_name)]
            if not sub.empty:
                print(f"\n[feature_mode={fmode}]")
                _print_comparison_table(
                    summary[summary["feature_mode"] == fmode],
                    model_name,
                    metric=args.metric,
                )

    if args.save:
        out_path = run_dir / "meta_summary.csv"
        summary.to_csv(out_path, index=False)
        logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
