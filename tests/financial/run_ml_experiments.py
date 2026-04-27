#!/usr/bin/env python3
"""
Canonical ML experiments — Walk-Forward OOS + IS grid search.

Methodology (all fixes applied)
--------------------------------
Fix 1 — Features are causal (rolling lookback only); no future leakage.
         RobustScaler is fit exclusively on each window's IS.
Fix 3 — t1 estimated with BDay(time_horizon) instead of calendar days.
Fix 4 — Real t1 (actual barrier-exit date) loaded from labels parquet when
         available (requires re-running 00_data_preparation.ipynb); falls back
         to BDay estimate.
Fix 5 — Walk-forward OOS: total OOS is divided into `n_oos_windows` temporal
         windows. For each window the IS grows (expanding window). Grid search
         via PurgedKFold is re-run on each IS. Final metrics are the average
         across all OOS windows.

Results: results/<ticker>/<model>_ml/metrics.json
Existing results are skipped automatically.

Usage: python run_ml_experiments.py
"""
import sys, warnings, json, time, traceback
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent.parent))

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
os.environ.setdefault("ARROW_PRE_0_15_IPC_FORMAT", "0")

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config.experiment_config import (
    TICKERS, ML_MODELS_CONFIG, ML_N_JOBS_OUTER, VALIDATION_CONFIG, BACKTEST_CONFIG,
    LABELING_CONFIG, ML_N_JOBS_MODEL, ML_PARAM_GRID
)
from src.data_loader import load_processed, load_labels, load_t1
from src.evaluation import ClassificationEvaluator, FinancialMetrics
from src.backtest import simulate_strategy, buy_and_hold_returns

RESULTS_DIR   = BASE / "results"
N_OOS_WINDOWS = VALIDATION_CONFIG.get("n_oos_windows", 2)   # walk-forward splits



# ── Model Factory ─────────────────────────────────────────────────────────
def build_model(model_name: str, overrides: dict | None = None):
    overrides = overrides or {}
    base = dict(ML_MODELS_CONFIG.get(model_name, {}))
    base.update(overrides)

    if model_name == "RandomForest":
        return RandomForestClassifier(**base)
    if model_name == "XGBoost":
        base.pop("use_label_encoder", None)
        base.pop("eval_metric", None)
        return XGBClassifier(**base, verbosity=0)
    if model_name == "LightGBM":
        return LGBMClassifier(**base)
    if model_name == "CatBoost":
        return CatBoostClassifier(**base)
    if model_name == "Stacking":
        n = overrides.get("base_n_estimators", 100)
        estimators = [
            ("rf",  RandomForestClassifier(n_estimators=n, n_jobs=ML_N_JOBS_MODEL, random_state=42)),
            ("xgb", XGBClassifier(n_estimators=n, nthread=ML_N_JOBS_MODEL, verbosity=0, random_state=42)),
            ("lgb", LGBMClassifier(n_estimators=n, num_threads=ML_N_JOBS_MODEL, verbose=-1, random_state=42)),
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            n_jobs=ML_N_JOBS_MODEL, passthrough=False,
        )
    raise ValueError(f"Unknown model: {model_name}")


# ── Helper: build t1 series (Fix 3+4) ────────────────────────────────────
def _build_t1(dates: pd.DatetimeIndex, t1_real: pd.Series | None,
              time_horizon: int) -> pd.Series:
    """
    Return event-end dates for PurgedKFold purge.
    Uses real t1 from Triple Barrier when available (Fix 4),
    otherwise estimates using business days (Fix 3).
    """
    if t1_real is not None:
        t1 = t1_real.reindex(dates)
        # Fill any gaps with BDay estimate
        missing = t1.isna()
        if missing.any():
            bday_est = pd.Series(
                [d + pd.tseries.offsets.BDay(time_horizon) for d in dates[missing]],
                index=dates[missing],
            )
            t1[missing] = bday_est
        return t1
    # Full BDay fallback (Fix 3)
    return pd.Series(
        [d + pd.tseries.offsets.BDay(time_horizon) for d in dates],
        index=dates,
    )


# ── Helper: grid search on one IS block ──────────────────────────────────
def _grid_search_is(X_is, y_is, dates_is, t1_is, model_name, pkf_cfg):
    """Run PurgedKFold grid search on IS block; return (best_overrides, best_score, grid_results)."""
    from utils.validation import PurgedKFold

    pct_embargo = pkf_cfg["embargo_days"] / len(y_is)
    pkf = PurgedKFold(
        n_splits=pkf_cfg["n_folds"],
        t1=t1_is,
        pct_embargo=pct_embargo,
    )
    X_is_df = pd.DataFrame(X_is, index=dates_is)

    param_grid     = ML_PARAM_GRID.get(model_name, [{}])
    best_score     = -np.inf
    best_overrides = param_grid[0]
    grid_results   = []

    for overrides in param_grid:
        fold_scores = []
        for train_val_idx, val_idx in pkf.split(X_is_df, y_is):
            n_val_inner = max(1, int(len(train_val_idx) * 0.15))
            train_idx   = train_val_idx[:-n_val_inner]

            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X_is[train_idx])
            X_va = scaler.transform(X_is[val_idx])

            model = build_model(model_name, overrides)
            model.fit(X_tr, y_is[train_idx])
            y_pred_va = np.asarray(model.predict(X_va)).ravel()

            fold_scores.append(
                f1_score(y_is[val_idx], y_pred_va, average="macro", zero_division=0)
            )

        mean_score = float(np.mean(fold_scores))
        grid_results.append({"overrides": overrides, "cv_f1_macro": mean_score})
        if mean_score > best_score:
            best_score     = mean_score
            best_overrides = overrides

    return best_overrides, best_score, grid_results


# ── Single Job Runner ─────────────────────────────────────────────────────
def run_ml_job(ticker: str, model_name: str) -> dict:
    result_path = RESULTS_DIR / ticker / f"{model_name}_ml" / "metrics.json"
    if result_path.exists():
        return {"ticker": ticker, "model_name": model_name, "status": "skipped"}

    try:
        # ── 1. Load data ─────────────────────────────────────────────────
        features = load_processed(ticker)
        labels   = load_labels(ticker)
        t1_real  = load_t1(ticker)          # None if not yet generated (Fix 4)

        common   = features.index.intersection(labels.index)
        features = features.loc[common].dropna()
        labels   = labels.loc[features.index]

        X     = features.values.astype(np.float32)
        y     = labels.values
        dates = features.index

        # Synthetic price series for backtest (compound log-returns)
        if "log_return_1" in features.columns:
            lr = features["log_return_1"].values
            price_series = pd.Series(np.exp(np.cumsum(lr)), index=dates)
        else:
            price_series = pd.Series(np.ones(len(dates)), index=dates)

        time_horizon = LABELING_CONFIG.get("time_horizon", 10)
        t1_all = _build_t1(dates, t1_real, time_horizon)    # Fix 3 + 4

        # ── 2. Walk-forward OOS windows (Fix 5) ──────────────────────────
        # OOS total = last test_years years, divided into N_OOS_WINDOWS windows.
        # Each window evaluates on a disjoint OOS slice while IS grows.
        #
        #   Window 1: IS = [start → oos_start],     OOS = [oos_start → wp1]
        #   Window 2: IS = [start → wp1],            OOS = [wp1 → wp2]
        #   ...
        #   Window N: IS = [start → wp(N-1)],        OOS = [wp(N-1) → end]
        test_years = VALIDATION_CONFIG.get("test_years", 6)
        last_date  = dates[-1]
        oos_start  = last_date - pd.DateOffset(years=test_years)

        # Evenly-spaced window boundaries within the OOS period
        window_boundaries = [
            oos_start + pd.DateOffset(years=round(test_years * k / N_OOS_WINDOWS, 2))
            for k in range(N_OOS_WINDOWS + 1)
        ]

        pkf_cfg = {
            "n_folds":      VALIDATION_CONFIG.get("n_folds", 5),
            "embargo_days": VALIDATION_CONFIG.get("embargo_days", 10),
        }

        wf_ml_results  = []
        wf_fin_results = []
        wf_info        = []
        all_y_true, all_y_pred = [], []

        for w in range(N_OOS_WINDOWS):
            win_is_end  = window_boundaries[w]
            win_oos_end = window_boundaries[w + 1]

            is_mask  = dates <= win_is_end
            oos_mask = (dates > win_is_end) & (dates <= win_oos_end)

            if is_mask.sum() < 100 or oos_mask.sum() < 10:
                continue

            X_is,   y_is,   dates_is = X[is_mask],  y[is_mask],  dates[is_mask]
            X_oos,  y_oos            = X[oos_mask], y[oos_mask]
            prices_oos               = price_series[oos_mask].reset_index(drop=True)
            t1_is = t1_all[is_mask]

            # ── 3. Grid search on this IS window ─────────────────────────
            best_overrides, best_score, grid_results = _grid_search_is(
                X_is, y_is, dates_is, t1_is, model_name, pkf_cfg
            )

            # ── 4. Retrain on full IS → evaluate OOS ─────────────────────
            final_scaler = RobustScaler()
            X_is_sc  = final_scaler.fit_transform(X_is)
            X_oos_sc = final_scaler.transform(X_oos)

            final_model = build_model(model_name, best_overrides)
            final_model.fit(X_is_sc, y_is)

            y_pred_oos = np.asarray(final_model.predict(X_oos_sc)).ravel()

            oos_ml    = ClassificationEvaluator.evaluate(y_oos, y_pred_oos)
            strat_oos = simulate_strategy(
                y_pred_oos, prices_oos,
                transaction_cost=BACKTEST_CONFIG["transaction_cost"],
                allow_short=BACKTEST_CONFIG["allow_short"],
            )
            bh_oos = buy_and_hold_returns(prices_oos)
            oos_fin = FinancialMetrics.compute(strat_oos, bh_oos)

            wf_ml_results.append(oos_ml)
            wf_fin_results.append(oos_fin)
            wf_info.append({
                "window":    w + 1,
                "is_end":    str(win_is_end.date()),
                "oos_start": str(dates[oos_mask][0].date()),
                "oos_end":   str(dates[oos_mask][-1].date()),
                "n_is":      int(is_mask.sum()),
                "n_oos":     int(oos_mask.sum()),
                "best_params":   best_overrides,
                "cv_f1_macro":   float(best_score),
                "grid_search":   grid_results,
                "oos_f1_macro":  float(oos_ml.get("f1_macro", float("nan"))),
            })
            all_y_true.append(y_oos)
            all_y_pred.append(y_pred_oos)

        if not wf_ml_results:
            raise ValueError("No valid walk-forward windows produced.")

        # ── 5. Aggregate across windows ───────────────────────────────────
        def _safe(v):
            try:
                return float(v) if np.isfinite(float(v)) else None
            except Exception:
                return None

        agg_ml  = {k: _safe(np.mean([w[k] for w in wf_ml_results]))
                   for k in wf_ml_results[0]}
        agg_fin = {k: _safe(np.mean([w[k] for w in wf_fin_results
                                      if k in w and w[k] is not None]))
                   for k in wf_fin_results[0]}

        # ── 6. Persist ───────────────────────────────────────────────────
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ml_metrics":  agg_ml,
            "fin_metrics": agg_fin,
            "walk_forward_windows": wf_info,
            "split_info": {
                "oos_start":       str(oos_start.date()),
                "test_years":      test_years,
                "n_oos_windows":   N_OOS_WINDOWS,
                "t1_source":       "real" if t1_real is not None else "bday_estimate",
            },
        }
        result_path.write_text(json.dumps(payload, indent=2))
        np.savez(
            result_path.parent / "predictions_oos.npz",
            y_true=np.concatenate(all_y_true),
            y_pred=np.concatenate(all_y_pred),
        )

        return {
            "ticker":      ticker,
            "model_name":  model_name,
            "status":      "done",
            "f1_macro":    agg_ml.get("f1_macro"),
            "cv_f1_macro": float(np.mean([w["cv_f1_macro"] for w in wf_info])),
            "n_windows":   len(wf_info),
        }

    except Exception:
        return {
            "ticker":     ticker,
            "model_name": model_name,
            "status":     "error",
            "error":      traceback.format_exc()[-600:],
        }


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ML_MODELS_TO_RUN = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "Stacking"]
    combos = [(t, m) for t in TICKERS for m in ML_MODELS_TO_RUN]
    n_jobs = ML_N_JOBS_OUTER

    print(f"Total jobs    : {len(combos)}")
    print(f"n_jobs        : {n_jobs}")
    print(f"OOS windows   : {N_OOS_WINDOWS} (walk-forward)")
    print(f"Results dir   : {RESULTS_DIR}")
    print("=" * 60)

    t0 = time.time()
    results = []
    done_count = 0

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
        futures = {pool.submit(run_ml_job, t, m): (t, m) for t, m in combos}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done_count += 1
            status = r["status"]
            extra = ""
            if status == "done":
                extra = (f"  OOS_F1={r.get('f1_macro') or 0:.3f}"
                         f"  CV_F1={r.get('cv_f1_macro', 0):.3f}"
                         f"  windows={r.get('n_windows', 0)}")
            elif status == "error":
                extra = f"  ERR: {r.get('error', '')[:120]}"
            print(f"  [{done_count:3d}/{len(combos)}] {r['ticker']}/{r['model_name']}: {status}{extra}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Completed {len(results)} jobs in {elapsed:.1f}s")

    import pandas as _pd
    results_df = _pd.DataFrame(results)
    print("\nStatus counts:")
    print(results_df["status"].value_counts().to_string())

    done_df = results_df[results_df["status"] == "done"]
    if not done_df.empty:
        print("\nWalk-Forward OOS F1-macro by model (mean across tickers):")
        print(done_df.groupby("model_name")["f1_macro"].mean()
              .sort_values(ascending=False).to_string())

