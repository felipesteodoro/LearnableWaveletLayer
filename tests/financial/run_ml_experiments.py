#!/usr/bin/env python3
"""
Canonical ML experiments — IS/OOS split + grid search on IS.

Methodology
-----------
1. Load pre-computed engineered features (load_processed) and labels.
2. Reserve the last `test_years` (6) years as a fixed OOS test set.
3. Grid-search hyperparameters using PurgedKFold (n_folds=5) on IS only.
4. Retrain the best config on the full IS set.
5. Evaluate on OOS → final reported ml_metrics / fin_metrics.

Results are saved to results/<ticker>/<model>_ml/metrics.json.
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
    LABELING_CONFIG, ML_N_JOBS_MODEL
)
from src.data_loader import load_processed, load_labels
from src.evaluation import ClassificationEvaluator, FinancialMetrics, ResultsManager
from src.backtest import simulate_strategy, buy_and_hold_returns

RESULTS_DIR = BASE / "results"


# ── Hyperparameter Grid (small search per model) ──────────────────────────
PARAM_GRID = {
    "RandomForest": [
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": 10,   "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5},
    ],
    "XGBoost": [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03},
    ],
    "LightGBM": [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,  "num_leaves": 15},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 31},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 63},
    ],
    "CatBoost": [
        {"iterations": 100, "depth": 4, "learning_rate": 0.1},
        {"iterations": 200, "depth": 6, "learning_rate": 0.05},
        {"iterations": 300, "depth": 6, "learning_rate": 0.03},
    ],
    "Stacking": [
        {"base_n_estimators": 50},
        {"base_n_estimators": 100},
    ],
}


# ── Model Factory ─────────────────────────────────────────────────────────
def build_model(model_name: str, overrides: dict | None = None):
    """Build model from base config + optional hyperparameter overrides."""
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


# ── Single Job Runner ─────────────────────────────────────────────────────
def run_ml_job(ticker: str, model_name: str) -> dict:
    from utils.validation import PurgedKFold

    result_path = RESULTS_DIR / ticker / f"{model_name}_ml" / "metrics.json"
    if result_path.exists():
        return {"ticker": ticker, "model_name": model_name, "status": "skipped"}

    try:
        # ── 1. Load data ─────────────────────────────────────────────────
        features = load_processed(ticker)
        labels   = load_labels(ticker)
        common   = features.index.intersection(labels.index)
        features = features.loc[common].dropna()
        labels   = labels.loc[features.index]

        X     = features.values.astype(np.float32)
        y     = labels.values
        dates = features.index

        # Synthetic price series from log-returns for backtest.
        # log_return_1 is a log-return; compound it so pct_change() inside
        # simulate_strategy yields the correct daily return.
        if "log_return_1" in features.columns:
            lr = features["log_return_1"].values
            price_series = pd.Series(np.exp(np.cumsum(lr)), index=dates)
        else:
            price_series = pd.Series(np.ones(len(dates)), index=dates)

        # ── 2. IS / OOS split ────────────────────────────────────────────
        test_years = VALIDATION_CONFIG.get("test_years", 6)
        oos_start  = dates[-1] - pd.DateOffset(years=test_years)

        is_mask  = dates <= oos_start
        oos_mask = dates  > oos_start

        if is_mask.sum() < 100:
            raise ValueError(f"Not enough IS samples ({is_mask.sum()}).")
        if oos_mask.sum() < 10:
            raise ValueError(f"Not enough OOS samples ({oos_mask.sum()}).")

        X_is, y_is, dates_is = X[is_mask], y[is_mask], dates[is_mask]
        X_oos, y_oos         = X[oos_mask], y[oos_mask]
        prices_oos           = price_series[oos_mask]

        # ── 3. PurgedKFold setup on IS ───────────────────────────────────
        time_horizon = LABELING_CONFIG.get("time_horizon", 10)
        t1_is = pd.Series(
            [d + pd.Timedelta(days=int(time_horizon * 1.5)) for d in dates_is],
            index=dates_is,
        )
        pct_embargo = VALIDATION_CONFIG.get("embargo_days", 10) / len(y_is)
        pkf = PurgedKFold(
            n_splits=VALIDATION_CONFIG.get("n_folds", 5),
            t1=t1_is,
            pct_embargo=pct_embargo,
        )
        X_is_df = pd.DataFrame(X_is, index=dates_is)

        # ── 4. Grid search on IS ─────────────────────────────────────────
        param_grid     = PARAM_GRID.get(model_name, [{}])
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

        # ── 5. CV metrics with best config (for reporting) ───────────────
        cv_ml_folds, cv_fin_folds = [], []

        for train_val_idx, val_idx in pkf.split(X_is_df, y_is):
            n_val_inner = max(1, int(len(train_val_idx) * 0.15))
            train_idx   = train_val_idx[:-n_val_inner]

            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X_is[train_idx])
            X_va = scaler.transform(X_is[val_idx])

            model = build_model(model_name, best_overrides)
            model.fit(X_tr, y_is[train_idx])
            y_pred_cv = np.asarray(model.predict(X_va)).ravel()

            cv_ml_folds.append(ClassificationEvaluator.evaluate(y_is[val_idx], y_pred_cv))
            strat_cv = simulate_strategy(
                y_pred_cv, price_series[is_mask][val_idx],
                transaction_cost=BACKTEST_CONFIG["transaction_cost"],
                allow_short=BACKTEST_CONFIG["allow_short"],
            )
            bh_cv = buy_and_hold_returns(price_series[is_mask][val_idx])
            cv_fin_folds.append(FinancialMetrics.compute(strat_cv, bh_cv))

        cv_ml  = {k: float(np.mean([f[k] for f in cv_ml_folds])) for k in cv_ml_folds[0]}
        cv_fin = FinancialMetrics.aggregate_cv(cv_fin_folds)

        # ── 6. Final model: full IS → evaluate OOS ───────────────────────
        final_scaler = RobustScaler()
        X_is_scaled  = final_scaler.fit_transform(X_is)
        X_oos_scaled = final_scaler.transform(X_oos)

        final_model = build_model(model_name, best_overrides)
        final_model.fit(X_is_scaled, y_is)

        y_pred_oos = np.asarray(final_model.predict(X_oos_scaled)).ravel()

        oos_ml    = ClassificationEvaluator.evaluate(y_oos, y_pred_oos)
        strat_oos = simulate_strategy(
            y_pred_oos, prices_oos,
            transaction_cost=BACKTEST_CONFIG["transaction_cost"],
            allow_short=BACKTEST_CONFIG["allow_short"],
        )
        bh_oos  = buy_and_hold_returns(prices_oos)
        oos_fin = FinancialMetrics.compute(strat_oos, bh_oos)

        # ── 7. Persist ───────────────────────────────────────────────────
        result_path.parent.mkdir(parents=True, exist_ok=True)

        def _safe(v):
            try:
                return float(v) if np.isfinite(float(v)) else None
            except Exception:
                return None

        payload = {
            "ml_metrics":       {k: _safe(v) for k, v in oos_ml.items()},
            "fin_metrics":      {k: _safe(v) for k, v in oos_fin.items()},
            "cv_ml_metrics":    {k: _safe(v) for k, v in cv_ml.items()},
            "cv_fin_metrics":   {k: _safe(v) for k, v in cv_fin.items()},
            "best_params":      best_overrides,
            "best_cv_f1_macro": float(best_score),
            "grid_search":      grid_results,
            "split_info": {
                "oos_start":  str(oos_start.date()),
                "n_is":       int(is_mask.sum()),
                "n_oos":      int(oos_mask.sum()),
                "test_years": test_years,
            },
        }
        result_path.write_text(json.dumps(payload, indent=2))
        np.savez(
            result_path.parent / "predictions_oos.npz",
            y_true=y_oos,
            y_pred=y_pred_oos,
        )

        return {
            "ticker":      ticker,
            "model_name":  model_name,
            "status":      "done",
            "f1_macro":    oos_ml.get("f1_macro", float("nan")),
            "cv_f1_macro": float(best_score),
            "best_params": str(best_overrides),
            "n_oos":       int(oos_mask.sum()),
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

    print(f"Total jobs : {len(combos)}")
    print(f"n_jobs     : {n_jobs}")
    print(f"Results dir: {RESULTS_DIR}")
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
                extra = (f"  OOS_F1={r.get('f1_macro', 0):.3f}"
                         f"  CV_F1={r.get('cv_f1_macro', 0):.3f}"
                         f"  n_oos={r.get('n_oos', 0)}"
                         f"  best={r.get('best_params', '')}")
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
        print("\nOOS F1-macro by model (mean across tickers):")
        print(done_df.groupby("model_name")["f1_macro"].mean()
              .sort_values(ascending=False).to_string())
        print("\nIS CV F1-macro by model (mean across tickers):")
        print(done_df.groupby("model_name")["cv_f1_macro"].mean()
              .sort_values(ascending=False).to_string())
