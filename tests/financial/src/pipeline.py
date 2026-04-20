"""
FinancialExperimentPipeline — runs a full PurgedKFold experiment for one job.
Called by experiment_runner.py (GPU queue) and by notebook 03_dl_experiments.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

_FINANCIAL_DIR = Path(__file__).parent.parent
_ROOT = _FINANCIAL_DIR.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_FINANCIAL_DIR))

from utils.validation import PurgedKFold  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_windows(
    X: np.ndarray,
    y: np.ndarray,
    prices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding window: (N-seq_len, seq_len, n_features), aligned labels/prices."""
    n = len(X)
    idx = np.arange(seq_len, n)
    X_win = np.stack([X[i - seq_len: i] for i in idx])   # (N', seq_len, F)
    y_win = y[idx]
    p_win = prices[idx]
    return X_win, y_win, p_win


def _class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _scale(X_train, X_val, X_test):
    orig_shape_train = X_train.shape
    orig_shape_val   = X_val.shape
    orig_shape_test  = X_test.shape
    n_feat = X_train.shape[-1]

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(orig_shape_train)
    X_v_s  = scaler.transform(X_val.reshape(-1, n_feat)).reshape(orig_shape_val)
    X_te_s = scaler.transform(X_test.reshape(-1, n_feat)).reshape(orig_shape_test)
    return X_tr_s, X_v_s, X_te_s, scaler


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class FinancialExperimentPipeline:
    """
    Full experiment for one (ticker, model_name, mode) job.

    Parameters
    ----------
    ticker      : stock ticker (e.g. "PETR4.SA")
    model_name  : CNN | LSTM | CNN_LSTM | Transformer
    mode        : raw | db4 | learned_wavelet
    config      : merged config dict (from run_dl_queue or notebook)
    results_dir : where to write metrics.json and predictions
    """

    def __init__(
        self,
        ticker: str,
        model_name: str,
        mode: str,
        config: Optional[dict] = None,
        results_dir: Optional[str] = None,
    ):
        self.ticker     = ticker
        self.model_name = model_name
        self.mode       = mode

        # Merge provided config with defaults
        from config.experiment_config import (
            DL_MODELS_CONFIG,
            DL_TRAINING_CONFIG,
            LABELING_CONFIG,
            LEARNED_WAVELET_CONFIG,
            VALIDATION_CONFIG,
            BACKTEST_CONFIG,
        )
        self.cfg = {
            **DL_TRAINING_CONFIG,
            **LEARNED_WAVELET_CONFIG,
            **DL_MODELS_CONFIG.get(model_name, {}),
            **VALIDATION_CONFIG,
            **LABELING_CONFIG,
            **BACKTEST_CONFIG,
            **(config or {}),
        }

        self.results_dir = Path(results_dir or (_FINANCIAL_DIR / "results"))
        self.models_dir  = _FINANCIAL_DIR / "saved_models"

    # ── Public ──────────────────────────────────────────────────────────────

    @property
    def job_key(self) -> str:
        return f"{self.model_name}_{self.mode}"

    @property
    def job_results_dir(self) -> Path:
        return self.results_dir / self.ticker / self.job_key

    def is_done(self) -> bool:
        return (self.job_results_dir / "metrics.json").exists()

    def run(self) -> dict:
        if self.is_done():
            logger.info("Already done: %s / %s — skipping.", self.ticker, self.job_key)
            with open(self.job_results_dir / "metrics.json") as f:
                return json.load(f)

        logger.info("Starting: %s / %s", self.ticker, self.job_key)
        t0 = time.time()

        X, y, prices = self._load_data()
        X_win, y_win, p_win = _make_windows(X, y, prices, self.cfg["sequence_length"])

        cw = _class_weights(y_win)
        pkf = self._build_cv(y_win)

        ml_fold_metrics:  list[dict] = []
        fin_fold_metrics: list[dict] = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(pkf.split(X_win, y_win)):
            logger.info("  Fold %d/%d", fold_idx + 1, self.cfg["n_folds"])

            # Inner val split (last val_split fraction of train)
            n_val = max(1, int(len(train_val_idx) * self.cfg.get("val_split", 0.15)))
            train_idx = train_val_idx[:-n_val]
            val_idx   = train_val_idx[-n_val:]

            X_tr, X_v, X_te, _ = _scale(X_win[train_idx], X_win[val_idx], X_win[test_idx])
            y_tr = y_win[train_idx]
            y_v  = y_win[val_idx]
            y_te = y_win[test_idx]

            model = self._build_model(X_tr.shape[1:])
            callbacks = self._callbacks(fold_idx)

            model.fit(
                X_tr, y_tr,
                validation_data=(X_v, y_v),
                class_weight=cw,
                epochs=self.cfg["epochs"],
                batch_size=self.cfg["batch_size"],
                callbacks=callbacks,
                verbose=0,
            )

            y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)

            from src.evaluation import ClassificationEvaluator, FinancialMetrics
            from src.backtest import simulate_strategy, buy_and_hold_returns

            ml_metrics  = ClassificationEvaluator.evaluate(y_te, y_pred)
            strat_ret   = simulate_strategy(
                y_pred,
                pd.Series(p_win[test_idx]),
                transaction_cost=self.cfg.get("transaction_cost", 0.001),
                allow_short=self.cfg.get("allow_short", True),
            )
            bh_ret = buy_and_hold_returns(pd.Series(p_win[test_idx]))
            fin_metrics = FinancialMetrics.compute(strat_ret, bh_ret)

            ml_fold_metrics.append(ml_metrics)
            fin_fold_metrics.append(fin_metrics)

            self._save_predictions(y_te, y_pred, fold_idx)

        # Aggregate
        from src.evaluation import ClassificationEvaluator, FinancialMetrics, ResultsManager
        agg_ml  = ClassificationEvaluator.evaluate_cv(
            [m.get("y_true", np.array([])) for m in ml_fold_metrics],  # placeholder — already computed
            [m.get("y_pred", np.array([])) for m in ml_fold_metrics],
        )
        # Re-aggregate directly from fold dicts
        agg_ml  = _aggregate_dicts(ml_fold_metrics)
        agg_fin = FinancialMetrics.aggregate_cv(fin_fold_metrics)

        elapsed = time.time() - t0
        rm = ResultsManager(self.results_dir)
        rm.log_experiment(
            ticker=self.ticker,
            model_name=self.model_name,
            mode=self.mode,
            ml_metrics=agg_ml,
            financial_metrics=agg_fin,
            config=self.cfg,
            extra={"elapsed_seconds": elapsed, "n_folds": self.cfg["n_folds"]},
        )

        logger.info("Done: %s / %s  (%.1fs)", self.ticker, self.job_key, elapsed)
        return {**agg_ml, **agg_fin}

    # ── Private ─────────────────────────────────────────────────────────────

    def _load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from src.data_loader import load_processed, load_labels
        features = load_processed(self.ticker)
        labels   = load_labels(self.ticker)

        # Align
        common = features.index.intersection(labels.index)
        features = features.loc[common]
        labels   = labels.loc[common]

        X      = features.values.astype(np.float32)
        y      = labels.values.astype(np.int32)
        prices = features["log_return_1"].values if "log_return_1" in features.columns else np.zeros(len(X))
        # Use raw close price if available via the parquet
        # (stored as part of features for backtest reference)
        return X, y, prices

    def _build_cv(self, y: np.ndarray) -> PurgedKFold:
        pct_embargo = self.cfg["embargo_days"] / len(y)
        return PurgedKFold(
            n_splits=self.cfg["n_folds"],
            pct_embargo=pct_embargo,
        )

    def _build_model(self, input_shape: tuple):
        from src.models import build_model
        return build_model(
            model_name=self.model_name,
            mode=self.mode,
            input_shape=input_shape,
            n_classes=3,
            cfg=self.cfg,
        )

    def _callbacks(self, fold_idx: int) -> list:
        from src.models import get_callbacks
        model_path = (
            self.models_dir / self.ticker / self.job_key / f"fold_{fold_idx}.keras"
        )
        return get_callbacks(
            model_path=model_path,
            early_patience=self.cfg["early_stopping_patience"],
            lr_patience=self.cfg["reduce_lr_patience"],
            lr_factor=self.cfg.get("reduce_lr_factor", 0.5),
            min_lr=self.cfg.get("min_lr", 1e-6),
        )

    def _save_predictions(self, y_true, y_pred, fold_idx):
        out_dir = self.job_results_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"predictions_fold{fold_idx}.npz",
            y_true=y_true,
            y_pred=y_pred,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _aggregate_dicts(dicts: list[dict]) -> dict:
    """Mean ± std across fold dicts (skip non-numeric)."""
    result = {}
    for key in dicts[0]:
        try:
            values = [float(d[key]) for d in dicts if not np.isnan(float(d[key]))]
            result[key]           = float(np.mean(values)) if values else float("nan")
            result[f"{key}_std"]  = float(np.std(values))  if values else float("nan")
        except (TypeError, ValueError):
            result[key] = dicts[0][key]
    return result
