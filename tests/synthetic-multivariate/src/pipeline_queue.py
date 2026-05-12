"""
Lightweight pipeline for GPU-queue-based execution.

Each subprocess runs one (model_name, mode, config, config_idx) combination,
loads pre-split data from disk, trains, evaluates, and saves metrics.json.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Ensure models/ is importable from within the subprocess
_ROOT = Path(__file__).parent.parent.parent.parent
_MODELS_DIR = str(_ROOT / "models")
for _p in (_MODELS_DIR, str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class SyntheticExperimentPipeline:
    """
    Loads pre-split data from disk, trains one (model_name, mode, config_idx)
    configuration, and saves metrics.json + model checkpoint.

    Parameters
    ----------
    model_name  : CNN | LSTM | CNN_LSTM | Transformer
    mode        : raw | learned_wavelet | learned_wavelet_no_warmup
    config      : merged dict (DL_MODELS_CONFIG + DL_TRAINING_CONFIG + LEARNED_WAVELET_CONFIG
                  + grid variation)
    config_idx  : grid index — used only for directory naming
    results_dir : base results directory for this run  (e.g. results/2026-05-09)
    data_dir    : directory containing X_train.npy, y_train.npy, etc.
    """

    def __init__(
        self,
        model_name: str,
        mode: str,
        config: dict,
        config_idx: int,
        results_dir: str | Path,
        data_dir: Optional[str | Path] = None,
    ):
        self.model_name = model_name
        self.mode = mode
        self.config = config
        self.config_idx = config_idx
        self.results_dir = Path(results_dir) / f"{model_name}_{mode}" / f"cfg{config_idx:03d}"
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "data"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        self.results_dir.mkdir(parents=True, exist_ok=True)

        X_train, y_train, X_val, y_val, X_test, y_test = self._load_data()
        input_shape = X_train.shape[1:]
        logger.info(
            "Data loaded — train %s  val %s  test %s  input_shape %s",
            X_train.shape, X_val.shape, X_test.shape, input_shape,
        )

        model = self._build_model(input_shape)
        history = self._train(model, X_train, y_train, X_val, y_val)
        metrics = self._evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, history)

        self._save_metrics(metrics)
        logger.info(
            "Done | %s/%s/cfg%03d — RMSE=%.5f  R²=%.4f",
            self.model_name, self.mode, self.config_idx,
            metrics["test_rmse"], metrics["test_r2"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_data(self):
        d = self.data_dir

        def _load(name):
            p = d / name
            if not p.exists():
                raise FileNotFoundError(
                    f"Data file not found: {p}\n"
                    "Run notebook 00_generate_synthetic_signal.ipynb first."
                )
            return np.load(p)

        X_train = _load("X_train.npy")
        y_train = _load("y_train.npy")
        X_val   = _load("X_val.npy")
        y_val   = _load("y_val.npy")
        X_test  = _load("X_test.npy")
        y_test  = _load("y_test.npy")

        # Ensure 3-D (batch, time, features)
        if X_train.ndim == 2:
            X_train = X_train[..., np.newaxis]
            X_val   = X_val[..., np.newaxis]
            X_test  = X_test[..., np.newaxis]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _build_model(self, input_shape: tuple):
        from base_models import build_model
        return build_model(
            model_name=self.model_name,
            mode=self.mode,
            input_shape=input_shape,
            task="regression",
            cfg=self.config,
        )

    def _train(self, model, X_train, y_train, X_val, y_val):
        from base_models import get_callbacks

        model_path = self.results_dir / "model.keras"
        callbacks = get_callbacks(
            model_path,
            early_patience=self.config.get("early_stopping_patience", 15),
            lr_patience=self.config.get("reduce_lr_patience", 7),
            lr_factor=self.config.get("reduce_lr_factor", 0.5),
            min_lr=self.config.get("min_lr", 1e-6),
            use_reduce_lr=not (self.model_name == "Transformer" and self.config.get("use_warmup", False)),
            verbose=0,
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get("epochs", 100),
            batch_size=self.config.get("batch_size", 64),
            callbacks=callbacks,
            verbose=self.config.get("verbose", 1),
        )
        return history

    def _evaluate(self, model, X_train, y_train, X_val, y_val, X_test, y_test, history):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        def _metrics(y_true, y_pred, prefix):
            mse  = float(mean_squared_error(y_true, y_pred))
            mae  = float(mean_absolute_error(y_true, y_pred))
            r2   = float(r2_score(y_true, y_pred))
            rmse = float(mse ** 0.5)
            return {
                f"{prefix}_mse":  mse,
                f"{prefix}_rmse": rmse,
                f"{prefix}_mae":  mae,
                f"{prefix}_r2":   r2,
            }

        y_pred_train = model.predict(X_train, verbose=0).flatten()
        y_pred_val   = model.predict(X_val,   verbose=0).flatten()
        y_pred_test  = model.predict(X_test,  verbose=0).flatten()

        metrics = {
            "model_name": self.model_name,
            "mode": self.mode,
            "config_idx": self.config_idx,
            "config": self.config,
            "epochs_trained": len(history.history["loss"]),
            **_metrics(y_train, y_pred_train, "train"),
            **_metrics(y_val,   y_pred_val,   "val"),
            **_metrics(y_test,  y_pred_test,  "test"),
        }
        return metrics

    def _save_metrics(self, metrics: dict) -> None:
        path = self.results_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
