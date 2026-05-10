"""
Lightweight pipeline for GPU-queue-based execution (Ford-A binary classification).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent.parent
_MODELS_DIR = str(_ROOT / "models")
for _p in (_MODELS_DIR, str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class FordAExperimentPipeline:
    """
    Loads pre-split Ford-A data, trains one (model_name, mode, config_idx)
    binary classifier, and saves metrics.json.

    Parameters
    ----------
    model_name  : CNN | LSTM | CNN_LSTM | Transformer
    mode        : raw | learned_wavelet | learned_wavelet_no_warmup
    config      : merged config dict
    config_idx  : grid index (used for directory naming only)
    results_dir : base results directory for this run
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
            "Done | %s/%s/cfg%03d — acc=%.4f  f1=%.4f  auc=%.4f",
            self.model_name, self.mode, self.config_idx,
            metrics["test_accuracy"], metrics["test_f1_macro"], metrics["test_auc_roc"],
        )
        return metrics

    def _load_data(self):
        d = self.data_dir

        def _load(name):
            p = d / name
            if not p.exists():
                raise FileNotFoundError(
                    f"Data file not found: {p}\n"
                    "Run notebook 00_download_and_feature_extraction.ipynb first."
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
            task="binary",
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
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
        )

        def _metrics(y_true, y_proba, prefix):
            y_pred = (y_proba > 0.5).astype(int)
            return {
                f"{prefix}_accuracy":        float(accuracy_score(y_true, y_pred)),
                f"{prefix}_f1_macro":        float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
                f"{prefix}_f1_weighted":     float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                f"{prefix}_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                f"{prefix}_recall_macro":    float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
                f"{prefix}_auc_roc":         float(roc_auc_score(y_true, y_proba)),
            }

        y_proba_train = model.predict(X_train, verbose=0).flatten()
        y_proba_val   = model.predict(X_val,   verbose=0).flatten()
        y_proba_test  = model.predict(X_test,  verbose=0).flatten()

        return {
            "model_name":    self.model_name,
            "mode":          self.mode,
            "config_idx":    self.config_idx,
            "config":        self.config,
            "epochs_trained": len(history.history["loss"]),
            **_metrics(y_train, y_proba_train, "train"),
            **_metrics(y_val,   y_proba_val,   "val"),
            **_metrics(y_test,  y_proba_test,  "test"),
        }

    def _save_metrics(self, metrics: dict) -> None:
        path = self.results_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
