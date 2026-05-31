"""
Model factories for the SelfRegulationSCP1 experiment (binary classification).

DL models delegate to models/base_models.py — all backbone implementations
live there so that synthetic, financial, ford-a, uwave and scp1 share the same code.
ML (scikit-learn) classifiers are experiment-specific and remain here.

NB: classificação BINÁRIA — modelos DL usam task="binary" (sigmoid +
binary_crossentropy, labels inteiros 0/1).
"""
from __future__ import annotations

import os as _os
import sys as _sys
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — add models/ so base_models / dl_utils / LWT are importable
# ---------------------------------------------------------------------------
_MODELS_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "models")
)
if _MODELS_DIR not in _sys.path:
    _sys.path.insert(0, _MODELS_DIR)

from base_models import (  # noqa: E402
    build_model as _build_model,
    get_callbacks,
    get_distribute_strategy,
)

# ---------------------------------------------------------------------------
# ML classifiers (scikit-learn) — experiment-specific
# ---------------------------------------------------------------------------

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


def create_linear_svc(params: Optional[Dict] = None) -> CalibratedClassifierCV:
    """LinearSVC wrapped in CalibratedClassifierCV for predict_proba support."""
    params = params or {}
    base = LinearSVC(
        C=params.get("C", 1.0),
        loss=params.get("loss", "squared_hinge"),
        max_iter=params.get("max_iter", 10000),
        random_state=42,
    )
    return CalibratedClassifierCV(base, cv=3)


def create_sgd_classifier(params: Optional[Dict] = None) -> SGDClassifier:
    params = params or {}
    return SGDClassifier(
        loss=params.get("loss", "hinge"),
        alpha=params.get("alpha", 1e-4),
        penalty=params.get("penalty", "l2"),
        l1_ratio=params.get("l1_ratio", 0.15),
        learning_rate=params.get("learning_rate", "optimal"),
        max_iter=params.get("max_iter", 5000),
        random_state=42,
    )


def create_logistic_regression(params: Optional[Dict] = None) -> LogisticRegression:
    params = params or {}
    return LogisticRegression(
        C=params.get("C", 1.0),
        penalty=params.get("penalty", "l2"),
        l1_ratio=params.get("l1_ratio", 0.5),
        solver=params.get("solver", "saga"),
        max_iter=params.get("max_iter", 10000),
        random_state=42,
    )


def create_rf_classifier(params: Optional[Dict] = None) -> RandomForestClassifier:
    params = params or {}
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", "sqrt"),
        random_state=42,
        n_jobs=params.get("n_jobs", -1),
    )


def create_xgb_classifier(params: Optional[Dict] = None):
    import xgboost as xgb
    params = params or {}
    return xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=params.get("n_jobs", -1),
        verbosity=0,
    )


def create_lgbm_classifier(params: Optional[Dict] = None):
    import lightgbm as lgb
    params = params or {}
    return lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", -1),
        learning_rate=params.get("learning_rate", 0.1),
        num_leaves=params.get("num_leaves", 31),
        subsample=params.get("subsample", 0.8),
        random_state=42,
        n_jobs=params.get("n_jobs", -1),
        verbose=-1,
    )


def create_catboost_classifier(params: Optional[Dict] = None):
    from catboost import CatBoostClassifier
    params = params or {}
    return CatBoostClassifier(
        iterations=params.get("iterations", 200),
        depth=params.get("depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
        random_seed=42,
        thread_count=params.get("thread_count", 4),
        verbose=0,
    )


# ---------------------------------------------------------------------------
# DL models — thin wrappers around base_models.build_model (task=binary)
# ---------------------------------------------------------------------------

def create_cnn_model(input_shape: Tuple[int, int], params: Optional[Dict] = None):
    return _build_model("CNN", "raw", input_shape, task="binary", cfg=params)


def create_lstm_model(input_shape: Tuple[int, int], params: Optional[Dict] = None):
    return _build_model("LSTM", "raw", input_shape, task="binary", cfg=params)


def create_cnn_lstm_model(input_shape: Tuple[int, int], params: Optional[Dict] = None):
    return _build_model("CNN_LSTM", "raw", input_shape, task="binary", cfg=params)


def create_transformer_model(input_shape: Tuple[int, int], params: Optional[Dict] = None):
    return _build_model("Transformer", "raw", input_shape, task="binary", cfg=params)


# ---------------------------------------------------------------------------
# Learned-wavelet DL models
# ---------------------------------------------------------------------------

def _merge(wavelet_config, model_params):
    return {**(wavelet_config or {}), **(model_params or {})}


def create_learned_wavelet_cnn_model(
    input_shape: Tuple[int, int],
    wavelet_config: Optional[Dict] = None,
    cnn_params: Optional[Dict] = None,
):
    cfg = _merge(wavelet_config, cnn_params)
    mode = "learned_wavelet" if cfg.get("warm_start_db4", False) else "learned_wavelet_no_warmup"
    return _build_model("CNN", mode, input_shape, task="binary", cfg=cfg)


def create_learned_wavelet_lstm_model(
    input_shape: Tuple[int, int],
    wavelet_config: Optional[Dict] = None,
    lstm_params: Optional[Dict] = None,
):
    cfg = _merge(wavelet_config, lstm_params)
    mode = "learned_wavelet" if cfg.get("warm_start_db4", False) else "learned_wavelet_no_warmup"
    return _build_model("LSTM", mode, input_shape, task="binary", cfg=cfg)


def create_learned_wavelet_cnn_lstm_model(
    input_shape: Tuple[int, int],
    wavelet_config: Optional[Dict] = None,
    cnn_lstm_params: Optional[Dict] = None,
):
    cfg = _merge(wavelet_config, cnn_lstm_params)
    mode = "learned_wavelet" if cfg.get("warm_start_db4", False) else "learned_wavelet_no_warmup"
    return _build_model("CNN_LSTM", mode, input_shape, task="binary", cfg=cfg)


def create_learned_wavelet_transformer_model(
    input_shape: Tuple[int, int],
    wavelet_config: Optional[Dict] = None,
    transformer_params: Optional[Dict] = None,
):
    cfg = _merge(wavelet_config, transformer_params)
    mode = "learned_wavelet" if cfg.get("warm_start_db4", False) else "learned_wavelet_no_warmup"
    return _build_model("Transformer", mode, input_shape, task="binary", cfg=cfg)
