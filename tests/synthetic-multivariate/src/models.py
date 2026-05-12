"""
Model factories for the synthetic experiment (regression task).

DL models delegate to models/base_models.py — all backbone implementations
live there so that synthetic, financial, and ford-a share the same code.
ML (scikit-learn) models are experiment-specific and remain here.
"""
from __future__ import annotations

import os as _os
import sys as _sys
from typing import Dict, Optional, Tuple

import numpy as np

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
# ML regressors (scikit-learn) — experiment-specific
# ---------------------------------------------------------------------------

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def create_svm_pipeline(params: Optional[Dict] = None) -> Pipeline:
    params = params or {}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            kernel=params.get("kernel", "rbf"),
            epsilon=params.get("epsilon", 0.1),
        )),
    ])


def create_random_forest(params: Optional[Dict] = None) -> RandomForestRegressor:
    params = params or {}
    return RandomForestRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        random_state=42,
        n_jobs=params.get("n_jobs", -1),
    )


def create_xgboost(params: Optional[Dict] = None):
    import xgboost as xgb
    params = params or {}
    return xgb.XGBRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        random_state=42,
        n_jobs=params.get("n_jobs", -1),
        verbosity=0,
    )


def create_lightgbm(params: Optional[Dict] = None):
    import lightgbm as lgb
    params = params or {}
    return lgb.LGBMRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", -1),
        learning_rate=params.get("learning_rate", 0.1),
        num_leaves=params.get("num_leaves", 31),
        subsample=params.get("subsample", 0.8),
        random_state=42,
        n_jobs=params.get("n_jobs", -1),
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# DL models — thin wrappers around base_models.build_model (task=regression)
# ---------------------------------------------------------------------------

def create_cnn_model(input_shape: Tuple[int, int], params: Optional[Dict] = None):
    return _build_model("CNN", "raw", input_shape, task="regression", cfg=params)


def create_lstm_model(input_shape: Tuple[int, int], params: Optional[Dict] = None):
    return _build_model("LSTM", "raw", input_shape, task="regression", cfg=params)


def create_cnn_lstm_model(input_shape: Tuple[int, int], params: Optional[Dict] = None):
    return _build_model("CNN_LSTM", "raw", input_shape, task="regression", cfg=params)


def create_transformer_model(input_shape: Tuple[int, int], params: Optional[Dict] = None):
    return _build_model("Transformer", "raw", input_shape, task="regression", cfg=params)


# ---------------------------------------------------------------------------
# Learned-wavelet DL models
# The wavelet_config and model_params dicts are merged; warm_start_db4 in
# wavelet_config determines whether the wavelet is warm-started from db4.
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
    return _build_model("CNN", mode, input_shape, task="regression", cfg=cfg)


def create_learned_wavelet_lstm_model(
    input_shape: Tuple[int, int],
    wavelet_config: Optional[Dict] = None,
    lstm_params: Optional[Dict] = None,
):
    cfg = _merge(wavelet_config, lstm_params)
    mode = "learned_wavelet" if cfg.get("warm_start_db4", False) else "learned_wavelet_no_warmup"
    return _build_model("LSTM", mode, input_shape, task="regression", cfg=cfg)


def create_learned_wavelet_cnn_lstm_model(
    input_shape: Tuple[int, int],
    wavelet_config: Optional[Dict] = None,
    cnn_lstm_params: Optional[Dict] = None,
):
    cfg = _merge(wavelet_config, cnn_lstm_params)
    mode = "learned_wavelet" if cfg.get("warm_start_db4", False) else "learned_wavelet_no_warmup"
    return _build_model("CNN_LSTM", mode, input_shape, task="regression", cfg=cfg)


def create_learned_wavelet_transformer_model(
    input_shape: Tuple[int, int],
    wavelet_config: Optional[Dict] = None,
    transformer_params: Optional[Dict] = None,
):
    cfg = _merge(wavelet_config, transformer_params)
    mode = "learned_wavelet" if cfg.get("warm_start_db4", False) else "learned_wavelet_no_warmup"
    return _build_model("Transformer", mode, input_shape, task="regression", cfg=cfg)


# ---------------------------------------------------------------------------
# Warmup/NoWarmup factory helpers (mirrors financial experiment's mode pattern)
# ---------------------------------------------------------------------------

def _with_warmup(create_fn, warm: bool):
    """Return a wrapper that forces warm_start_db4=warm in wavelet_config."""
    def wrapper(input_shape, wavelet_config=None, **kwargs):
        cfg = dict(wavelet_config or {})
        cfg["warm_start_db4"] = warm
        return create_fn(input_shape, wavelet_config=cfg, **kwargs)
    wrapper.__name__ = create_fn.__name__ + ("_warmup" if warm else "_no_warmup")
    return wrapper


# ---------------------------------------------------------------------------
# MODEL_FACTORY — for notebook-level create_model(name, ...) calls
# ---------------------------------------------------------------------------

MODEL_FACTORY = {
    # ML
    "SVM":          create_svm_pipeline,
    "RandomForest": create_random_forest,
    "XGBoost":      create_xgboost,
    "LightGBM":     create_lightgbm,

    # DL — raw signal
    "CNN":         create_cnn_model,
    "LSTM":        create_lstm_model,
    "CNN_LSTM":    create_cnn_lstm_model,
    "Transformer": create_transformer_model,

    # DL — learned wavelet (warm_start_db4 via wavelet_config)
    "LearnedWavelet_CNN":         create_learned_wavelet_cnn_model,
    "LearnedWavelet_LSTM":        create_learned_wavelet_lstm_model,
    "LearnedWavelet_CNN_LSTM":    create_learned_wavelet_cnn_lstm_model,
    "LearnedWavelet_Transformer": create_learned_wavelet_transformer_model,

    # Explicit warmup / no-warmup variants
    "LearnedWavelet_CNN_Warmup":           _with_warmup(create_learned_wavelet_cnn_model,         warm=True),
    "LearnedWavelet_CNN_NoWarmup":         _with_warmup(create_learned_wavelet_cnn_model,         warm=False),
    "LearnedWavelet_LSTM_Warmup":          _with_warmup(create_learned_wavelet_lstm_model,        warm=True),
    "LearnedWavelet_LSTM_NoWarmup":        _with_warmup(create_learned_wavelet_lstm_model,        warm=False),
    "LearnedWavelet_CNN_LSTM_Warmup":      _with_warmup(create_learned_wavelet_cnn_lstm_model,    warm=True),
    "LearnedWavelet_CNN_LSTM_NoWarmup":    _with_warmup(create_learned_wavelet_cnn_lstm_model,    warm=False),
    "LearnedWavelet_Transformer_Warmup":   _with_warmup(create_learned_wavelet_transformer_model, warm=True),
    "LearnedWavelet_Transformer_NoWarmup": _with_warmup(create_learned_wavelet_transformer_model, warm=False),
}


def create_model(model_name: str, **kwargs):
    if model_name not in MODEL_FACTORY:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list(MODEL_FACTORY)}"
        )
    return MODEL_FACTORY[model_name](**kwargs)
