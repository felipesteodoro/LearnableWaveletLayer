"""
Centralised configuration for financial experiments.
Mirrors the structure of tests/synthetic/config/experiment_config.py.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

TICKERS = [
    "ABEV3.SA", "B3SA3.SA",  "BBAS3.SA", "BBDC4.SA", "BRKM5.SA",
    "COGN3.SA", "CSNA3.SA",  "CYRE3.SA", "EZTC3.SA", "GGBR4.SA",
    "HYPE3.SA", "ITUB4.SA",  "LREN3.SA", "MGLU3.SA", "MRVE3.SA",
    "MULT3.SA", "PETR4.SA",  "RADL3.SA", "RENT3.SA", "SUZB3.SA",
    "UGPA3.SA", "USIM5.SA",  "VALE3.SA", "VIVT3.SA", "WEGE3.SA",
]

DL_MODELS  = ["CNN", "LSTM", "CNN_LSTM", "Transformer"]
ML_MODELS  = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "Stacking"]
MODES      = ["raw", "db4", "learned_wavelet"]

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_CONFIG = {
    "sequence_length": 30,           # sliding window (trading days)
    # Distribution-quality thresholds — features failing these are dropped
    "adf_max_failing_pct": 0.30,     # drop if non-stationary in >30% of assets
    "max_kurtosis": 20.0,            # after 1%-99% winsorisation
    "max_nan_pct": 0.05,             # after warmup period
    "max_correlation": 0.95,         # pairwise — keep more interpretable one
    # Rolling-window sizes
    "rsi_period": 14,
    "stoch_period": 14,
    "williams_period": 14,
    "roc_period": 10,
    "ema_period": 20,
    "sma_period": 50,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "adx_period": 14,
    "atr_period": 14,
    "bb_period": 20,
    "hv_period": 21,
    "vol_sma_period": 20,
    "zscore_period": 20,
    "autocorr_period": 20,
    "autocorr_lag": 1,
}

# ---------------------------------------------------------------------------
# Labeling — Triple Barrier Method
# ---------------------------------------------------------------------------

LABELING_CONFIG = {
    "pt_sl": [1.5, 1.0],    # take-profit / stop-loss multipliers of daily ATR
    "time_horizon": 10,      # max holding period in trading days
    "min_ret": 0.001,        # minimum return threshold to count a barrier
    "vol_span": 100,         # EWM span for daily volatility estimate
    # Label mapping: {-1 → 0 (sell), 0 → 1 (hold), 1 → 2 (buy)}
    "n_classes": 3,
}

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

VALIDATION_CONFIG = {
    "n_folds": 5,
    "embargo_days": 10,       # trading days embargo between train and test
    "val_split": 0.15,        # fraction of training fold used as validation
    "test_years": 6,          # last N years reserved as out-of-sample test
}

# ---------------------------------------------------------------------------
# Wavelet
# ---------------------------------------------------------------------------

WAVELET_CONFIG = {
    "wavelet_type": "db2",
    "decomposition_level": 3,
}

LEARNED_WAVELET_CONFIG = {
    "levels": 2,
    "kernel_size": 32,
    "wavelet_net_units": 32,
    "reg_energy": 1e-2,
    "reg_high_dc": 1e-2,
    "reg_smooth": 1e-3,
    "output_mode": "concat",
}

# ---------------------------------------------------------------------------
# Deep Learning
# ---------------------------------------------------------------------------

DL_TRAINING_CONFIG = {
    "epochs": int(os.environ.get("EPOCHS_OVERRIDE", "100")),
    "batch_size": 64,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 7,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6,
    "learning_rate": 1e-3,
}

DL_MODELS_CONFIG = {
    "CNN": {
        "filters": [64, 128, 256],
        "kernel_size": 3,
        "dropout_rate": 0.3,
        "l2_reg": 1e-3,
    },
    "LSTM": {
        "units": [128, 64],
        "dropout_rate": 0.3,
        "recurrent_dropout": 0.1,
        "l2_reg": 1e-3,
    },
    "CNN_LSTM": {
        "filters": [64, 128],
        "lstm_units": [64, 32],
        "dropout_rate": 0.3,
        "l2_reg": 1e-3,
    },
    "Transformer": {
        "num_heads": 4,
        "head_size": 32,
        "ff_dim": 128,
        "num_blocks": 2,
        "dropout_rate": 0.2,
        "l2_reg": 1e-4,
    },
}

# Grid axes for DL hyperparameter search (subset, controlled by MAX_GRID_CONFIGS)
DL_GRID_AXES = {
    "CNN": {
        "dropout_rate": [0.2, 0.3, 0.4],
        "l2_reg": [1e-4, 1e-3, 1e-2],
    },
    "LSTM": {
        "dropout_rate": [0.2, 0.3],
        "l2_reg": [1e-4, 1e-3],
    },
    "CNN_LSTM": {
        "dropout_rate": [0.2, 0.3],
        "l2_reg": [1e-4, 1e-3],
    },
    "Transformer": {
        "dropout_rate": [0.1, 0.2],
        "num_heads": [2, 4],
    },
}

# ---------------------------------------------------------------------------
# Machine Learning
# ---------------------------------------------------------------------------

ML_N_JOBS_OUTER = 16                                    # parallel (ticker × model)
ML_N_JOBS_MODEL = 2                                     # per-model thread count

ML_MODELS_CONFIG = {
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "n_jobs": ML_N_JOBS_MODEL,
        "random_state": 42,
        "class_weight": "balanced",
    },
    "XGBoost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "nthread": ML_N_JOBS_MODEL,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
    },
    "LightGBM": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "num_threads": ML_N_JOBS_MODEL,
        "random_state": 42,
        "class_weight": "balanced",
        "verbose": -1,
    },
    "CatBoost": {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "thread_count": ML_N_JOBS_MODEL,
        "random_state": 42,
        "auto_class_weights": "Balanced",
        "verbose": 0,
    },
}

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

BACKTEST_CONFIG = {
    "transaction_cost": 0.001,   # 0.1% per trade (round-trip = 0.2%)
    "allow_short": True,         # True: sell=short; False: sell=flat
}

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

RESULTS_CONFIG = {
    "results_dir": "results",
    "models_dir": "saved_models",
}
