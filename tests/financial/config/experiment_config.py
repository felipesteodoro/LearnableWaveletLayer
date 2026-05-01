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
    # sequence_length=60 garante que kernel_size=8 seja válido até o nível 2:
    # nível 1 → L/2 = 30 amostras (>= kernel_size=8 OK)
    # nível 2 → L/4 = 15 amostras (>= kernel_size=8 OK)
    "sequence_length": 60,           # sliding window (trading days)
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
    "n_oos_windows": 2,       # walk-forward OOS windows (Fix 5)
}

# ---------------------------------------------------------------------------
# Wavelet
# ---------------------------------------------------------------------------

WAVELET_CONFIG = {
    "wavelet_type": "db4",
    # align="pad_to_first" evita aliasing espectral causado por interpolação bilinear
    # (tf.image.resize com bilinear introduz artefatos de alta frequência)
    "align": "pad_to_first",
    "wavelet_levels": 2,
}

LEARNED_WAVELET_CONFIG = {
    "levels": 2,
    # kernel_size=8 é compatível com seq_len=60:
    #   nível 1: 60//2=30 amostras > 8 coeficientes → overlap bem definido
    #   nível 2: 30//2=15 amostras > 8 coeficientes → mesma garantia
    # kernel_size=32 era problemático pois no nível 2 tínhamos ~7 amostras
    # com 32 coeficientes — quase tudo padding, sem informação de wavelet real.
    "kernel_size": 8,
    "wavelet_net_units": 32,
    "reg_energy": 1e-2,
    "reg_high_dc": 1e-2,
    "reg_smooth": 1e-3,
    # align="pad_to_first": alinha coeficientes de detalhe por zero-padding no final.
    # Preserva as posições temporais originais sem introduzir artefatos de interpolação.
    "align": "pad_to_first",
    # warm_start_db4=True: inicializa a rede wavelet para aproximar db4 antes de treinar,
    # dando um ponto de partida melhor do que inicialização aleatória.
    "warm_start_db4": True,
    # wavelet_levels exposto aqui para que _apply_wavelet_frontend possa lê-lo
    "wavelet_levels": 2,
}

# ---------------------------------------------------------------------------
# Deep Learning
# ---------------------------------------------------------------------------

DL_TRAINING_CONFIG = {
    # EPOCHS_OVERRIDE permite validação rápida: EPOCHS_OVERRIDE=3 python run.py
    "epochs": int(os.environ.get("EPOCHS_OVERRIDE", "100")),
    "batch_size": 64,
    "early_stopping_patience": 15,
    # ReduceLROnPlateau unificado para todas as arquiteturas:
    # garante que CNN, LSTM, CNN_LSTM e Transformer usem o mesmo scheduler,
    # tornando a comparação entre modos (raw/db4/learned) mais justa.
    "reduce_lr_patience": 7,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6,
    "learning_rate": 1e-3,
    # Projeção linear após o frontend wavelet para controlar capacidade:
    # raw=(N,L,F) vs learned_wavelet=(N,L,(levels+1)*F) — sem projeção,
    # o backbone vê dimensionalidades diferentes e tem capacidades distintas.
    "wavelet_projection_dim": 32,
    # sequence_length aqui para que pipeline.py possa importar via DL_TRAINING_CONFIG
    # sem depender do config dict passado pelo job (que não deve sobrescrever defaults).
    "sequence_length": FEATURE_CONFIG["sequence_length"],
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

# Grid axes para busca aleatória de hiperparâmetros DL.
# MAX_GRID_CONFIGS garante orçamento balanceado: cada arquitetura recebe
# o mesmo número de combinações avaliadas, independentemente do tamanho do grid.
# Sem esse limite, arquiteturas com grids menores são testadas exaustivamente
# enquanto as maiores são sub-amostradas de forma desigual.
DL_GRID_AXES = {
    "CNN": {
        "dropout_rate": [0.2, 0.3, 0.4],
        "l2_reg": [1e-4, 1e-3, 1e-2],
        "filters": [[32, 64, 128], [64, 128, 256]],
    },
    "LSTM": {
        "dropout_rate": [0.2, 0.3],
        "l2_reg": [1e-4, 1e-3],
        "units": [[64, 32], [128, 64]],
    },
    "CNN_LSTM": {
        "dropout_rate": [0.2, 0.3],
        "l2_reg": [1e-4, 1e-3],
    },
    "Transformer": {
        "dropout_rate": [0.1, 0.2],
        "num_heads": [2, 4],
        "ff_dim": [64, 128],
    },
}

# Número máximo de configurações avaliadas por arquitetura no random search.
# Valor 6 garante que: mesmo a arquitetura com grid 3×3×2=18 combinações
# avalia apenas 6 configs, igual às arquiteturas com grid menor (ex: 2×2=4).
MAX_GRID_CONFIGS = {
    "CNN": 6,
    "LSTM": 6,
    "CNN_LSTM": 4,
    "Transformer": 6,
}

# ---------------------------------------------------------------------------
# Machine Learning
# ---------------------------------------------------------------------------

ML_N_JOBS_OUTER = 4                                     # parallel (ticker × model)
ML_N_JOBS_MODEL = 2                                     # per-model thread count

# Hyperparameter grid for ML models — each list entry is a dict of overrides
# applied on top of ML_MODELS_CONFIG defaults.  6 configs per tree-based model,
# ordered from shallow/fast → deep/slow, kept equivalent across models:
#
#  lvl 1 — shallow,  fast  (n_est=100, depth=4/None, lr=0.10)
#  lvl 2 — shallow+  fast  (n_est=100, depth=6,      lr=0.10)
#  lvl 3 — medium,   mid   (n_est=200, depth=4/10,   lr=0.05)
#  lvl 4 — medium+   mid   (n_est=200, depth=6,      lr=0.05)
#  lvl 5 — deep,     slow  (n_est=300, depth=6,      lr=0.03)
#  lvl 6 — deep+reg, slow  (n_est=300, depth=8,      lr=0.01)
ML_PARAM_GRID = {
    "RandomForest": [
        # lvl 1
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 2},
        # lvl 2
        {"n_estimators": 100, "max_depth": 10,   "min_samples_leaf": 2},
        # lvl 3
        {"n_estimators": 150, "max_depth": None, "min_samples_leaf": 2},
        # lvl 4
        {"n_estimators": 150, "max_depth": 10,   "min_samples_leaf": 2},
        # lvl 5
        {"n_estimators": 200, "max_depth": 10,   "min_samples_leaf": 2},
        # lvl 6
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5},
        # lvl 7
        {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 2},
        # lvl 8
        {"n_estimators": 250, "max_depth": 15,   "min_samples_leaf": 5},
        # lvl 9
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
        # lvl 10
        {"n_estimators": 300, "max_depth": 15,   "min_samples_leaf": 5},
        # lvl 11
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2},
        # lvl 12
        {"n_estimators": 400, "max_depth": 15,   "min_samples_leaf": 5},
    ],
    "XGBoost": [
        # lvl 1
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10},
        # lvl 2
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.10},
        # lvl 3
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.08},
        # lvl 4
        {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08},
        # lvl 5
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        # lvl 6
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
        # lvl 7
        {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.03},
        # lvl 8
        {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.03},
        # lvl 9
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03},
        # lvl 10
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.02},
        # lvl 11
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.01},
        # lvl 12
        {"n_estimators": 400, "max_depth": 8, "learning_rate": 0.01},
    ],
    "LightGBM": [
        # lvl 1
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10, "num_leaves": 15},
        # lvl 2
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.10, "num_leaves": 31},
        # lvl 3
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.08, "num_leaves": 15},
        # lvl 4
        {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08, "num_leaves": 31},
        # lvl 5
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "num_leaves": 15},
        # lvl 6
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 31},
        # lvl 7
        {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 63},
        # lvl 8
        {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.03, "num_leaves": 63},
        # lvl 9
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 63},
        # lvl 10
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.02, "num_leaves": 127},
        # lvl 11
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.01, "num_leaves": 63},
        # lvl 12
        {"n_estimators": 400, "max_depth": 8, "learning_rate": 0.01, "num_leaves": 127},
    ],
    "CatBoost": [
        # lvl 1
        {"iterations": 100, "depth": 4, "learning_rate": 0.10},
        # lvl 2
        {"iterations": 100, "depth": 6, "learning_rate": 0.10},
        # lvl 3
        {"iterations": 150, "depth": 4, "learning_rate": 0.08},
        # lvl 4
        {"iterations": 150, "depth": 6, "learning_rate": 0.08},
        # lvl 5
        {"iterations": 200, "depth": 4, "learning_rate": 0.05},
        # lvl 6
        {"iterations": 200, "depth": 6, "learning_rate": 0.05},
        # lvl 7
        {"iterations": 250, "depth": 6, "learning_rate": 0.03},
        # lvl 8
        {"iterations": 250, "depth": 8, "learning_rate": 0.03},
        # lvl 9
        {"iterations": 300, "depth": 6, "learning_rate": 0.03},
        # lvl 10
        {"iterations": 300, "depth": 8, "learning_rate": 0.02},
        # lvl 11
        {"iterations": 400, "depth": 6, "learning_rate": 0.01},
        # lvl 12
        {"iterations": 400, "depth": 8, "learning_rate": 0.01},
    ],
    # Stacking: 4 → 8 configs
    "Stacking": [
        {"base_n_estimators":  50},
        {"base_n_estimators":  75},
        {"base_n_estimators": 100},
        {"base_n_estimators": 125},
        {"base_n_estimators": 150},
        {"base_n_estimators": 175},
        {"base_n_estimators": 200},
        {"base_n_estimators": 250},
    ],
}

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
