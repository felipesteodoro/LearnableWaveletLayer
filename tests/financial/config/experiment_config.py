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

DL_MODELS     = ["CNN", "LSTM", "CNN_LSTM", "MLP", "Transformer"]
ML_MODELS     = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "Stacking"]
MODES         = ["raw", "db4", "learned_wavelet"]
FEATURE_MODES = ["features", "ohlcv"]

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_CONFIG = {
    # sequence_length=40 garante que kernel_size=8 seja válido até o nível 2:
    # nível 1 → L/2 = 20 amostras (>= kernel_size=8 OK, razão 2.5×)
    # nível 2 → L/4 = 10 amostras (>= kernel_size=8 OK, razão 1.25× → 2.5× vs seq_len=20)
    "sequence_length": 40,           # sliding window (trading days)
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
    # pt_sl=[2.0,2.0] produz distribuição quase balanceada: ~31% sell / 29% hold / 40% buy
    # (imbalance 1.4x vs 3.1x com [1.5,1.5]). Barreiras mais afastadas também são mais
    # alinhadas com movimentos significativos de médio prazo (2× vol = ~2 ATR diários).
    "pt_sl": [2.0, 2.0],    # take-profit / stop-loss multipliers of daily EWM vol
    "time_horizon": 10,      # max holding period in trading days
    "min_ret": 0.001,        # minimum return threshold to count a barrier
    "vol_span": 100,         # EWM span for daily volatility estimate
    # Label mapping para n_classes=3: {-1 → 0 (sell), 0 → 1 (hold), 1 → 2 (buy)}
    # Label mapping para n_classes=2 (meta-labeling): mantém apenas barreiras de preço
    #   {sell=-1 → 0 (down/SL), buy=+1 → 1 (up/TP)}; holds descartados.
    "n_classes": 2,
    "drop_holds": True,   # descarta label 'hold' para meta-labeling binário
}

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

VALIDATION_CONFIG = {
    "n_folds": 5,
    "embargo_days": 10,       # trading days embargo between train and test
    "val_split": 0.15,        # fraction of training fold used as validation
    "test_years": 6,          # last N years reserved as out-of-sample test
    "n_oos_windows": 6,       # walk-forward OOS windows (Fix 5)
    "oos_protocol": "rolling",    # expanding | rolling
    "oos_block_years": 1.0,   # retraining cadence for forward OOS blocks
    "rolling_train_years": 3.0,  # janela de treino para oos_protocol="rolling"
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
    # kernel_size=8 é compatível com seq_len=40:
    #   nível 1: 40//2=20 amostras > 8 coeficientes → razão 2.5×, overlap bem definido
    #   nível 2: 20//2=10 amostras > 8 coeficientes → razão 1.25× (mínimo aceitável)
    # Benefício vs kernel_size=4: ~4 parâmetros livres no passa-baixa (vs ~2),
    # espaço real para o filtro aprendido divergir do warm-start db4.
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
    "early_stopping_patience": 25,  # era 15; aumentado para permitir convergência com menor regularização
    # ReduceLROnPlateau unificado para todas as arquiteturas:
    # garante que CNN, LSTM, CNN_LSTM e Transformer usem o mesmo scheduler,
    # tornando a comparação entre modos (raw/db4/learned) mais justa.
    "reduce_lr_patience": 12,  # era 7
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6,
    "learning_rate": 1e-3,
    # Projeção linear após o frontend wavelet para controlar capacidade:
    # raw=(N,L,F) vs learned_wavelet=(N,L,(levels+1)*F) — sem projeção,
    # o backbone vê dimensionalidades diferentes e tem capacidades distintas.
    "wavelet_projection_dim": 16,
    # sequence_length aqui para que pipeline.py possa importar via DL_TRAINING_CONFIG
    # sem depender do config dict passado pelo job (que não deve sobrescrever defaults).
    "sequence_length": FEATURE_CONFIG["sequence_length"],
}

DL_MODELS_CONFIG = {
    "CNN": {
        "filters": [32, 64, 128],
        "cnn_kernel_size": 3,   # lido por _cnn_backbone via cfg.get("cnn_kernel_size", 3)
        "pool_sizes": None,     # sem MaxPooling (GlobalAveragePooling1D ao final)
        "dense_units": [64],    # cabeça densa antes da saída
        "dropout_rate": 0.1,   # era 0.3; reduzido para combater underfitting (val_loss > log(3))
        "l2_reg": 1e-4,        # era 1e-3
    },
    "LSTM": {
        "units": [64, 32],
        "dense_units": [32],    # cabeça densa antes da saída
        "dropout_rate": 0.1,   # era 0.3
        "recurrent_dropout": 0.0,  # era 0.1
        "l2_reg": 1e-4,        # era 1e-3
    },
    "CNN_LSTM": {
        "filters": [32, 64],
        "lstm_units": [32, 16],
        "dense_units": [32],    # cabeça densa antes da saída
        "dropout_rate": 0.1,   # era 0.3
        "l2_reg": 1e-4,        # era 1e-3
    },
    "Transformer": {
        "num_heads": 4,
        "head_size": 16,
        "ff_dim": 64,
        "num_blocks": 1,
        "mlp_units": [32],      # cabeça MLP antes da saída
        "dropout_rate": 0.1,   # era 0.2
        "l2_reg": 1e-4,
    },
    "MLP": {
        "mlp_units": [128, 64, 32],
        "dropout_rate": 0.1,   # era 0.3
        "l2_reg": 1e-4,        # era 1e-3
    },
}

# Grid axes para busca aleatória de hiperparâmetros DL.
# MAX_GRID_CONFIGS garante orçamento balanceado: cada arquitetura recebe
# o mesmo número de combinações avaliadas, independentemente do tamanho do grid.
# Sem esse limite, arquiteturas com grids menores são testadas exaustivamente
# enquanto as maiores são sub-amostradas de forma desigual.
DL_GRID_AXES = {
    "CNN": {
        "dropout_rate": [0.05, 0.1, 0.2],   # reduzido: center em 0.1
        "l2_reg": [1e-5, 1e-4, 1e-3],       # reduzido: center em 1e-4
        "filters": [[32, 64, 128], [64, 128, 256]],
    },
    "LSTM": {
        "dropout_rate": [0.05, 0.1, 0.2],
        "l2_reg": [1e-5, 1e-4, 1e-3],
        "units": [[64, 32], [128, 64]],
    },
    "CNN_LSTM": {
        "dropout_rate": [0.05, 0.1, 0.2],
        "l2_reg": [1e-5, 1e-4, 1e-3],
    },
    "Transformer": {
        "dropout_rate": [0.05, 0.1, 0.2],
        "num_heads": [2, 4],
        "ff_dim": [64, 128],
    },
    "MLP": {
        "dropout_rate": [0.05, 0.1, 0.2],
        "l2_reg": [1e-5, 1e-4, 1e-3],
        "mlp_units": [[128, 64], [256, 128, 64]],
    },
}

# Número máximo de configurações avaliadas por arquitetura no random search.
# Valor 6 garante que: mesmo a arquitetura com grid 3×3×2=18 combinações
# avalia apenas 6 configs, igual às arquiteturas com grid menor (ex: 2×2=4).
MAX_GRID_CONFIGS = {
    "CNN": 6,
    "LSTM": 6,
    "CNN_LSTM": 4,
    "MLP": 6,
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
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 100, "max_depth": 10,   "min_samples_leaf": 2},
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5},
        {"n_estimators": 100, "max_depth": 10,   "min_samples_leaf": 5},
        {"n_estimators": 150, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 150, "max_depth": 10,   "min_samples_leaf": 2},
        {"n_estimators": 150, "max_depth": None, "min_samples_leaf": 5},
        {"n_estimators": 150, "max_depth": 15,   "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 10,   "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 15,   "min_samples_leaf": 5},
        {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 250, "max_depth": 15,   "min_samples_leaf": 2},
        {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 5},
        {"n_estimators": 250, "max_depth": 15,   "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 15,   "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 15,   "min_samples_leaf": 5},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 400, "max_depth": 15,   "min_samples_leaf": 2},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 5},
        {"n_estimators": 400, "max_depth": 15,   "min_samples_leaf": 5},
    ],
    "XGBoost": [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10},
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.10},
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.08},
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.08},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.08},
        {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.03},
        {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.03},
        {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.03},
        {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.02},
        {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.02},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03},
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.02},
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.02},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.01},
        {"n_estimators": 400, "max_depth": 8, "learning_rate": 0.01},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.01},
        {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.01},
    ],
    "LightGBM": [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10, "num_leaves": 15},
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.10, "num_leaves": 31},
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.08, "num_leaves": 15},
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.08, "num_leaves": 31},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.08, "num_leaves": 15},
        {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08, "num_leaves": 31},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "num_leaves": 15},
        {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 31},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "num_leaves": 15},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 31},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03, "num_leaves": 15},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 31},
        {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 63},
        {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.03, "num_leaves": 63},
        {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.02, "num_leaves": 63},
        {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.02, "num_leaves": 127},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 63},
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03, "num_leaves": 127},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.02, "num_leaves": 63},
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.02, "num_leaves": 127},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.01, "num_leaves": 63},
        {"n_estimators": 400, "max_depth": 8, "learning_rate": 0.01, "num_leaves": 127},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.01, "num_leaves": 63},
        {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.01, "num_leaves": 127},
    ],
    "CatBoost": [
        {"iterations": 100, "depth": 4, "learning_rate": 0.10},
        {"iterations": 100, "depth": 6, "learning_rate": 0.10},
        {"iterations": 100, "depth": 4, "learning_rate": 0.08},
        {"iterations": 100, "depth": 6, "learning_rate": 0.08},
        {"iterations": 150, "depth": 4, "learning_rate": 0.08},
        {"iterations": 150, "depth": 6, "learning_rate": 0.08},
        {"iterations": 150, "depth": 4, "learning_rate": 0.05},
        {"iterations": 150, "depth": 6, "learning_rate": 0.05},
        {"iterations": 200, "depth": 4, "learning_rate": 0.05},
        {"iterations": 200, "depth": 6, "learning_rate": 0.05},
        {"iterations": 200, "depth": 4, "learning_rate": 0.03},
        {"iterations": 200, "depth": 6, "learning_rate": 0.03},
        {"iterations": 250, "depth": 6, "learning_rate": 0.03},
        {"iterations": 250, "depth": 8, "learning_rate": 0.03},
        {"iterations": 250, "depth": 6, "learning_rate": 0.02},
        {"iterations": 250, "depth": 8, "learning_rate": 0.02},
        {"iterations": 300, "depth": 6, "learning_rate": 0.03},
        {"iterations": 300, "depth": 8, "learning_rate": 0.03},
        {"iterations": 300, "depth": 6, "learning_rate": 0.02},
        {"iterations": 300, "depth": 8, "learning_rate": 0.02},
        {"iterations": 400, "depth": 6, "learning_rate": 0.01},
        {"iterations": 400, "depth": 8, "learning_rate": 0.01},
        {"iterations": 500, "depth": 6, "learning_rate": 0.01},
        {"iterations": 500, "depth": 8, "learning_rate": 0.01},
    ],
    "Stacking": [
        {"base_n_estimators":  50},
        {"base_n_estimators":  75},
        {"base_n_estimators": 100},
        {"base_n_estimators": 125},
        {"base_n_estimators": 150},
        {"base_n_estimators": 175},
        {"base_n_estimators": 200},
        {"base_n_estimators": 225},
        {"base_n_estimators": 250},
        {"base_n_estimators": 275},
        {"base_n_estimators": 300},
        {"base_n_estimators": 325},
        {"base_n_estimators": 350},
        {"base_n_estimators": 375},
        {"base_n_estimators": 400},
        {"base_n_estimators": 450},
        {"base_n_estimators": 500},
        {"base_n_estimators": 550},
        {"base_n_estimators": 600},
        {"base_n_estimators": 650},
        {"base_n_estimators": 700},
        {"base_n_estimators": 750},
        {"base_n_estimators": 800},
        {"base_n_estimators": 900},
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
    "transaction_cost": 0.0005,  # 0.05% per trade (round-trip = 0.1%)
    "allow_short": True,         # True: sell=short; False: sell=flat
    # Taxa Selic anualizada usada como risk-free rate no Sharpe, Sortino e Alpha.
    # Valor constante representando média histórica brasileira; ajustar conforme
    # período analisado (Selic variou entre ~2% em 2020 e ~13.75% em 2022-25).
    "annual_risk_free": 0.1075,
    # Backtest multi-day: cada predição captura o retorno acumulado do dia de entrada
    # até t1 (quando a barreira foi atingida), em vez de apenas o retorno do dia seguinte.
    # Alinha a avaliação com o significado do label: "buy" significa que a barreira
    # superior foi atingida em algum momento nos próximos time_horizon dias.
    "use_event_returns": True,
}

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

RESULTS_CONFIG = {
    "results_dir": "results",
    "models_dir": "saved_models",
}
