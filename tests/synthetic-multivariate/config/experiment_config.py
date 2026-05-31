"""
Configurações centralizadas para todos os experimentos do pipeline.

Todos os hiperparâmetros de ML e DL estão centralizados aqui.
Nenhum parâmetro deve ser definido diretamente nos notebooks.

Variáveis de ambiente opcionais (para execução multi-GPU / teste rápido):
    GPU_ID           – id da GPU a usar (ex: "0" ou "1"). Se definido, seta
                       CUDA_VISIBLE_DEVICES **antes** de importar TensorFlow.
    EPOCHS_OVERRIDE  – substitui o número de epochs (ex: "1" para smoke test).
    MAX_GRID_CONFIGS – limita cada grid search a N configurações (0 = sem limite).
"""
import os
import itertools
from copy import deepcopy
from pathlib import Path

# ============================================================================
# CONTROLE VIA VARIÁVEIS DE AMBIENTE (multi-GPU / teste rápido)
# ============================================================================
GPU_ID = os.environ.get('GPU_ID', '')              # "" = usar todas as GPUs
EPOCHS_OVERRIDE = int(os.environ.get('EPOCHS_OVERRIDE', 0))   # 0 = sem override
MAX_GRID_CONFIGS = int(os.environ.get('MAX_GRID_CONFIGS', 0))  # 0 = sem limite

# Selecionar GPU **antes** que TensorFlow seja importado (quem importa este
# módulo normalmente o faz antes de `import tensorflow`).
if GPU_ID:
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

# ============================================================================
# CAMINHOS DO PROJETO
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# Criar diretórios se não existirem
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SEMENTE GLOBAL PARA REPRODUTIBILIDADE
# ============================================================================
SEED = 42

# ============================================================================
# CONFIGURAÇÃO GLOBAL DE PARALELISMO
# ============================================================================
N_JOBS = max(1, os.cpu_count() // 2)   # Metade dos cores — conservador para não travar o SO
N_JOBS_QUARTER = max(1, os.cpu_count() // 4)  # 1/4 dos cores — para modelos que já paralelizam internamente

# ============================================================================
# CONFIGURAÇÕES DO SINAL SINTÉTICO MULTIVARIADO (HETEROGÊNEO POR CANAL)
# ============================================================================
# Parâmetros globais do dataset (independentes de canal).
SYNTHETIC_SIGNAL_CONFIG = {
    "n_samples": 25000,            # Número de timesteps (por canal)
    "sequence_length": 256,        # Comprimento de cada janela
}

# Perfis ESPECÍFICOS por canal — cada um vira **kwargs em SyntheticSignalGenerator.
# Objetivo: cada canal tem natureza espectral distinta para justificar filtros
# wavelet diferentes por canal (per-channel DWT na LWT).
CHANNEL_PROFILES_HETEROGENEOUS = [
    # Canal 0 — BAIXA FREQUÊNCIA (tendência longa + harmônicos lentos).
    # Banda dominante: aproximação A_L (low-pass largo).
    {
        "trend_degree":      3,
        "n_harmonics":       2,
        "base_frequency":    0.002,
        "regime_changes":    1,
        "noise_level":       0.15,
        "spike_probability": 0.0,
        "spike_magnitude":   0.0,
    },
    # Canal 1 — MÉDIA FREQUÊNCIA (vários harmônicos médios).
    # Banda dominante: D2/D3 (passa-banda intermediária).
    {
        "trend_degree":      1,
        "n_harmonics":       8,
        "base_frequency":    0.01,
        "regime_changes":    0,
        "noise_level":       0.25,
        "spike_probability": 0.0,
        "spike_magnitude":   0.0,
    },
    # Canal 2 — ALTA FREQUÊNCIA + ruído (oscilações rápidas).
    # Banda dominante: D1 (high-pass agudo).
    {
        "trend_degree":      0,
        "n_harmonics":       3,
        "base_frequency":    0.08,
        "regime_changes":    0,
        "noise_level":       0.4,
        "spike_probability": 0.0,
        "spike_magnitude":   0.0,
    },
    # Canal 3 — TRANSIENTES esparsos (impulsos curtos + spikes).
    # Beneficia-se de wavelet de suporte compacto (kernel pequeno, tipo Haar/db2).
    {
        "trend_degree":      0,
        "n_harmonics":       1,
        "base_frequency":    0.005,
        "regime_changes":    0,
        "noise_level":       0.1,
        "spike_probability": 0.05,
        "spike_magnitude":   4.0,
    },
    # Canal 4 — NÃO-ESTACIONÁRIO (chirp + muitas mudanças de regime).
    # Beneficia-se de wavelets que não acoplam demais entre níveis.
    {
        "trend_degree":      1,
        "n_harmonics":       3,
        "base_frequency":    0.02,
        "regime_changes":    6,
        "noise_level":       0.3,
        "spike_probability": 0.0,
        "spike_magnitude":   0.0,
    },
]

# Configuração específica do experimento multivariado (5 canais → 1 alvo).
# Target MULTI-ESCALA TEMPORAL (FFT bandpass, bandas ALINHADAS com oitavas wavelet):
#   Wavelet 2-level dyadic subbands: A2=[0,0.125], D2=[0.125,0.25], D1=[0.25,0.5]
#   y = Σ_c w_c * band_feature(ch_c, octave_c)
#       + alpha * cross_band_corr(ch0, ch4, A2)
#       + beta  * D1_energy(ch2) * D1_var(ch3)
#       + gamma * A2_slope(ch0) * D1_var(ch3)
# Cada canal contribui via sua OITAVA WAVELET → LWT decompõe naturalmente.
MULTIVARIATE_CONFIG = {
    "n_channels": 5,
    "channel_seeds": [42, 43, 44, 45, 46],
    "channel_profiles": CHANNEL_PROFILES_HETEROGENEOUS,
    "target_band_weights": [0.25, 0.25, 0.20, 0.15, 0.15],
    "target_band_specs": [
        (0.0,   0.125),   # ch0: A2 octave — low-freq energy (trend)
        (0.125, 0.250),   # ch1: D2 octave — mid-freq energy (harmonics)
        (0.250, 0.500),   # ch2: D1 octave — high-freq energy
        (0.250, 0.500),   # ch3: D1 octave — high-freq variance (transients)
        (0.0,   0.125),   # ch4: A2 octave — low-freq trend slope
    ],
    "target_cross_alpha": 0.3,
    "target_cross_beta":  0.2,
    "target_trend_gamma": 0.15,
    "fs": 1.0,
    "horizon": 1,
    "stride":  1,
}

# ============================================================================
# CONFIGURAÇÕES DE WAVELETS
# ============================================================================
WAVELET_CONFIG = {
    "wavelet_type": "db2",         # Wavelet para extração de features
    "decomposition_level": 2,       # Níveis de decomposição
    "mode": "symmetric",            # Modo de extensão de borda
}

LEARNED_WAVELET_CONFIG = {
    "levels": 2,
    "kernel_size": 8,
    "wavelet_net_units": 32,
    "reg_energy": 1e-2,
    "reg_high_dc": 1e-2,
    "reg_smooth": 1e-3,
    "align": "pad_to_first",
    "warm_start_db4": False,
}

# ============================================================================
# CONFIGURAÇÕES DE VALIDAÇÃO
# ============================================================================
VALIDATION_CONFIG = {
    "test_size": 0.2,              # 20% para teste
    "val_size": 0.15,              # 15% do treino para validação
    "n_folds": 5,                  # K-Fold cross-validation
    "shuffle": False,              # Não embaralhar para séries temporais
    "gap": 10,                     # Gap para purged k-fold
}

# ============================================================================
# CONFIGURAÇÕES DE FEATURE EXTRACTION (ML)
# ============================================================================
ML_FEATURE_CONFIG = {
    "variance_threshold": 1e-8,     # Limiar para remover features constantes
    "mi_subsample": 5000,           # Amostras para cálculo de Mutual Information
    "mi_n_neighbors": 5,            # Vizinhos para mutual_info_regression
    "mi_top_k": 15,                 # Quantas top features exibir
}

# ============================================================================
# CONFIGURAÇÕES DE BUSCA DE HIPERPARÂMETROS (ML)
# ============================================================================
ML_SEARCH_CONFIG = {
    "cv_splits": 3,                            # Número de splits para TimeSeriesSplit
    "scoring": "neg_mean_squared_error",       # Métrica para otimização
    "n_jobs": 16,                               # Jobs paralelos para RandomizedSearchCV (limitado)
    "verbose": 1,                              # Verbosidade do RandomizedSearchCV
    "random_state": SEED,                      # Semente para reprodutibilidade
}

# ============================================================================
# CONFIGURAÇÕES DE MODELOS ML CLÁSSICOS (RandomizedSearchCV)
#
# Cada modelo tem:
#   - model_kwargs: parâmetros fixos do modelo (n_jobs, random_state, etc.)
#   - param_dist: distribuições scipy para RandomizedSearchCV
#   - n_iter: número de iterações do RandomizedSearchCV
# ============================================================================
ML_MODELS_CONFIG = {
    "LinearSVR": {
        "model_kwargs": {"max_iter": 10000, "random_state": 42},
        "param_dist": {
            "C": ("loguniform", 1e-2, 1e3),
            "epsilon": ("uniform", 0, 0.5),
            "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        },
        "n_iter": 15,
    },
    "SGDRegressor": {
        "model_kwargs": {"max_iter": 5000, "random_state": 42},
        "param_dist": {
            "loss": ["squared_error", "huber", "epsilon_insensitive"],
            "alpha": ("loguniform", 1e-6, 1e-1),
            "epsilon": ("uniform", 0.01, 0.5),
            "penalty": ["l1", "l2", "elasticnet"],
            "l1_ratio": ("uniform", 0.0, 1.0),
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        },
        "n_iter": 20,
    },
    "ElasticNet": {
        "model_kwargs": {"max_iter": 10000, "random_state": 42},
        "param_dist": {
            "alpha": ("loguniform", 1e-5, 10),
            "l1_ratio": ("uniform", 0.0, 1.0),
        },
        "n_iter": 15,
    },
    "RandomForest": {
        "model_kwargs": {"random_state": 42, "n_jobs": 4},
        "param_dist": {
            "n_estimators": ("randint", 50, 200),
            "max_depth": [5, 10, 20],
            "min_samples_split": ("randint", 2, 10),
            "min_samples_leaf": ("randint", 1, 5),
            "max_features": ["sqrt", "log2", 0.5],
        },
        "n_iter": 15,
    },
    "XGBoost": {
        "model_kwargs": {"random_state": 42, "n_jobs": 4, "verbosity": 0},
        "param_dist": {
            "n_estimators": ("randint", 50, 200),
            "max_depth": ("randint", 3, 8),
            "learning_rate": ("loguniform", 1e-3, 0.3),
            "subsample": ("uniform", 0.6, 0.4),
            "colsample_bytree": ("uniform", 0.6, 0.4),
            "reg_alpha": ("loguniform", 1e-4, 10),
            "reg_lambda": ("loguniform", 1e-4, 10),
            "min_child_weight": ("randint", 1, 10),
        },
        "n_iter": 15,
    },
    "LightGBM": {
        "model_kwargs": {"random_state": 42, "n_jobs": 4, "verbose": -1},
        "param_dist": {
            "n_estimators": ("randint", 50, 200),
            "max_depth": ("randint", 3, 8),
            "learning_rate": ("loguniform", 1e-3, 0.3),
            "num_leaves": ("randint", 20, 50),
            "subsample": ("uniform", 0.6, 0.4),
            "colsample_bytree": ("uniform", 0.6, 0.4),
            "reg_alpha": ("loguniform", 1e-4, 10),
            "reg_lambda": ("loguniform", 1e-4, 10),
            "min_child_samples": ("randint", 5, 30),
        },
        "n_iter": 15,
    },
    "CatBoost": {
        "model_kwargs": {"random_seed": 42, "thread_count": 4, "verbose": 0},
        "param_dist": {
            "iterations": ("randint", 100, 300),
            "depth": ("randint", 4, 8),
            "learning_rate": ("loguniform", 1e-3, 0.3),
            "l2_leaf_reg": ("loguniform", 1e-2, 10),
            "bagging_temperature": ("uniform", 0, 2.0),
            "random_strength": ("uniform", 0.5, 2.0),
        },
        "n_iter": 15,
    },
    "Stacking": {
        # Configuração especial — usa os melhores modelos como base learners
        "base_learners": ["RandomForest", "XGBoost", "LightGBM", "CatBoost"],
        "meta_learner": "RidgeCV",
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "base_n_jobs": 4,             # n_jobs para cada base learner no stacking
        "model_kwargs": {"n_jobs": 4, "passthrough": False},
    },
}


def build_param_dist(param_spec: dict) -> dict:
    """
    Converte especificações de distribuição do config em objetos scipy.

    Suporta:
      - ("loguniform", low, high) → scipy.stats.loguniform(low, high)
      - ("uniform", loc, scale)   → scipy.stats.uniform(loc, scale)
      - ("randint", low, high)    → scipy.stats.randint(low, high)
      - list                      → mantém como lista (categorical)
    """
    from scipy.stats import loguniform, uniform, randint

    dist_map = {
        "loguniform": lambda a, b: loguniform(a, b),
        "uniform": lambda a, b: uniform(a, b),
        "randint": lambda a, b: randint(a, b),
    }

    out = {}
    for key, val in param_spec.items():
        if isinstance(val, tuple) and len(val) == 3 and val[0] in dist_map:
            out[key] = dist_map[val[0]](val[1], val[2])
        else:
            out[key] = val
    return out


# ============================================================================
# CONFIGURAÇÕES DE MODELOS DEEP LEARNING
# ============================================================================
DL_TRAINING_CONFIG = {
    "epochs": 50,  # Fixado em 50 para exploração de hiper-parâmetros
    "batch_size": 256,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 7,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6,
    "verbose": 1,
}

# ---- Configuração base (default) de cada arquitetura DL ----
DL_MODELS_CONFIG = {
    "CNN": {
        "filters": [64, 128, 256],
        "kernel_sizes": [7, 5, 3],
        "pool_sizes": [2, 2, 2],
        "dense_units": [128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "l2_reg": 0.001,
    },
    "LSTM": {
        "units": [128, 64],
        "dropout_rate": 0.3,
        "recurrent_dropout": 0.0,   # 0.0 para habilitar kernel CuDNN
        "learning_rate": 0.001,
        "l2_reg": 0.001,
    },
    "CNN_LSTM": {
        "cnn_filters": [64, 128],
        "cnn_kernel_sizes": [5, 3],
        "lstm_units": [100, 50],
        "dense_units": [64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "l2_reg": 0.001,
    },
    "Transformer": {
        "head_size": 64,
        "num_heads": 4,
        "ff_dim": 128,
        "num_transformer_blocks": 2,
        "mlp_units": [128, 64],
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "l2_reg": 0.001,
        "use_warmup": True,
        "warmup_steps": 500,
    },
    "MLP": {
        "mlp_units": [256, 128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "l2_reg": 0.001,
    },
}

# ---- Eixos de variação do Grid para cada arquitetura DL ----
# Redução de 70%: mantendo 30% do grid original para exploração eficiente.
DL_GRID_AXES = {
    "CNN": {
        "dropout_rate": [0.2, 0.3, 0.4],
        "l2_reg": [1e-4, 1e-3, 1e-2],
        "filters": [[32, 64, 128], [64, 128, 256]],
        "kernel_sizes": [[7, 5, 3], [5, 3, 3]],
    },
    # LSTM: 2×2×4 = 16 configs per raw/db4 (32 total) + 24 wavelet = 56 total
    "LSTM": {
        "dropout_rate":  [0.2, 0.3],
        "l2_reg":        [1e-4, 1e-2],
        "units":         [[96, 48], [128, 64], [192, 96], [256, 128]],
    },
    # CNN_LSTM: 2×2 = 4 configs per raw/db4 (8 total) + 16 wavelet = 24 total
    "CNN_LSTM": {
        "dropout_rate": [0.2, 0.3],
        "lstm_units":   [[64, 32], [128, 64]],
    },
    # Transformer: 2×2×4 = 16 configs per raw/db4 (32 total) + 64 wavelet = 96 total
    "Transformer": {
        "dropout_rate": [0.15, 0.25],
        "num_heads":    [2, 4],
        "patch_size":   [1, 2, 4, 8],
    },
    # MLP: 2×2×2 = 8 configs per raw/db4 (16 total) + 32 wavelet = 48 total
    "MLP": {
        "dropout_rate": [0.2, 0.3],
        "l2_reg":       [1e-4, 1e-2],
        "mlp_units":    [[128, 64], [256, 128]],
    },
}


def generate_dl_grid(model_name: str) -> list:
    """
    Gera lista de dicts com todas as combinações do grid para um modelo DL.

    Cada dict contém apenas os parâmetros que variam; devem ser mesclados
    com DL_MODELS_CONFIG[model_name] no notebook via {**base, **variation}.

    Returns:
        Lista de dicts, um por combinação do grid.
    """
    axes = DL_GRID_AXES.get(model_name, {})
    if not axes:
        return [{}]  # sem grid — uma única execução com defaults

    keys = list(axes.keys())
    values = [axes[k] for k in keys]

    grid = []
    for combo in itertools.product(*values):
        grid.append(dict(zip(keys, combo)))

    if MAX_GRID_CONFIGS > 0:
        grid = grid[:MAX_GRID_CONFIGS]
    return grid


# ============================================================================
# CONFIGURAÇÕES ESPECÍFICAS PARA LEARNED WAVELETS DL
# ============================================================================
# Overrides/adições por modelo learned wavelet (mesclados com DL_MODELS_CONFIG)
LEARNED_WAVELET_MODELS_CONFIG = {
    "CNN": {
        "filters": [64, 128, 256],
        "kernel_sizes": [7, 5, 3],
        "pool_sizes": [2, 2, 2],
        "dense_units": [96, 48],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "l2_reg": 0.001,
    },
    "LSTM": {
        "units": [128, 64],
        "dropout_rate": 0.3,
        "recurrent_dropout": 0.0,
        "learning_rate": 0.001,
        "l2_reg": 0.001,
    },
    "CNN_LSTM": {
        "cnn_filters": [64, 128],
        "cnn_kernel_sizes": [5, 3],
        "lstm_units": [100, 50],
        "dense_units": [64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "l2_reg": 0.001,
    },
    "Transformer": {
        "head_size": 64,
        "num_heads": 4,
        "ff_dim": 128,
        "num_transformer_blocks": 2,
        "mlp_units": [96, 48],
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "l2_reg": 0.001,
        "use_warmup": True,
        "warmup_steps": 500,
    },
    "MLP": {
        "mlp_units": [256, 128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "l2_reg": 0.001,
    },
}

# Grid axes específicos para learned wavelets (30% do original para eficiência).
LEARNED_WAVELET_GRID_AXES = {
    "CNN": {
        "kernel_size":            [4, 8, 16],
        "levels":                 [2, 3],
        "reg_energy":             [0.0, 1e-3, 1e-2],
        "reg_high_dc":            [0.0, 1e-3, 1e-2],
        "wavelet_projection_dim": [0, 16, 32],
        "filters":                [[32, 64, 128], [64, 128, 256]],
        "kernel_sizes":           [[7, 5, 3], [5, 3, 3]],
    },
    # LSTM: 3×2×2 = 12 configs por modo LWT (24 total)
    "LSTM": {
        "kernel_size": [4, 8, 16],
        "levels":      [2, 3],
        "reg_energy":  [0.0, 1e-2],
    },
    # CNN_LSTM: 4×2 = 8 configs por modo LWT (16 total)
    "CNN_LSTM": {
        "kernel_size": [4, 8, 12, 16],
        "levels":      [2, 3],
    },
    # Transformer: 4×3×2×2 = 48 configs por modo LWT (96 total)
    "Transformer": {
        "kernel_size": [4, 8, 12, 16],
        "levels":      [2, 3, 4],
        "patch_size":  [1, 8],
        "reg_energy":  [0.0, 1e-2],
    },
    # MLP: 4×2×2 = 16 configs por modo LWT (32 total)
    "MLP": {
        "kernel_size": [4, 8, 12, 16],
        "levels":      [2, 3],
        "reg_energy":  [0.0, 1e-2],
    },
}


def generate_learned_wavelet_grid(model_name: str) -> list:
    """Gera grid para modelos learned wavelet."""
    axes = LEARNED_WAVELET_GRID_AXES.get(model_name, {})
    if not axes:
        return [{}]
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    grid = []
    for combo in itertools.product(*values):
        grid.append(dict(zip(keys, combo)))

    if MAX_GRID_CONFIGS > 0:
        grid = grid[:MAX_GRID_CONFIGS]
    return grid


# ============================================================================
# CONFIGURAÇÕES DE OPTUNA (Otimização de Hiperparâmetros)
# ============================================================================
OPTUNA_CONFIG = {
    "n_trials": 50,                # Número de trials por modelo
    "timeout": 3600,               # Timeout em segundos (1 hora)
    "n_jobs": 1,                   # Jobs paralelos
    "direction": "minimize",       # Minimizar erro
    "sampler": "TPESampler",       # Sampler do Optuna
    "pruner": "MedianPruner",      # Pruner para early stopping
}

# ============================================================================
# MÉTRICAS DE AVALIAÇÃO
# ============================================================================
# ============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO (ML)
# ============================================================================
ML_VIS_CONFIG = {
    "prediction_n_samples": 500,     # Amostras a exibir no gráfico de predições
    "feature_importance_top_k": 25,  # Top-K features no gráfico de importância
}

# ============================================================================
# MÉTRICAS DE AVALIAÇÃO
# ============================================================================
METRICS = [
    "mse",          # Mean Squared Error
    "rmse",         # Root Mean Squared Error
    "mae",          # Mean Absolute Error
    "mape",         # Mean Absolute Percentage Error
    "r2",           # R² Score
    "explained_var", # Explained Variance
]

# ============================================================================
# EXPERIMENTOS A EXECUTAR
# ============================================================================
EXPERIMENTS = {
    # ML Clássico com Wavelets
    "ml_wavelet": [
        {"name": "Wavelet_DB2_SVM", "feature_extractor": "wavelet", "model": "SVM"},
        {"name": "Wavelet_DB2_RF", "feature_extractor": "wavelet", "model": "RandomForest"},
        {"name": "Wavelet_DB2_XGB", "feature_extractor": "wavelet", "model": "XGBoost"},
        {"name": "Wavelet_DB2_LGBM", "feature_extractor": "wavelet", "model": "LightGBM"},
    ],
    
    # DL com sinal raw
    "dl_raw": [
        {"name": "Raw_CNN", "feature_extractor": "raw", "model": "CNN"},
        {"name": "Raw_LSTM", "feature_extractor": "raw", "model": "LSTM"},
        {"name": "Raw_CNN_LSTM", "feature_extractor": "raw", "model": "CNN_LSTM"},
        {"name": "Raw_Transformer", "feature_extractor": "raw", "model": "Transformer"},
    ],
    
    # DL com Wavelets fixas (db2)
    "dl_wavelet": [
        {"name": "Wavelet_DB2_CNN", "feature_extractor": "wavelet", "model": "CNN"},
        {"name": "Wavelet_DB2_LSTM", "feature_extractor": "wavelet", "model": "LSTM"},
        {"name": "Wavelet_DB2_CNN_LSTM", "feature_extractor": "wavelet", "model": "CNN_LSTM"},
        {"name": "Wavelet_DB2_Transformer", "feature_extractor": "wavelet", "model": "Transformer"},
    ],
    
    # DL com Learned Wavelets
    "dl_learned_wavelet": [
        {"name": "LearnedWavelet_CNN", "feature_extractor": "learned_wavelet", "model": "CNN"},
        {"name": "LearnedWavelet_LSTM", "feature_extractor": "learned_wavelet", "model": "LSTM"},
        {"name": "LearnedWavelet_CNN_LSTM", "feature_extractor": "learned_wavelet", "model": "CNN_LSTM"},
        {"name": "LearnedWavelet_Transformer", "feature_extractor": "learned_wavelet", "model": "Transformer"},
    ],
}
