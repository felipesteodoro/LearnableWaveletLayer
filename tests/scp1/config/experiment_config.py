"""
Configurações centralizadas para os experimentos SelfRegulationSCP1
(classificação BINÁRIA MULTIVARIADA — EEG, 6 canais, 896 timesteps, 2 classes).

Dataset UEA: sinais EEG de auto-regulação cortical (slow cortical potentials).
Classes: 'negativity' / 'positivity' (mapeadas para 0/1).

Variáveis de ambiente opcionais (multi-GPU / teste rápido):
    GPU_ID, EPOCHS_OVERRIDE, MAX_GRID_CONFIGS.
"""
import os
import itertools
from pathlib import Path

# ============================================================================
# CONTROLE VIA VARIÁVEIS DE AMBIENTE
# ============================================================================
GPU_ID = os.environ.get('GPU_ID', '')
EPOCHS_OVERRIDE = int(os.environ.get('EPOCHS_OVERRIDE', 0))
MAX_GRID_CONFIGS = int(os.environ.get('MAX_GRID_CONFIGS', 0))

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

for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SEMENTE GLOBAL
# ============================================================================
SEED = 42

# ============================================================================
# PARALELISMO
# ============================================================================
N_JOBS = max(1, os.cpu_count() // 2)
N_JOBS_QUARTER = max(1, os.cpu_count() // 4)

# ============================================================================
# CONFIGURAÇÃO DO DATASET SelfRegulationSCP1
# ----------------------------------------------------------------------------
# UEA archive: EEG (slow cortical potentials), 6 canais, comprimento 896,
# 2 classes ('negativity'/'positivity'). 268 treino / 293 teste, balanceado.
# Domínio canônico de wavelets (sinais EEG não-estacionários, multi-escala).
# ============================================================================
SCP1_CONFIG = {
    "dataset_name": "SelfRegulationSCP1",
    "n_classes": 2,
    "sequence_length": 896,
    "n_features": 6,                  # Multivariado: 6 canais EEG
    "class_labels": {0: "negativity", 1: "positivity"},
    "val_fraction": 0.15,             # 15% do treino para validação
    "random_seed": SEED,
}

# ============================================================================
# CONFIGURAÇÕES DE WAVELETS
# ============================================================================
WAVELET_CONFIG = {
    "wavelet_type": "db4",
    "decomposition_level": 3,
    "mode": "symmetric",
}

LEARNED_WAVELET_CONFIG = {
    "levels": 3,
    # kernel_size=16 é adequado para seq_len=896:
    #   nível 1: 896//2 = 448 >> 16 coeficientes
    #   nível 2: 448//2 = 224 >> 16
    #   nível 3: 224//2 = 112 >> 16 (suporte bem definido em todos)
    "kernel_size": 16,
    "wavelet_net_units": 32,
    "reg_energy": 1e-2,
    "reg_high_dc": 1e-2,
    "reg_smooth": 1e-3,
    "normalize_low": "l2",
    "align": "pad_to_first",
    "warm_start_db4": True,
}

# ============================================================================
# CONFIGURAÇÕES DE VALIDAÇÃO
# ============================================================================
VALIDATION_CONFIG = {
    "val_size": 0.15,
    "n_folds": 5,
    "shuffle": True,               # Stratified shuffle para classificação
    "stratify": True,
}

# ============================================================================
# CONFIGURAÇÕES DE FEATURE EXTRACTION (ML)
# ============================================================================
ML_FEATURE_CONFIG = {
    "variance_threshold": 1e-8,
    "mi_subsample": 3000,
    "mi_n_neighbors": 5,
    "mi_top_k": 15,
}

# ============================================================================
# CONFIGURAÇÕES DE BUSCA DE HIPERPARÂMETROS (ML)
# ----------------------------------------------------------------------------
# scoring="accuracy": binário balanceado (accuracy é apropriado; f1 também).
# ============================================================================
ML_SEARCH_CONFIG = {
    "cv_splits": 5,
    "scoring": "accuracy",
    "n_jobs": 16,
    "verbose": 1,
    "random_state": SEED,
}

# ============================================================================
# MODELOS ML CLÁSSICOS (Classificação binária)
# ============================================================================
ML_MODELS_CONFIG = {
    "LinearSVC": {
        "model_kwargs": {"max_iter": 10000, "random_state": SEED},
        "param_dist": {
            "estimator__C": ("loguniform", 1e-2, 1e3),
            "estimator__loss": ["hinge", "squared_hinge"],
        },
        "n_iter": 15,
    },
    "SGDClassifier": {
        "model_kwargs": {"max_iter": 5000, "random_state": SEED},
        "param_dist": {
            "loss": ["hinge", "log_loss", "modified_huber"],
            "alpha": ("loguniform", 1e-6, 1e-1),
            "penalty": ["l1", "l2", "elasticnet"],
            "l1_ratio": ("uniform", 0.0, 1.0),
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        },
        "n_iter": 20,
    },
    "LogisticRegression": {
        "model_kwargs": {"max_iter": 10000, "random_state": SEED, "solver": "saga"},
        "param_dist": {
            "C": ("loguniform", 1e-3, 1e3),
            "penalty": ["l1", "l2", "elasticnet"],
            "l1_ratio": ("uniform", 0.0, 1.0),
        },
        "n_iter": 15,
    },
    "RandomForest": {
        "model_kwargs": {"random_state": SEED, "n_jobs": 4},
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
        "model_kwargs": {"random_state": SEED, "n_jobs": 4, "verbosity": 0,
                         "use_label_encoder": False, "eval_metric": "logloss"},
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
        "model_kwargs": {"random_state": SEED, "n_jobs": 4, "verbose": -1},
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
        "model_kwargs": {"random_seed": SEED, "thread_count": 4, "verbose": 0},
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
        "base_learners": ["RandomForest", "XGBoost", "LightGBM", "CatBoost"],
        "meta_learner": "LogisticRegressionCV",
        "base_n_jobs": 4,
        "model_kwargs": {"n_jobs": 4, "passthrough": False},
    },
}


def build_param_dist(param_spec: dict) -> dict:
    """Converte especificações de distribuição do config em objetos scipy."""
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
# CONFIGURAÇÕES DL
# ============================================================================
DL_TRAINING_CONFIG = {
    "epochs": EPOCHS_OVERRIDE if EPOCHS_OVERRIDE > 0 else 100,
    "batch_size": 64,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 7,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6,
    "verbose": 1,
}

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

DL_GRID_AXES = {
    "CNN": {
        "dropout_rate": [0.2, 0.3, 0.4],
        "l2_reg": [1e-4, 1e-3, 1e-2],
        "filters": [[32, 64, 128], [64, 128, 256]],
        "kernel_sizes": [[7, 5, 3], [5, 3, 3]],
    },
    "LSTM": {
        "subsample_factor":  [8, 4],
        "dropout_rate":      [0.2, 0.3],
        "l2_reg":            [1e-4, 1e-3, 1e-2],
        "units":             [[64, 32], [128, 64], [256, 128], [256, 128, 64]],
        "bidirectional":     [False],
        "recurrent_dropout": [0.0, 0.15],
    },
    "CNN_LSTM": {
        "dropout_rate": [0.2, 0.3, 0.4],
        "l2_reg": [1e-4, 1e-3, 1e-2],
        "cnn_filters": [[32, 64], [64, 128]],
        "lstm_units": [[64, 32], [100, 50]],
    },
    "Transformer": {
        "patch_size": [16],
        "dropout_rate": [0.15, 0.2, 0.3],
        "num_heads": [2, 4],
        "ff_dim": [64, 128],
        "num_transformer_blocks": [2, 3],
        "l2_reg": [1e-4, 1e-3],
    },
    "MLP": {
        "dropout_rate": [0.2, 0.3, 0.4],
        "l2_reg": [1e-4, 1e-3, 1e-2],
        "mlp_units": [[256, 128, 64], [128, 64, 32]],
    },
}


def generate_dl_grid(model_name: str) -> list:
    """Gera lista de dicts com todas as combinações do grid para um modelo DL."""
    axes = DL_GRID_AXES.get(model_name, {})
    if not axes:
        return [{}]
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if MAX_GRID_CONFIGS > 0:
        grid = grid[:MAX_GRID_CONFIGS]
    return grid


# ============================================================================
# LEARNED WAVELETS DL
# ============================================================================
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

LEARNED_WAVELET_GRID_AXES = {
    "CNN": {
        "kernel_size": [8, 16, 32],
        "dropout_rate": [0.2, 0.3, 0.4],
        "l2_reg": [1e-4, 1e-3, 1e-2],
        "filters": [[32, 64, 128], [64, 128, 256]],
    },
    "LSTM": {
        "subsample_factor":  [8, 4],
        "kernel_size":       [8, 16, 32],
        "dropout_rate":      [0.2, 0.3],
        "l2_reg":            [1e-4, 1e-3, 1e-2],
        "units":             [[64, 32], [128, 64], [256, 128], [256, 128, 64]],
        "bidirectional":     [False],
        "recurrent_dropout": [0.0, 0.15],
    },
    "CNN_LSTM": {
        "kernel_size": [8, 16, 32],
        "dropout_rate": [0.2, 0.3, 0.4],
        "l2_reg": [1e-4, 1e-3, 1e-2],
        "cnn_filters": [[32, 64], [64, 128]],
        "lstm_units": [[64, 32], [100, 50]],
    },
    "Transformer": {
        "kernel_size": [8, 16, 32],
        "patch_size": [16],
        "dropout_rate": [0.15, 0.2, 0.3],
        "num_heads": [2, 4],
        "ff_dim": [64, 128],
        "num_transformer_blocks": [2, 3],
        "l2_reg": [1e-4, 1e-3],
    },
    "MLP": {
        "kernel_size": [8, 16, 32],
        "dropout_rate": [0.2, 0.3, 0.4],
        "l2_reg": [1e-4, 1e-3, 1e-2],
        "mlp_units": [[256, 128, 64], [128, 64, 32]],
    },
}


def generate_learned_wavelet_grid(model_name: str) -> list:
    """Gera grid para modelos learned wavelet."""
    axes = LEARNED_WAVELET_GRID_AXES.get(model_name, {})
    if not axes:
        return [{}]
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if MAX_GRID_CONFIGS > 0:
        grid = grid[:MAX_GRID_CONFIGS]
    return grid


# ============================================================================
# MÉTRICAS DE AVALIAÇÃO (binário)
# ============================================================================
METRICS = [
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "auc_roc",
]

# ============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO (ML)
# ============================================================================
ML_VIS_CONFIG = {
    "prediction_n_samples": 500,
    "feature_importance_top_k": 25,
}

# ============================================================================
# EXPERIMENTOS A EXECUTAR
# ============================================================================
EXPERIMENTS = {
    "ml_wavelet": [
        {"name": "Wavelet_DB4_SVC", "feature_extractor": "wavelet", "model": "SVC"},
        {"name": "Wavelet_DB4_RF", "feature_extractor": "wavelet", "model": "RandomForest"},
        {"name": "Wavelet_DB4_XGB", "feature_extractor": "wavelet", "model": "XGBoost"},
        {"name": "Wavelet_DB4_LGBM", "feature_extractor": "wavelet", "model": "LightGBM"},
    ],
    "dl_raw": [
        {"name": "Raw_CNN", "feature_extractor": "raw", "model": "CNN"},
        {"name": "Raw_LSTM", "feature_extractor": "raw", "model": "LSTM"},
        {"name": "Raw_CNN_LSTM", "feature_extractor": "raw", "model": "CNN_LSTM"},
        {"name": "Raw_Transformer", "feature_extractor": "raw", "model": "Transformer"},
    ],
    "dl_wavelet": [
        {"name": "Wavelet_DB4_CNN", "feature_extractor": "wavelet", "model": "CNN"},
        {"name": "Wavelet_DB4_LSTM", "feature_extractor": "wavelet", "model": "LSTM"},
        {"name": "Wavelet_DB4_CNN_LSTM", "feature_extractor": "wavelet", "model": "CNN_LSTM"},
        {"name": "Wavelet_DB4_Transformer", "feature_extractor": "wavelet", "model": "Transformer"},
    ],
    "dl_learned_wavelet": [
        {"name": "LearnedWavelet_CNN", "feature_extractor": "learned_wavelet", "model": "CNN"},
        {"name": "LearnedWavelet_LSTM", "feature_extractor": "learned_wavelet", "model": "LSTM"},
        {"name": "LearnedWavelet_CNN_LSTM", "feature_extractor": "learned_wavelet", "model": "CNN_LSTM"},
        {"name": "LearnedWavelet_Transformer", "feature_extractor": "learned_wavelet", "model": "Transformer"},
    ],
}
