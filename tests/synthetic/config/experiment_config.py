"""
Configurações centralizadas para todos os experimentos do pipeline.
"""
import os
from pathlib import Path

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
# CONFIGURAÇÕES DO SINAL SINTÉTICO
# ============================================================================
SYNTHETIC_SIGNAL_CONFIG = {
    # Tamanho e estrutura
    "n_samples": 50000,           # Número de amostras (grande para DL)
    "sequence_length": 256,        # Comprimento de cada sequência para modelos
    "n_features": 1,               # Univariado
    
    # Componentes do sinal
    "trend_degree": 2,             # Grau do polinômio de tendência
    "n_harmonics": 5,              # Número de harmônicos
    "base_frequency": 0.01,        # Frequência base
    
    # Ruído
    "noise_level": 0.3,            # Nível de ruído gaussiano
    "spike_probability": 0.02,     # Probabilidade de spikes
    "spike_magnitude": 3.0,        # Magnitude dos spikes
    
    # Não-estacionariedade
    "regime_changes": 3,           # Mudanças de regime
    
    # Semente para reprodutibilidade
    "random_seed": 42,
}

# ============================================================================
# CONFIGURAÇÕES DE WAVELETS
# ============================================================================
WAVELET_CONFIG = {
    "wavelet_type": "db2",         # Wavelet para extração de features
    "decomposition_level": 4,       # Níveis de decomposição
    "mode": "symmetric",            # Modo de extensão de borda
}

LEARNED_WAVELET_CONFIG = {
    "levels": 3,
    "kernel_size": 32,
    "wavelet_net_units": 32,
    "reg_energy": 1e-2,
    "reg_high_dc": 1e-2,
    "reg_smooth": 1e-3,
    "normalize_low": "sum1",
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
# CONFIGURAÇÕES DE MODELOS ML CLÁSSICOS
# ============================================================================
ML_MODELS_CONFIG = {
    "SVM": {
        "param_grid": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.1, 0.01],
            "kernel": ["rbf", "poly"],
            "epsilon": [0.01, 0.1, 0.2],
        }
    },
    "RandomForest": {
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    },
    "XGBoost": {
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }
    },
    "LightGBM": {
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 50, 100],
            "subsample": [0.8, 0.9, 1.0],
        }
    },
}

# ============================================================================
# CONFIGURAÇÕES DE MODELOS DEEP LEARNING
# ============================================================================
DL_TRAINING_CONFIG = {
    "epochs": 100,
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
        "recurrent_dropout": 0.2,
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
    },
}

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
        {"name": "LearnedWavelet_Transformer", "feature_extractor": "learned_wavelet", "model": "Transformer"},
    ],
}
