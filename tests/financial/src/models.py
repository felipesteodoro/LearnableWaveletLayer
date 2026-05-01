"""
DL model factory for 3-class financial classification.
Mirrors tests/synthetic/src/models.py — same backbones, softmax output.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Project root on path so models/ is importable
_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
except ImportError:
    raise ImportError("TensorFlow not found. Install tensorflow[and-cuda].")

# Shared DL utilities (centralised in models/dl_utils.py)
from models.dl_utils import SinusoidalPositionalEncoding, TransformerBlock  # noqa: E402


# ---------------------------------------------------------------------------
# Backbone builders
# ---------------------------------------------------------------------------

def _cnn_backbone(x, cfg: dict):
    l2 = regularizers.l2(cfg.get("l2_reg", 1e-3))
    dr = cfg.get("dropout_rate", 0.3)
    for f in cfg.get("filters", [64, 128, 256]):
        x = layers.Conv1D(f, kernel_size=cfg.get("cnn_kernel_size", 3),
                          padding="same", activation="relu",
                          kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(dr)(x)
    return x


def _lstm_backbone(x, cfg: dict):
    l2 = regularizers.l2(cfg.get("l2_reg", 1e-3))
    dr = cfg.get("dropout_rate", 0.3)
    rdr = cfg.get("recurrent_dropout", 0.1)
    units = cfg.get("units", [128, 64])
    for i, u in enumerate(units):
        return_seq = i < len(units) - 1
        x = layers.LSTM(u, return_sequences=return_seq,
                        dropout=dr, recurrent_dropout=rdr,
                        kernel_regularizer=l2)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(dr)(x)
    return x


def _cnn_lstm_backbone(x, cfg: dict):
    l2 = regularizers.l2(cfg.get("l2_reg", 1e-3))
    dr = cfg.get("dropout_rate", 0.3)
    for f in cfg.get("filters", [64, 128]):
        x = layers.Conv1D(f, 3, padding="same", activation="relu",
                          kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
    for i, u in enumerate(cfg.get("lstm_units", [64, 32])):
        return_seq = i < len(cfg.get("lstm_units", [64, 32])) - 1
        x = layers.LSTM(u, return_sequences=return_seq,
                        dropout=dr, kernel_regularizer=l2)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(dr)(x)
    return x


def _transformer_backbone(x, cfg: dict):
    d_model = cfg.get("head_size", 32) * cfg.get("num_heads", 4)
    l2 = cfg.get("l2_reg", 1e-4)
    dr = cfg.get("dropout_rate", 0.2)
    x = layers.Dense(d_model)(x)
    x = SinusoidalPositionalEncoding()(x)
    for _ in range(cfg.get("num_blocks", 2)):
        x = TransformerBlock(
            head_size=cfg.get("head_size", 32),
            num_heads=cfg.get("num_heads", 4),
            ff_dim=cfg.get("ff_dim", 128),
            dropout=dr, l2_reg=l2,
        )(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dr)(x)
    return x


_BACKBONES = {
    "CNN":         _cnn_backbone,
    "LSTM":        _lstm_backbone,
    "CNN_LSTM":    _cnn_lstm_backbone,
    "Transformer": _transformer_backbone,
}

# ---------------------------------------------------------------------------
# Wavelet front-ends
# ---------------------------------------------------------------------------

def _apply_wavelet_frontend(x, mode: str, cfg: dict):
    """
    Prepend wavelet transform layer(s) to the input tensor.

    Após o frontend, aplica uma projeção linear (Dense) para mapear todos os modos
    (raw, db4, learned_wavelet) para a mesma dimensionalidade (wavelet_projection_dim).

    Motivação: sem essa projeção, raw=(N,L,F) e learned_wavelet=(N,L,(levels+1)*F)
    chegam ao backbone com capacidades efetivas diferentes — o backbone wavelet tem
    mais parâmetros na primeira camada, o que é um confundidor na comparação de modos.

    Com a projeção, todos os modos entram no backbone com (N, L, proj_dim),
    tornando a comparação mais justa.
    """
    if mode == "raw":
        # Sem transformada; projeção ainda é aplicada para equalizar capacidade
        x_out = x

    elif mode == "db4":
        from models.LWT.fixed_db4_dwt import FixedDb4DWT1D
        x_out = FixedDb4DWT1D(
            levels=cfg.get("wavelet_levels", 2),
            mode="concat",
            # pad_to_first: sem interpolação bilinear = sem aliasing espectral
            align=cfg.get("align", "pad_to_first"),
        )(x)

    elif mode == "learned_wavelet":
        from models.LWT.learned_wavelet_dwt_qmf import LearnedWaveletDWT1D_QMF
        x_out = LearnedWaveletDWT1D_QMF(
            levels=cfg.get("wavelet_levels", 2),
            kernel_size=cfg.get("kernel_size", 8),
            wavelet_net_units=cfg.get("wavelet_net_units", 32),
            reg_energy=cfg.get("reg_energy", 1e-2),
            reg_high_dc=cfg.get("reg_high_dc", 1e-2),
            reg_smooth=cfg.get("reg_smooth", 1e-3),
            align=cfg.get("align", "pad_to_first"),
            warm_start_db4=cfg.get("warm_start_db4", False),
            mode="concat",
        )(x)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Projeção linear para dimensionalidade uniforme entre os modos.
    # Sem isso, raw=(N,L,F) vs learned_wavelet=(N,L,3F) — a primeira camada
    # do backbone teria número de parâmetros 3× maior para wavelet vs raw.
    proj_dim = cfg.get("wavelet_projection_dim", 0)
    if proj_dim and proj_dim > 0:
        x_out = layers.Dense(proj_dim, use_bias=False, name=f"wavelet_proj_{mode}")(x_out)

    return x_out


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    mode: str,
    input_shape: tuple,
    n_classes: int = 3,
    cfg: Optional[dict] = None,
) -> keras.Model:
    """
    Build and compile a classification model.

    Parameters
    ----------
    model_name  : CNN | LSTM | CNN_LSTM | Transformer
    mode        : raw | db4 | learned_wavelet
    input_shape : (sequence_length, n_features)
    n_classes   : 3 (sell/hold/buy)
    cfg         : model config dict (from experiment_config.DL_MODELS_CONFIG)
    """
    if cfg is None:
        from config.experiment_config import DL_MODELS_CONFIG, DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG
        cfg = {**DL_MODELS_CONFIG.get(model_name, {}), **DL_TRAINING_CONFIG, **LEARNED_WAVELET_CONFIG}

    backbone_fn = _BACKBONES.get(model_name)
    if backbone_fn is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(_BACKBONES)}")

    inputs = keras.Input(shape=input_shape, name="input")
    x = _apply_wavelet_frontend(inputs, mode, cfg)
    x = backbone_fn(x, cfg)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name=f"{model_name}_{mode}")

    # Otimizador unificado: Adam com lr inicial igual para todas as arquiteturas.
    # ReduceLROnPlateau é definido nos callbacks (get_callbacks), não aqui,
    # para que o scheduler seja registrado no histórico de treino.
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.get("learning_rate", 1e-3)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_with_continuous_signal(
    model_name: str,
    mode: str,
    input_shape: tuple,
    cfg: Optional[dict] = None,
) -> keras.Model:
    """
    Variante que produz sinal de posição contínuo em [-1, +1] em vez de softmax.

    Saída: p_buy - p_sell ∈ [-1, +1]
    - +1: compra forte (p_buy ≈ 1, p_sell ≈ 0)
    - -1: venda forte
    -  0: hold (equilíbrio ou dominância de hold)

    Isso permite tamanho de posição fracionário no backtest, eliminando
    o threshold binário que descarta confiança do modelo.
    """
    if cfg is None:
        from config.experiment_config import DL_MODELS_CONFIG, DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG
        cfg = {**DL_MODELS_CONFIG.get(model_name, {}), **DL_TRAINING_CONFIG, **LEARNED_WAVELET_CONFIG}

    backbone_fn = _BACKBONES.get(model_name)
    if backbone_fn is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(_BACKBONES)}")

    inputs = keras.Input(shape=input_shape, name="input")
    x = _apply_wavelet_frontend(inputs, mode, cfg)
    x = backbone_fn(x, cfg)

    # Softmax de 3 classes para obter probabilidades
    probs = layers.Dense(3, activation="softmax", name="probs")(x)

    # Sinal contínuo: diferença entre probabilidade de compra e venda
    # Lambda não é serializável; em produção, substituir por uma Layer customizada
    signal = layers.Lambda(
        lambda p: p[:, 2:3] - p[:, 0:1],
        name="position_signal",
    )(probs)

    model = keras.Model(inputs, [probs, signal], name=f"{model_name}_{mode}_continuous")
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.get("learning_rate", 1e-3)),
        loss={"probs": "sparse_categorical_crossentropy", "position_signal": None},
        metrics={"probs": "accuracy"},
    )
    return model


def get_callbacks(
    model_path: Path,
    early_patience: int = 15,
    lr_patience: int = 7,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list:
    """
    Callbacks padrão para todos os modelos.

    ReduceLROnPlateau é aplicado uniformemente a CNN, LSTM, CNN_LSTM e Transformer,
    garantindo que a comparação entre modos (raw/db4/learned) seja sobre o frontend
    wavelet e não sobre diferenças no regime de learning rate.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_patience,
            restore_best_weights=True, verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=lr_factor,
            patience=lr_patience, min_lr=min_lr, verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss",
            save_best_only=True, verbose=0,
        ),
    ]
