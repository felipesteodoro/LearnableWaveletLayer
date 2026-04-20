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


# ---------------------------------------------------------------------------
# Positional encoding & Transformer block (same as synthetic)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(layers.Layer):
    def __init__(self, max_len: int = 512, **kw):
        super().__init__(**kw)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]
        pos = np.arange(self.max_len)[:, None]
        i   = np.arange(d_model)[None, :]
        angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])
        self._encoding = tf.cast(angle[None, :, :], tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self._encoding[:, :seq_len, :]


class TransformerBlock(layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, l2=1e-4, **kw):
        super().__init__(**kw)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size,
            kernel_regularizer=regularizers.l2(l2),
        )
        self.ff1  = layers.Dense(ff_dim, activation="relu",
                                 kernel_regularizer=regularizers.l2(l2))
        self.ff2  = layers.Dense(head_size * num_heads,
                                 kernel_regularizer=regularizers.l2(l2))
        self.ln1  = layers.LayerNormalization(epsilon=1e-6)
        self.ln2  = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        x = self.ln1(x + self.drop1(attn_out, training=training))
        ff_out = self.ff2(self.ff1(x))
        return self.ln2(x + self.drop2(ff_out, training=training))


# ---------------------------------------------------------------------------
# Backbone builders
# ---------------------------------------------------------------------------

def _cnn_backbone(x, cfg: dict):
    l2 = regularizers.l2(cfg.get("l2_reg", 1e-3))
    dr = cfg.get("dropout_rate", 0.3)
    for f in cfg.get("filters", [64, 128, 256]):
        x = layers.Conv1D(f, kernel_size=cfg.get("kernel_size", 3),
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
            dropout=dr, l2=l2,
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
    """Prepend wavelet transform layer(s) to the input tensor."""
    if mode == "raw":
        return x

    if mode == "db4":
        from models.LWT.fixed_db4_dwt import FixedDb4DWT1D
        return FixedDb4DWT1D(
            levels=cfg.get("wavelet_levels", 2),
            output_mode="concat",
        )(x)

    if mode == "learned_wavelet":
        from models.LWT.learned_wavelet_dwt_qmf import LearnedWaveletDWT1D_QMF
        return LearnedWaveletDWT1D_QMF(
            levels=cfg.get("wavelet_levels", 2),
            kernel_size=cfg.get("kernel_size", 32),
            wavelet_net_units=cfg.get("wavelet_net_units", 32),
            reg_energy=cfg.get("reg_energy", 1e-2),
            reg_high_dc=cfg.get("reg_high_dc", 1e-2),
            reg_smooth=cfg.get("reg_smooth", 1e-3),
            output_mode="concat",
        )(x)

    raise ValueError(f"Unknown mode: {mode}")


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
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.get("learning_rate", 1e-3)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(
    model_path: Path,
    early_patience: int = 15,
    lr_patience: int = 7,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list:
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
