"""
Shared DL backbone implementations for all experiment tracks.

Provides a unified factory so that synthetic, financial, and ford-a
experiments can share the same CNN / LSTM / CNN_LSTM / Transformer / MLP
implementations, differing only in the output head and wavelet mode.

Usage (from an experiment's models.py):
    import sys, os
    _MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
    if _MODELS_DIR not in sys.path:
        sys.path.insert(0, _MODELS_DIR)
    from base_models import build_model, get_callbacks, get_distribute_strategy

Or from project root (if root is on sys.path):
    from models.base_models import build_model, get_callbacks, get_distribute_strategy
"""
from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path
from typing import Optional

# Ensure models/ is on path so sibling imports (dl_utils, LWT) always work.
_MODELS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _MODELS_DIR not in _sys.path:
    _sys.path.insert(0, _MODELS_DIR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from dl_utils import SinusoidalPositionalEncoding, TransformerBlock, TransformerWarmupSchedule


# ---------------------------------------------------------------------------
# Backbone builders — tensor-in / tensor-out, fully config-dict driven
# ---------------------------------------------------------------------------

def _cnn_backbone(x, cfg: dict):
    """
    Conv1D backbone.

    Config keys
    -----------
    filters       : list[int]  — conv channels per layer  (default [64, 128, 256])
    kernel_sizes  : list[int] | int — kernel per layer or single shared value.
                    Also accepts "cnn_kernel_size" (financial-style alias).
    pool_sizes    : list[int] | None — MaxPooling per layer; None = no pooling.
    dense_units   : list[int] — dense head units (default [64])
    dropout_rate  : float (default 0.3)
    l2_reg        : float (default 1e-3)
    """
    l2 = regularizers.l2(cfg.get("l2_reg", 1e-3))
    dr = cfg.get("dropout_rate", 0.3)
    filters = cfg.get("filters", [64, 128, 256])
    # Support both "kernel_sizes" (list) and "cnn_kernel_size" (financial single-int alias)
    kernel_sizes = cfg.get("kernel_sizes", cfg.get("cnn_kernel_size", 3))
    pool_sizes = cfg.get("pool_sizes", None)
    dense_units = cfg.get("dense_units", [64])

    for i, f in enumerate(filters):
        k = kernel_sizes[i] if isinstance(kernel_sizes, list) else kernel_sizes
        x = layers.Conv1D(f, k, padding="same", activation="relu",
                          kernel_regularizer=l2, name=f"conv_{i+1}")(x)
        x = layers.BatchNormalization()(x)
        if pool_sizes is not None:
            p = pool_sizes[i] if isinstance(pool_sizes, list) else pool_sizes
            x = layers.MaxPooling1D(pool_size=p)(x)
        x = layers.Dropout(dr)(x)

    x = layers.GlobalAveragePooling1D()(x)
    for i, u in enumerate(dense_units):
        x = layers.Dense(u, activation="relu", kernel_regularizer=l2,
                         name=f"dense_{i+1}")(x)
        x = layers.Dropout(dr)(x)
    return x


def _lstm_backbone(x, cfg: dict):
    """
    Stacked LSTM backbone with optional bidirectionality.

    Config keys
    -----------
    units             : list[int]  (default [128, 64])
    recurrent_dropout : float (default 0.0)
    bidirectional     : bool  (default False)
    dense_units       : list[int] (default [64])
    dropout_rate, l2_reg
    """
    l2 = regularizers.l2(cfg.get("l2_reg", 1e-3))
    dr = cfg.get("dropout_rate", 0.3)
    rdr = cfg.get("recurrent_dropout", 0.0)
    units = cfg.get("units", [128, 64])
    bidirectional = cfg.get("bidirectional", False)
    dense_units = cfg.get("dense_units", [64])

    for i, u in enumerate(units):
        return_seq = i < len(units) - 1
        lstm = layers.LSTM(u, return_sequences=return_seq,
                           dropout=dr, recurrent_dropout=rdr,
                           kernel_regularizer=l2, name=f"lstm_{i+1}")
        x = layers.Bidirectional(lstm)(x) if bidirectional else lstm(x)

    for u in dense_units:
        x = layers.Dense(u, activation="relu", kernel_regularizer=l2)(x)
        x = layers.Dropout(dr)(x)
    return x


def _cnn_lstm_backbone(x, cfg: dict):
    """
    CNN feature extraction followed by LSTM temporal modelling.

    Config keys
    -----------
    filters / cnn_filters       : list[int]  (default [64, 128])
    kernel_sizes / cnn_kernel_sizes : list[int] (default [5, 3])
    lstm_units                  : list[int]  (default [100, 50])
    dense_units                 : list[int]  (default [64])
    dropout_rate, l2_reg
    """
    l2 = regularizers.l2(cfg.get("l2_reg", 1e-3))
    dr = cfg.get("dropout_rate", 0.3)
    # Accept both "cnn_filters" (synthetic) and "filters" (financial) aliases
    cnn_filters = cfg.get("cnn_filters", cfg.get("filters", [64, 128]))
    cnn_ks = cfg.get("cnn_kernel_sizes", cfg.get("kernel_sizes", [5, 3]))

    for i, (f, k) in enumerate(zip(cnn_filters, cnn_ks)):
        x = layers.Conv1D(f, k, padding="same", activation="relu",
                          kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(dr)(x)

    lstm_units = cfg.get("lstm_units", [100, 50])
    for i, u in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        x = layers.LSTM(u, return_sequences=return_seq,
                        dropout=dr, kernel_regularizer=l2)(x)

    for u in cfg.get("dense_units", [64]):
        x = layers.Dense(u, activation="relu", kernel_regularizer=l2)(x)
        x = layers.Dropout(dr)(x)
    return x


def _transformer_backbone(x, cfg: dict):
    """
    Transformer encoder backbone with sinusoidal positional encoding.

    Config keys
    -----------
    head_size               : int  (default 64)
    num_heads               : int  (default 4)
    ff_dim                  : int  (default 128)
    num_transformer_blocks / num_blocks : int (default 2)
    mlp_units               : list[int] (default [128, 64])
    dropout_rate, l2_reg
    """
    head_size = cfg.get("head_size", 64)
    num_heads = cfg.get("num_heads", 4)
    ff_dim = cfg.get("ff_dim", 128)
    # Accept both "num_transformer_blocks" (synthetic) and "num_blocks" (financial)
    num_blocks = cfg.get("num_transformer_blocks", cfg.get("num_blocks", 2))
    dr = cfg.get("dropout_rate", 0.2)
    l2_reg_val = cfg.get("l2_reg", 1e-4)
    mlp_units = cfg.get("mlp_units", [128, 64])

    embed_dim = head_size * num_heads
    x = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(l2_reg_val))(x)
    x = SinusoidalPositionalEncoding()(x)
    x = layers.Dropout(dr)(x)
    for _ in range(num_blocks):
        x = TransformerBlock(head_size=head_size, num_heads=num_heads,
                             ff_dim=ff_dim, dropout=dr, l2_reg=l2_reg_val)(x)
    x = layers.GlobalAveragePooling1D()(x)
    for u in mlp_units:
        x = layers.Dense(u, activation="relu")(x)
        x = layers.Dropout(dr)(x)
    return x


def _mlp_backbone(x, cfg: dict):
    """
    Flatten + stacked Dense layers (MLP baseline).

    Config keys
    -----------
    mlp_units    : list[int] (default [128, 64, 32])
    dropout_rate, l2_reg
    """
    l2 = regularizers.l2(cfg.get("l2_reg", 1e-3))
    dr = cfg.get("dropout_rate", 0.3)
    x = layers.Flatten()(x)
    for u in cfg.get("mlp_units", [128, 64, 32]):
        x = layers.Dense(u, activation="relu", kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
    return x


_BACKBONES = {
    "CNN":         _cnn_backbone,
    "LSTM":        _lstm_backbone,
    "CNN_LSTM":    _cnn_lstm_backbone,
    "Transformer": _transformer_backbone,
    "MLP":         _mlp_backbone,
}


# ---------------------------------------------------------------------------
# Wavelet frontend
# ---------------------------------------------------------------------------

def apply_wavelet_frontend(x, mode: str, cfg: dict):
    """
    Optionally prepend a wavelet transform to tensor x.

    Modes
    -----
    raw                       : identity (no transform)
    db4                       : fixed db4 DWT (FixedDb4DWT1D)
    learned_wavelet           : learnable QMF; warm_start_db4 read from cfg
    learned_wavelet_no_warmup : same arch, warm_start_db4 forced False

    Config keys used
    ----------------
    wavelet_levels / levels, kernel_size, wavelet_net_units,
    reg_energy, reg_high_dc, reg_smooth, align, warm_start_db4,
    wavelet_projection_dim (optional linear projection after transform)
    """
    if mode == "raw":
        x_out = x

    elif mode == "db4":
        from LWT.fixed_db4_dwt import FixedDb4DWT1D
        x_out = FixedDb4DWT1D(
            levels=cfg.get("wavelet_levels", cfg.get("levels", 2)),
            mode="concat",
            align=cfg.get("align", "pad_to_first"),
        )(x)

    elif mode in ("learned_wavelet", "learned_wavelet_no_warmup"):
        from LWT.learned_wavelet_dwt_qmf import LearnedWaveletDWT1D_QMF
        warm = False if mode == "learned_wavelet_no_warmup" else cfg.get("warm_start_db4", False)
        x_out = LearnedWaveletDWT1D_QMF(
            levels=cfg.get("wavelet_levels", cfg.get("levels", 2)),
            kernel_size=cfg.get("kernel_size", 8),
            wavelet_net_units=cfg.get("wavelet_net_units", 32),
            reg_energy=cfg.get("reg_energy", 1e-2),
            reg_high_dc=cfg.get("reg_high_dc", 1e-2),
            reg_smooth=cfg.get("reg_smooth", 1e-3),
            align=cfg.get("align", "pad_to_first"),
            warm_start_db4=warm,
            mode="concat",
        )(x)

    else:
        raise ValueError(
            f"Unknown wavelet mode '{mode}'. "
            "Choose: raw, db4, learned_wavelet, learned_wavelet_no_warmup"
        )

    proj_dim = cfg.get("wavelet_projection_dim", 0)
    if proj_dim:
        x_out = layers.Dense(proj_dim, use_bias=False,
                              name=f"wavelet_proj_{mode}")(x_out)
    return x_out


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    mode: str,
    input_shape: tuple,
    task: str = "regression",
    n_classes: Optional[int] = None,
    cfg: Optional[dict] = None,
) -> keras.Model:
    """
    Build and compile a model.

    Parameters
    ----------
    model_name  : CNN | LSTM | CNN_LSTM | Transformer | MLP
    mode        : raw | db4 | learned_wavelet | learned_wavelet_no_warmup
    input_shape : (sequence_length, n_features)
    task        : regression | binary | multiclass
    n_classes   : required when task='multiclass'
    cfg         : merged config dict (model + training + wavelet params)

    Returns
    -------
    Compiled keras.Model
    """
    if cfg is None:
        cfg = {}
    backbone_fn = _BACKBONES.get(model_name)
    if backbone_fn is None:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(_BACKBONES)}")

    inputs = keras.Input(shape=input_shape, name="input")
    x = apply_wavelet_frontend(inputs, mode, cfg)
    x = backbone_fn(x, cfg)

    lr = cfg.get("learning_rate", 1e-3)

    if task == "regression":
        outputs = layers.Dense(1, name="output")(x)
        loss, metrics = "mse", ["mae"]
        opt = keras.optimizers.Adam(lr)

    elif task == "binary":
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        loss, metrics = "binary_crossentropy", ["accuracy"]
        opt = keras.optimizers.Adam(lr)

    elif task == "multiclass":
        if n_classes is None:
            raise ValueError("n_classes must be provided for task='multiclass'")
        outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)
        loss, metrics = "sparse_categorical_crossentropy", ["accuracy"]
        opt = keras.optimizers.Adam(lr)

    else:
        raise ValueError(f"Unknown task '{task}'. Choose: regression, binary, multiclass")

    # Transformer: optionally replace Adam with warmup LR schedule
    if model_name == "Transformer" and cfg.get("use_warmup", False):
        embed_dim = cfg.get("head_size", 64) * cfg.get("num_heads", 4)
        lr_sched = TransformerWarmupSchedule(
            d_model=embed_dim,
            warmup_steps=cfg.get("warmup_steps", 500),
        )
        opt = keras.optimizers.Adam(lr_sched, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model = keras.Model(inputs, outputs, name=f"{model_name}_{mode}")
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def get_callbacks(
    model_path,
    early_patience: int = 15,
    lr_patience: int = 7,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    monitor: str = "val_loss",
    use_reduce_lr: bool = True,
    verbose: int = 0,
) -> list:
    """
    Standard training callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau).

    Set use_reduce_lr=False when using a LearningRateSchedule optimizer
    (e.g. Transformer with warmup), as ReduceLROnPlateau conflicts with
    keras LR schedules.
    """
    Path(str(model_path)).parent.mkdir(parents=True, exist_ok=True)
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=early_patience,
            restore_best_weights=True, verbose=verbose,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor=monitor,
            save_best_only=True, verbose=verbose,
        ),
    ]
    if use_reduce_lr:
        cbs.append(keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=lr_factor,
            patience=lr_patience, min_lr=min_lr, verbose=verbose,
        ))
    return cbs


# ---------------------------------------------------------------------------
# Distribution strategy
# ---------------------------------------------------------------------------

def get_distribute_strategy() -> tf.distribute.Strategy:
    """Return the best available TF distribution strategy."""
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
        print(f"MirroredStrategy: {strategy.num_replicas_in_sync} GPUs")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print("OneDeviceStrategy: GPU:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("OneDeviceStrategy: CPU")
    return strategy


__all__ = [
    "build_model",
    "apply_wavelet_frontend",
    "get_callbacks",
    "get_distribute_strategy",
    # Re-exported so consumers don't need a direct dl_utils import
    "SinusoidalPositionalEncoding",
    "TransformerBlock",
    "TransformerWarmupSchedule",
]
