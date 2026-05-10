"""
Model factories for the financial experiment (multiclass classification).

DL models delegate to models/base_models.py — all backbone implementations
live there so that synthetic, financial, and ford-a share the same code.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Path bootstrap — add root and models/ so base_models / dl_utils / LWT work
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent.parent
for _p in (str(_ROOT / "models"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base_models import (  # noqa: E402
    build_model as _build_model,
    apply_wavelet_frontend,
    get_callbacks,            # re-exported for notebook consumers
    get_distribute_strategy,  # re-exported for notebook consumers
    TransformerWarmupSchedule,
    _BACKBONES,
)


# ---------------------------------------------------------------------------
# Public factory — multiclass classification wrapper
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    mode: str,
    input_shape: tuple,
    n_classes: int = 3,
    cfg: Optional[dict] = None,
) -> keras.Model:
    """
    Build and compile a classification model for the financial experiment.

    Parameters
    ----------
    model_name  : CNN | LSTM | CNN_LSTM | Transformer | MLP
    mode        : raw | db4 | learned_wavelet | learned_wavelet_no_warmup
    input_shape : (sequence_length, n_features)
    n_classes   : number of output classes (default 3: sell/hold/buy)
    cfg         : merged config dict (DL_MODELS_CONFIG + DL_TRAINING_CONFIG
                  + LEARNED_WAVELET_CONFIG)
    """
    if cfg is None:
        from config.experiment_config import (
            DL_MODELS_CONFIG, DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG,
        )
        cfg = {
            **DL_MODELS_CONFIG.get(model_name, {}),
            **DL_TRAINING_CONFIG,
            **LEARNED_WAVELET_CONFIG,
        }

    return _build_model(
        model_name=model_name,
        mode=mode,
        input_shape=input_shape,
        task="multiclass",
        n_classes=n_classes,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Continuous-signal variant (financial-specific)
# Produces p_buy - p_sell ∈ [-1, +1] alongside the softmax probabilities.
# ---------------------------------------------------------------------------

def build_model_with_continuous_signal(
    model_name: str,
    mode: str,
    input_shape: tuple,
    cfg: Optional[dict] = None,
) -> keras.Model:
    """
    Variant that outputs a continuous position signal in [-1, +1] in addition
    to the softmax probabilities.

    Output: (probs [B, n_classes], signal [B, 1])
      signal = p_buy - p_sell ∈ [-1, +1]
      +1 → strong buy, -1 → strong sell, 0 → hold / balanced
    """
    if cfg is None:
        from config.experiment_config import (
            DL_MODELS_CONFIG, DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG,
        )
        cfg = {
            **DL_MODELS_CONFIG.get(model_name, {}),
            **DL_TRAINING_CONFIG,
            **LEARNED_WAVELET_CONFIG,
        }

    backbone_fn = _BACKBONES.get(model_name)
    if backbone_fn is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(_BACKBONES)}")

    inputs = keras.Input(shape=input_shape, name="input")
    x = apply_wavelet_frontend(inputs, mode, cfg)
    x = backbone_fn(x, cfg)

    probs = layers.Dense(3, activation="softmax", name="probs")(x)
    # Lambda is not serialisable; replace with a custom Layer in production.
    signal = layers.Lambda(
        lambda p: p[:, 2:3] - p[:, 0:1],
        name="position_signal",
    )(probs)

    model = keras.Model(inputs, [probs, signal],
                        name=f"{model_name}_{mode}_continuous")

    lr = cfg.get("learning_rate", 1e-3)
    if model_name == "Transformer" and cfg.get("use_warmup", False):
        embed_dim = cfg.get("head_size", 64) * cfg.get("num_heads", 4)
        lr = TransformerWarmupSchedule(
            d_model=embed_dim,
            warmup_steps=cfg.get("warmup_steps", 500),
        )
        opt = keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    else:
        opt = keras.optimizers.Adam(lr)

    model.compile(
        optimizer=opt,
        loss={"probs": "sparse_categorical_crossentropy", "position_signal": None},
        metrics={"probs": "accuracy"},
    )
    return model
