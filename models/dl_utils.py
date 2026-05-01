"""
Shared DL building blocks used across all experiment tracks.

Import from here instead of redefining in each models.py:

    from models.dl_utils import (
        SinusoidalPositionalEncoding,
        TransformerBlock,
        TransformerWarmupSchedule,
    )
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers


class SinusoidalPositionalEncoding(layers.Layer):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, max_len: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]
        positions = np.arange(self.max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self._pe = tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self._pe[:, :seq_len, :]

    def get_config(self):
        return {**super().get_config(), "max_len": self.max_len}


class TransformerBlock(layers.Layer):
    """
    Standard Transformer encoder block with pre-LayerNorm residual connections.

    Parameters
    ----------
    head_size  : key/query/value dimension per head
    num_heads  : number of attention heads
    ff_dim     : feed-forward hidden dimension
    dropout    : dropout rate applied after attention and FFN
    l2_reg     : L2 regularisation weight on dense kernels (0 = disabled)
    """

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        l2_reg: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.l2_reg = l2_reg

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        reg = regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None

        self.att = layers.MultiHeadAttention(
            key_dim=self.head_size,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            kernel_regularizer=reg,
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ff_dim, activation="relu", kernel_regularizer=reg),
            layers.Dropout(self.dropout_rate),
            layers.Dense(embed_dim, kernel_regularizer=reg),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(self.dropout_rate)
        self.do2 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=False):
        attn = self.do1(self.att(inputs, inputs, training=training), training=training)
        out1 = self.ln1(inputs + attn)
        ffn = self.do2(self.ffn(out1, training=training), training=training)
        return self.ln2(out1 + ffn)

    def get_config(self):
        return {
            **super().get_config(),
            "head_size":  self.head_size,
            "num_heads":  self.num_heads,
            "ff_dim":     self.ff_dim,
            "dropout":    self.dropout_rate,
            "l2_reg":     self.l2_reg,
        }


class TransformerWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup + inverse-sqrt decay (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, warmup_steps: int = 500):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step + 1)
        arg2 = (step + 1) * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model":       int(self.d_model.numpy()),
            "warmup_steps":  int(self.warmup_steps.numpy()),
        }


__all__ = [
    "SinusoidalPositionalEncoding",
    "TransformerBlock",
    "TransformerWarmupSchedule",
]
