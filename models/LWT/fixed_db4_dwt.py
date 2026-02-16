import tensorflow as tf
from tensorflow.keras.layers import Layer
import pywt


class FixedDb4DWT1D(Layer):
    """
    DWT multi-nível FIXA usando wavelet db4 (PyWavelets).
    Aplica conv (low/high) e downsample por 2 em cada nível.

    Retornos:
      - mode="coeffs": (A_L, [D1..DL])
      - mode="concat": concatena [D1..DL, A_L] no eixo de canais, alinhando comprimentos.
    """

    def __init__(self, levels=3, mode="concat", align="upsample", padding="SAME", **kwargs):
        super().__init__(**kwargs)
        self.levels = int(levels)
        self.mode = mode
        self.align = align
        self.padding = padding

        w = pywt.Wavelet("db4")
        self.h = tf.constant(w.dec_lo, dtype=tf.float32)
        self.g = tf.constant(w.dec_hi, dtype=tf.float32)
        self.kernel_size = len(w.dec_lo)

    def _depthwise_conv(self, x, k):
        # x: (B,L,C), k: (K,) -> (B,L,C) depthwise por canal
        C = x.shape[-1]
        k = tf.reshape(k, [self.kernel_size, 1, 1])  # (K,1,1)
        if isinstance(C, int):
            k = tf.tile(k, [1, C, 1])               # (K,C,1)
        else:
            k = tf.tile(k, [1, tf.shape(x)[-1], 1])

        x2 = tf.expand_dims(x, axis=2)              # (B,L,1,C)
        k2 = tf.expand_dims(k, axis=1)              # (K,1,C,1)
        y2 = tf.nn.depthwise_conv2d(x2, k2, strides=[1, 1, 1, 1], padding=self.padding)
        return tf.squeeze(y2, axis=2)               # (B,L,C)

    def _upsample_to_len(self, x, target_len):
        x2 = tf.expand_dims(x, axis=2)
        x2 = tf.image.resize(x2, (target_len, 1))
        return x2[:, :, 0, :]

    def _pad_to_len(self, x, target_len):
        L = tf.shape(x)[1]
        pad = tf.maximum(0, target_len - L)
        paddings = tf.stack([[0, 0], [0, pad], [0, 0]])
        return tf.pad(x, paddings)

    def call(self, inputs):
        x = inputs
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)
        if len(x.shape) != 3:
            raise ValueError("inputs deve ter shape (batch, length) ou (batch, length, channels)")

        details = []
        A = x
        for _ in range(self.levels):
            A_full = self._depthwise_conv(A, self.h)
            D_full = self._depthwise_conv(A, self.g)
            A = A_full[:, ::2, :]
            D = D_full[:, ::2, :]
            details.append(D)

        if self.mode == "coeffs":
            return A, details

        if self.mode == "concat":
            target_len = tf.shape(details[0])[1]
            if self.align == "upsample":
                parts = [self._upsample_to_len(d, target_len) for d in details]
                parts.append(self._upsample_to_len(A, target_len))
                return tf.concat(parts, axis=-1)
            if self.align == "pad_to_first":
                parts = [self._pad_to_len(d, target_len) for d in details]
                parts.append(self._pad_to_len(A, target_len))
                return tf.concat(parts, axis=-1)

            parts = [self._upsample_to_len(d, target_len) for d in details]
            parts.append(self._upsample_to_len(A, target_len))
            return tf.concat(parts, axis=-1)

        return A, details


__all__ = ["FixedDb4DWT1D"]
