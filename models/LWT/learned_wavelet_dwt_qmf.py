import tensorflow as tf
from tensorflow.keras.layers import Layer

from learned_wavelet_pair_qmf import LearnedWaveletPair1D_QMF


class LearnedWaveletDWT1D_QMF(Layer):
    """
    Pirâmide multi-nível estilo DWT, usando pares (low/high) com QMF embutido.
    QMF - Quadrature Mirror Filter: os filtros low e high são relacionados por um flip + shift, garantindo que eles sejam complementares e cubram toda a banda de frequências.
    Em cada nível:
      (A, D) = LearnedWaveletPair1D_QMF(A_prev)
    e o próximo nível recebe apenas A.

    Retornos:
      - mode="coeffs": (A_L, [D1, D2, ..., DL])
      - mode="concat": concatena [D1..DL, A_L] no eixo de canais, após alinhar comprimento.
    """

    def __init__(
        self,
        levels: int = 3,
        kernel_size: int = 32,
        wavelet_net_units: int = 32,
        padding: str = "SAME",
        mode: str = "coeffs",          # "coeffs" | "concat"
        align: str = "upsample",       # "upsample" | "pad_to_first"
        # regularizadores repassados ao par QMF
        reg_energy: float = 1e-2,
        reg_high_dc: float = 1e-2,
        reg_smooth: float = 1e-3,
        normalize_low: str = "sum1",   # "sum1" | "l2" | "none"
        **kwargs
    ):
        super().__init__(**kwargs)
        self.levels = int(levels)
        self.kernel_size = int(kernel_size)
        self.wavelet_net_units = int(wavelet_net_units)
        self.padding = padding
        self.mode = mode
        self.align = align

        self.pairs = [
            LearnedWaveletPair1D_QMF(
                kernel_size=self.kernel_size,
                wavelet_net_units=self.wavelet_net_units,
                padding=self.padding,
                normalize_low=normalize_low,
                reg_energy=reg_energy,
                reg_high_dc=reg_high_dc,
                reg_smooth=reg_smooth,
            )
            for _ in range(self.levels)
        ]

    def _upsample_to_len(self, x, target_len):
        # x: (B, L, C) -> (B, target_len, C) via resize (linear)
        x2 = tf.expand_dims(x, axis=2)            # (B, L, 1, C)
        x2 = tf.image.resize(x2, (target_len, 1)) # (B, target_len, 1, C)
        return x2[:, :, 0, :]                     # (B, target_len, C)

    def _pad_to_len(self, x, target_len):
        # x: (B, L, C) -> pad final até target_len
        L = tf.shape(x)[1]
        pad = tf.maximum(0, target_len - L)
        paddings = tf.stack([[0, 0], [0, pad], [0, 0]])
        return tf.pad(x, paddings)

    def call(self, inputs):
        """
        inputs:
          - (B, L) ou (B, L, C)
        """
        x = inputs
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)  # (B, L, 1)
        if len(x.shape) != 3:
            raise ValueError("inputs deve ter shape (batch, length) ou (batch, length, channels)")

        details = []
        A = x

        for lvl in range(self.levels):
            A, D = self.pairs[lvl](A)
            details.append(D)

        if self.mode == "coeffs":
            return A, details

        if self.mode == "concat":
            # alinhar no comprimento do nível 1 (D1)
            target_len = tf.shape(details[0])[1]

            if self.align == "upsample":
                parts = [self._upsample_to_len(d, target_len) for d in details]
                parts.append(self._upsample_to_len(A, target_len))
                return tf.concat(parts, axis=-1)

            if self.align == "pad_to_first":
                parts = [self._pad_to_len(d, target_len) for d in details]
                parts.append(self._pad_to_len(A, target_len))
                return tf.concat(parts, axis=-1)

            # fallback
            parts = [self._upsample_to_len(d, target_len) for d in details]
            parts.append(self._upsample_to_len(A, target_len))
            return tf.concat(parts, axis=-1)

        # fallback
        return A, details


__all__ = ["LearnedWaveletDWT1D_QMF"]
