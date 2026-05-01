import tensorflow as tf
from tensorflow.keras.layers import Layer

from .learned_wavelet_pair_qmf import LearnedWaveletPair1D_QMF


class LearnedWaveletDWT1D_QMF(Layer):
    """
    Pirâmide multi-nível estilo DWT, usando pares (low/high) com QMF embutido.

    QMF - Quadrature Mirror Filter: os filtros low e high são relacionados por
    um flip + alternância de sinal, garantindo cobertura completa da banda sem
    aliasing. A condição de energia é reforçada pela normalização L2 do filtro.

    Em cada nível:
      (A_prev, D_prev) = LearnedWaveletPair1D_QMF(A)
    e o próximo nível recebe apenas A (aproximação).

    Retornos:
      - mode="coeffs": (A_L, [D1, D2, ..., DL])
      - mode="concat": concatena [D1..DL, A_L] no eixo de canais, após alinhar comprimento.

    Parâmetros de alinhamento (mode="concat"):
      - align="pad_to_first" (padrão): zero-padding ao final dos coeficientes de detalhe
        para igualar o comprimento de D1. Preserva posições temporais originais e
        não introduz artefatos espectrais.
      - align="upsample": interpolação bilinear via tf.image.resize. CUIDADO: a
        interpolação bilinear introduz aliasing espectral (artefatos de alta frequência)
        que podem confundir o modelo downstream. Use apenas se necessitar comprimento
        exato sem lacunas de zeros.
    """

    def __init__(
        self,
        levels: int = 3,
        kernel_size: int = 8,
        wavelet_net_units: int = 32,
        padding: str = "SAME",
        mode: str = "coeffs",           # "coeffs" | "concat"
        align: str = "pad_to_first",    # "pad_to_first" (recomendado) | "upsample"
        # regularizadores repassados ao par QMF
        reg_energy: float = 1e-2,
        reg_high_dc: float = 1e-2,
        reg_smooth: float = 1e-3,
        normalize_low: str = "l2",      # "sum1" | "l2" | "none"
        warm_start_db4: bool = False,   # inicializa filtro próximo ao db4
        **kwargs
    ):
        super().__init__(**kwargs)
        self.levels            = int(levels)
        self.kernel_size       = int(kernel_size)
        self.wavelet_net_units = int(wavelet_net_units)
        self.padding           = padding
        self.mode              = mode
        self.align             = align
        self.normalize_low     = normalize_low
        self.warm_start_db4    = warm_start_db4

        # Um par QMF por nível: cada nível pode aprender filtros diferentes,
        # o que é mais expressivo do que compartilhar um único filtro.
        # Em prática, os gradientes frequentemente converge para filtros similares
        # nos níveis superiores (onde o sinal é mais suave).
        self.pairs = [
            LearnedWaveletPair1D_QMF(
                kernel_size=self.kernel_size,
                wavelet_net_units=self.wavelet_net_units,
                padding=self.padding,
                normalize_low=normalize_low,
                reg_energy=reg_energy,
                reg_high_dc=reg_high_dc,
                reg_smooth=reg_smooth,
                warm_start_db4=warm_start_db4,
            )
            for _ in range(self.levels)
        ]

    def _upsample_to_len(self, x, target_len):
        """
        Upsample via tf.image.resize (bilinear).

        AVISO: interpola os valores, introduzindo artefatos espectrais.
        Prefira _pad_to_len para comparações quantitativas entre modos.
        """
        x2 = tf.expand_dims(x, axis=2)            # (B, L, 1, C)
        x2 = tf.image.resize(x2, (target_len, 1)) # (B, target_len, 1, C)
        return x2[:, :, 0, :]                     # (B, target_len, C)

    def _pad_to_len(self, x, target_len):
        """
        Alinha comprimento por zero-padding à direita.

        Os zeros adicionados representam "sem informação wavelet nessa posição",
        o que é semanticamente correto: nos níveis mais profundos, a resolução
        temporal é menor, então a lacuna entre posições é natural.
        """
        L = tf.shape(x)[1]
        pad = tf.maximum(0, target_len - L)
        paddings = tf.stack([[0, 0], [0, pad], [0, 0]])
        return tf.pad(x, paddings)

    def call(self, inputs):
        """
        inputs:
          - (B, L) ou (B, L, C)

        Em mode="concat", retorna (B, L_D1, C*(levels+1)) onde L_D1 = L//2.
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
            # Alinha todos os coeficientes ao comprimento de D1 (nível 1)
            # D1 tem comprimento L//2, D2 tem L//4, A tem L//(2^L)
            target_len = tf.shape(details[0])[1]

            if self.align == "pad_to_first":
                # Recomendado: zero-padding preserva posições temporais
                parts = [self._pad_to_len(d, target_len) for d in details]
                parts.append(self._pad_to_len(A, target_len))
                return tf.concat(parts, axis=-1)

            if self.align == "upsample":
                # Alternativa: interpolação bilinear (pode introduzir artefatos)
                parts = [self._upsample_to_len(d, target_len) for d in details]
                parts.append(self._upsample_to_len(A, target_len))
                return tf.concat(parts, axis=-1)

            # Fallback para pad_to_first
            parts = [self._pad_to_len(d, target_len) for d in details]
            parts.append(self._pad_to_len(A, target_len))
            return tf.concat(parts, axis=-1)

        # Fallback
        return A, details

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "levels":            self.levels,
            "kernel_size":       self.kernel_size,
            "wavelet_net_units": self.wavelet_net_units,
            "padding":           self.padding,
            "mode":              self.mode,
            "align":             self.align,
            "normalize_low":     self.normalize_low,
            "warm_start_db4":    self.warm_start_db4,
        })
        return cfg


__all__ = ["LearnedWaveletDWT1D_QMF"]
