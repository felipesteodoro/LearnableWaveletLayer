import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class LearnedWaveletPair1D_QMF(Layer):
    """
    Aprende apenas o filtro passa-baixa h (low-pass) e deriva o passa-alta g (high-pass)
    pela relação QMF: g[n] = (-1)^n * h[L-1-n].

    Retorna (A, D) já com downsample por 2 (estilo DWT).

    Regularizadores via self.add_loss():
      - Energia do low-pass (||h||2 ~ 1)
      - DC do high-pass (sum(g) ~ 0)
      - Suavidade do low-pass (segunda diferença pequena)
    """

    def __init__(
        self,
        kernel_size: int,
        wavelet_net_units: int = 32,
        padding: str = "SAME",
        eps: float = 1e-6,
        normalize_low: str = "sum1",   # "sum1" | "l2" | "none"
        # pesos dos regularizadores
        reg_energy: float = 1e-2,
        reg_high_dc: float = 1e-2,
        reg_smooth: float = 1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.wavelet_net_units = int(wavelet_net_units)
        self.padding = padding
        self.eps = float(eps)
        self.normalize_low = normalize_low

        self.reg_energy = float(reg_energy)
        self.reg_high_dc = float(reg_high_dc)
        self.reg_smooth = float(reg_smooth)

    def build(self, input_shape):
        self.base_net = tf.keras.Sequential([
            Dense(self.wavelet_net_units, activation="relu"),
            Dense(self.wavelet_net_units, activation="relu"),
        ])
        self.low_head = Dense(self.kernel_size)

        self.raw_scale = self.add_weight(
            name="raw_scale", shape=(), initializer="random_normal",
            trainable=True, dtype=tf.float32
        )
        self.translation = self.add_weight(
            name="translation", shape=(), initializer="random_normal",
            trainable=True, dtype=tf.float32
        )
        super().build(input_shape)

    def _make_t(self):
        t = tf.linspace(-self.kernel_size // 2, self.kernel_size // 2, self.kernel_size)
        t = tf.cast(t, tf.float32)
        return tf.reshape(t, [1, -1])  # (1, K)

    def _normalize_h(self, h):
        if self.normalize_low == "sum1":
            s = tf.reduce_sum(h, axis=-1, keepdims=True)
            return h / (s + self.eps)
        elif self.normalize_low == "l2":
            n = tf.norm(h, axis=-1, keepdims=True)
            return h / (n + self.eps)
        return h  # "none"

    def _qmf_from_h(self, h):
        g = tf.reverse(h, axis=[-1])
        sign = tf.pow(-1.0, tf.cast(tf.range(self.kernel_size), tf.float32))
        sign = tf.reshape(sign, [1, -1])
        return g * sign

    def _depthwise_conv1d_per_channel(self, x, k):
        """
        x: (B, L, C), k: (K, C, 1) -> (B, L, C)
        """
        x2 = tf.expand_dims(x, axis=2)   # (B, L, 1, C)
        k2 = tf.expand_dims(k, axis=1)   # (K, 1, C, 1)
        y2 = tf.nn.depthwise_conv2d(x2, k2, strides=[1, 1, 1, 1], padding=self.padding)
        return tf.squeeze(y2, axis=2)

    def call(self, inputs):
        """
        inputs:
          - (batch, length) ou (batch, length, channels)

        returns:
          A: (batch, ceil(length/2), channels)
          D: (batch, ceil(length/2), channels)
        """
        x = inputs
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)  # (B, L, 1)
        if len(x.shape) != 3:
            raise ValueError("inputs deve ter shape (batch, length) ou (batch, length, channels)")

        C = x.shape[-1]
        if C is None:
            C = tf.shape(x)[-1]

        # t ajustado por escala/translação
        t = self._make_t()
        scale = tf.nn.softplus(self.raw_scale) + 1e-3
        t_adj = (t - self.translation) / scale

        # Aprende h e deriva g por QMF
        z = self.base_net(t_adj)     # (1, hidden)
        h = self.low_head(z)         # (1, K)
        h = self._normalize_h(h)     # (1, K)
        g = self._qmf_from_h(h)      # (1, K)

        # --------- Regularizadores via add_loss ----------
        # 1) Energia do low-pass: ||h||2 ~ 1
        if self.reg_energy > 0.0:
            h_l2 = tf.norm(h, axis=-1)  # (1,)
            self.add_loss(self.reg_energy * tf.reduce_mean(tf.square(h_l2 - 1.0)))

        # 2) DC do high-pass: sum(g) ~ 0
        if self.reg_high_dc > 0.0:
            g_sum = tf.reduce_sum(g, axis=-1)  # (1,)
            self.add_loss(self.reg_high_dc * tf.reduce_mean(tf.square(g_sum)))

        # 3) Suavidade do low-pass: segunda diferença pequena
        if self.reg_smooth > 0.0 and self.kernel_size >= 3:
            d2 = h[:, 2:] - 2.0 * h[:, 1:-1] + h[:, :-2]
            self.add_loss(self.reg_smooth * tf.reduce_mean(tf.square(d2)))
        # ------------------------------------------------

        # Monta kernels depthwise: (K, C, 1)
        h_k = tf.reshape(h, [self.kernel_size, 1, 1])
        g_k = tf.reshape(g, [self.kernel_size, 1, 1])

        if isinstance(C, int):
            h_k = tf.tile(h_k, [1, C, 1])
            g_k = tf.tile(g_k, [1, C, 1])
        else:
            h_k = tf.tile(h_k, [1, tf.shape(x)[-1], 1])
            g_k = tf.tile(g_k, [1, tf.shape(x)[-1], 1])

        # Filtra e downsample (DWT)
        A_full = self._depthwise_conv1d_per_channel(x, h_k)  # (B, L, C)
        D_full = self._depthwise_conv1d_per_channel(x, g_k)  # (B, L, C)

        A = A_full[:, ::2, :]
        D = D_full[:, ::2, :]

        return A, D


__all__ = ["LearnedWaveletPair1D_QMF"]
