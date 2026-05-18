import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False


def _db4_low_pass(kernel_size: int) -> np.ndarray:
    """
    Retorna os coeficientes do filtro passa-baixa de db4 interpolados/truncados
    para o kernel_size solicitado.

    Se kernel_size == 8 (comprimento natural do db4), retorna os coeficientes exatos.
    Para outros tamanhos, reamosta por interpolação linear — isso não produz
    um wavelet de Daubechies perfeito, mas é um warm-start razoável para a rede.
    """
    if not _PYWT_AVAILABLE:
        # Sem PyWavelets, inicialização plana (equivale a sem warm-start)
        h = np.ones(kernel_size, dtype=np.float32)
        return h / np.linalg.norm(h)

    w = pywt.Wavelet("db4")
    h_exact = np.array(w.dec_lo, dtype=np.float32)   # comprimento = 8

    if kernel_size == len(h_exact):
        return h_exact

    # Interpolar para kernel_size diferente de 8
    x_src = np.linspace(0, 1, len(h_exact))
    x_dst = np.linspace(0, 1, kernel_size)
    h_interp = np.interp(x_dst, x_src, h_exact).astype(np.float32)
    # Renormalizar para norma L2 = 1 (condição de energia)
    return h_interp / (np.linalg.norm(h_interp) + 1e-8)


class LearnedWaveletPair1D_QMF(Layer):
    """
    Aprende apenas o filtro passa-baixa h (low-pass) e deriva o passa-alta g (high-pass)
    pela relação QMF: g[n] = (-1)^n * h[L-1-n].

    Retorna (A, D) já com downsample por 2 (estilo DWT).

    Regularizadores via self.add_loss():
      - Energia do low-pass (||h||2 ~ 1)
      - DC do high-pass (sum(g) ~ 0)
      - Suavidade do low-pass (segunda diferença pequena)

    Parâmetros
    ----------
    warm_start_db4 : bool
        Se True, inicializa a rede para produzir coeficientes próximos ao db4.
        Isso dá um ponto de partida bem melhor do que inicialização aleatória,
        pois db4 já satisfaz as condições de orthogonalidade e admissibilidade.
        O warm-start é realizado uma vez no build(); o treino posterior pode
        afastar os coeficientes do db4 se o dado demandar.
    """

    def __init__(
        self,
        kernel_size: int,
        wavelet_net_units: int = 32,
        padding: str = "SAME",
        eps: float = 1e-6,
        normalize_low: str = "l2",     # "sum1" | "l2" | "none"
        # pesos dos regularizadores
        reg_energy: float = 1e-2,
        reg_high_dc: float = 1e-2,
        reg_smooth: float = 1e-3,
        # warm-start: inicializa rede para produzir filtro próximo ao db4
        warm_start_db4: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size      = int(kernel_size)
        self.wavelet_net_units = int(wavelet_net_units)
        self.padding          = padding
        self.eps              = float(eps)
        self.normalize_low    = normalize_low
        self.reg_energy       = float(reg_energy)
        self.reg_high_dc      = float(reg_high_dc)
        self.reg_smooth       = float(reg_smooth)
        self.warm_start_db4   = warm_start_db4

    def build(self, input_shape):
        # Número de canais do problema — cada canal terá seu próprio par (h, g) aprendido.
        # Se input_shape não informar o canal (None), assumimos 1 canal por compatibilidade
        # com chamadas (B, L) sem dimensão de canal explícita.
        n_channels = input_shape[-1] if input_shape[-1] is not None else 1
        self.n_channels = int(n_channels)

        # Rede base: transforma o vetor de tempos ajustados em representação latente
        self.base_net = tf.keras.Sequential([
            Dense(self.wavelet_net_units, activation="relu"),
            Dense(self.wavelet_net_units, activation="relu"),
        ])
        # Cabeça de saída: mapeia representação latente → coeficientes do filtro,
        # produzindo K coeficientes para CADA um dos C canais (total K*C).
        self.low_head = Dense(self.kernel_size * self.n_channels)

        # Força build de base_net e low_head com inputs de shape conhecida
        # (necessário para registrar os pesos antes de acessá-los no warm-start)
        self.base_net.build((None, self.kernel_size))
        self.low_head.build((None, self.wavelet_net_units))

        # Escala e translação aprendíveis do eixo de tempo sintético:
        # t_adj = (t - translation) / scale
        # Isso permite que a rede posicione e dilate o suporte do wavelet livremente.
        self.raw_scale = self.add_weight(
            name="raw_scale", shape=(), initializer="random_normal",
            trainable=True, dtype=tf.float32
        )
        self.translation = self.add_weight(
            name="translation", shape=(), initializer="random_normal",
            trainable=True, dtype=tf.float32
        )
        # Flag de controle do warm-start (adiado para primeira chamada eager)
        self._warm_started = False
        super().build(input_shape)

        # Tenta warm-start em build(); pode falhar em modo simbólico Keras3
        # (scratch_graph), caso em que será tentado novamente na primeira
        # chamada eager (em call()). Isso evita o erro:
        #   'SymbolicTensor' object has no attribute 'numpy'
        if self.warm_start_db4:
            try:
                self._warm_start_from_db4()
                self._warm_started = True
            except (AttributeError, TypeError):
                pass  # Será refeito na primeira chamada eager

    def _warm_start_from_db4(self):
        """
        Ajusta os pesos de low_head para que a saída inicial approxime db4.

        Estratégia: usa regressão de mínimos quadrados para encontrar pesos W
        tal que base_net(t) @ W ≈ h_db4. Isso não é um treino completo —
        apenas calibra a última camada (low_head) sem alterar base_net.

        O base_net permanece aleatório, mas low_head é calibrado uma vez
        para que o ponto de partida seja mais sensato do que ruído puro.

        IMPORTANTE: toda a computação usa numpy puro — sem tf.Tensor, sem
        tf.constant, sem operações de TensorFlow. Isso é necessário porque
        build() pode ser chamado durante a construção simbólica do grafo Keras
        (compute_output_spec/scratch_graph), onde tensores TF criados localmente
        ficam "out of scope" e causam erros ao serem acessados posteriormente.
        """
        h_target = _db4_low_pass(self.kernel_size)  # (K,) numpy
        # Replica o alvo db4 para cada canal — assim cada canal parte do mesmo
        # warm-start, mas pode divergir durante o treino (filtros independentes).
        h_target_full = np.tile(h_target, self.n_channels).reshape(1, -1)  # (1, K*C)

        # Vetor de tempos em numpy
        t_np = np.linspace(-self.kernel_size // 2, self.kernel_size // 2, self.kernel_size,
                           dtype=np.float32).reshape(1, -1)    # (1, K)

        # Forward pass manual pelo base_net em numpy puro:
        #   base_net = Dense(units, relu) → Dense(units, relu)
        # Extrai pesos numpy das duas camadas Dense do Sequential
        dense1, dense2 = self.base_net.layers
        W1 = dense1.kernel.numpy()   # (K, units)
        b1 = dense1.bias.numpy()     # (units,)
        W2 = dense2.kernel.numpy()   # (units, units)
        b2 = dense2.bias.numpy()     # (units,)

        z = t_np @ W1 + b1          # (1, units)
        z = np.maximum(z, 0)        # ReLU
        z = z @ W2 + b2             # (1, units)
        z = np.maximum(z, 0)        # ReLU   → (1, wavelet_net_units)

        # Resolve z @ W = h_target_full via pseudo-inversa (mínimos quadrados)
        # W: (wavelet_net_units, kernel_size * n_channels)
        W_ls, _, _, _ = np.linalg.lstsq(z, h_target_full, rcond=None)

        # Ajusta apenas os pesos (kernel) do low_head; bias fica zero
        self.low_head.kernel.assign(W_ls.astype(np.float32))
        if self.low_head.bias is not None:
            self.low_head.bias.assign(
                np.zeros(self.kernel_size * self.n_channels, dtype=np.float32)
            )

    def _make_t(self):
        """Cria vetor de posições temporais normalizado para a rede."""
        t = tf.linspace(-self.kernel_size // 2, self.kernel_size // 2, self.kernel_size)
        t = tf.cast(t, tf.float32)
        return tf.reshape(t, [1, -1])  # (1, K)

    def _normalize_h(self, h):
        """
        Normaliza o filtro passa-baixa aprendido.
        - 'l2'  : norma L2 = 1  → preserva energia na cascata DWT
        - 'sum1': soma = 1      → preserva média (DC)
        - 'none': sem normalização
        """
        if self.normalize_low == "sum1":
            s = tf.reduce_sum(h, axis=-1, keepdims=True)
            return h / (s + self.eps)
        elif self.normalize_low == "l2":
            n = tf.norm(h, axis=-1, keepdims=True)
            return h / (n + self.eps)
        return h  # "none"

    def _qmf_from_h(self, h):
        """
        Deriva o filtro passa-alta via relação QMF:
          g[n] = (-1)^n * h[K-1-n]

        Essa relação garante que (h, g) formem um banco de filtros de espelho em
        quadratura: os dois filtros cobrem toda a banda sem sobreposição excessiva,
        e as subbandas resultantes são complementares em energia.
        """
        g = tf.reverse(h, axis=[-1])
        sign = tf.pow(-1.0, tf.cast(tf.range(self.kernel_size), tf.float32))
        sign = tf.reshape(sign, [1, -1])
        return g * sign

    def _depthwise_conv1d_per_channel(self, x, k):
        """
        Convolução depthwise 1D: aplica o mesmo kernel a cada canal independentemente.

        x: (B, L, C), k: (K, C, 1) -> (B, L, C)

        Usamos depthwise_conv2d para aproveitar o suporte otimizado do TF:
        expandimos a dimensão espacial 2 para transformar 1D → 2D fake,
        aplicamos conv e removemos a dimensão extra.
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
          A: (batch, ceil(length/2), channels)  — aproximação (low-pass)
          D: (batch, ceil(length/2), channels)  — detalhe (high-pass)
        """
        # Warm-start adiado: se build() estava em modo simbólico (Keras3 scratch_graph),
        # executa aqui na primeira chamada eager — quando os pesos já são EagerTensors.
        if self.warm_start_db4 and not self._warm_started:
            try:
                self._warm_start_from_db4()
            except Exception:
                pass  # Se falhou novamente, segue com pesos aleatórios
            self._warm_started = True  # Não tenta mais

        x = inputs
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)  # (B, L, 1)
        if len(x.shape) != 3:
            raise ValueError("inputs deve ter shape (batch, length) ou (batch, length, channels)")

        C = x.shape[-1]
        if C is None:
            C = self.n_channels

        # Gera vetor de tempo e aplica escala/translação aprendíveis.
        # Isso permite que a rede "deslize" o suporte do wavelet ao longo do eixo de tempo.
        t = self._make_t()
        scale = tf.nn.softplus(self.raw_scale) + 1e-3   # garante escala > 0
        t_adj = (t - self.translation) / scale

        # Aprende h por canal e deriva g por QMF.
        # low_head emite K*C coeficientes; reshape para (C, K) — uma linha por canal.
        z = self.base_net(t_adj)                                     # (1, hidden)
        h_flat = self.low_head(z)                                    # (1, K*C)
        h = tf.reshape(h_flat, [self.n_channels, self.kernel_size])  # (C, K)
        h = self._normalize_h(h)                                     # (C, K) — normaliza por canal
        g = self._qmf_from_h(h)                                      # (C, K)

        # --------- Regularizadores via add_loss ----------
        # Penalidades são médias sobre os canais — cada canal contribui igualmente.
        # 1) Energia do low-pass: ||h_c||_2 ~ 1 para cada canal c
        if self.reg_energy > 0.0:
            h_l2 = tf.norm(h, axis=-1)  # (C,)
            self.add_loss(self.reg_energy * tf.reduce_mean(tf.square(h_l2 - 1.0)))

        # 2) DC do high-pass: sum(g_c) ~ 0 para cada canal c
        if self.reg_high_dc > 0.0:
            g_sum = tf.reduce_sum(g, axis=-1)  # (C,)
            self.add_loss(self.reg_high_dc * tf.reduce_mean(tf.square(g_sum)))

        # 3) Suavidade do low-pass: segunda diferença pequena (por canal)
        if self.reg_smooth > 0.0 and self.kernel_size >= 3:
            d2 = h[:, 2:] - 2.0 * h[:, 1:-1] + h[:, :-2]  # (C, K-2)
            self.add_loss(self.reg_smooth * tf.reduce_mean(tf.square(d2)))
        # ------------------------------------------------

        # Monta kernels depthwise: (K, C, 1) — UM filtro distinto por canal.
        # h tem shape (C, K); transpondo para (K, C) e expandindo eixo final.
        h_k = tf.expand_dims(tf.transpose(h), axis=-1)  # (K, C, 1)
        g_k = tf.expand_dims(tf.transpose(g), axis=-1)  # (K, C, 1)

        # Filtra e downsample por 2 (DWT: stride=2 no domínio temporal)
        A_full = self._depthwise_conv1d_per_channel(x, h_k)  # (B, L, C)
        D_full = self._depthwise_conv1d_per_channel(x, g_k)  # (B, L, C)

        A = A_full[:, ::2, :]   # subsample par de amostras
        D = D_full[:, ::2, :]

        return A, D

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "kernel_size":       self.kernel_size,
            "wavelet_net_units": self.wavelet_net_units,
            "padding":           self.padding,
            "eps":               self.eps,
            "normalize_low":     self.normalize_low,
            "reg_energy":        self.reg_energy,
            "reg_high_dc":       self.reg_high_dc,
            "reg_smooth":        self.reg_smooth,
            "warm_start_db4":    self.warm_start_db4,
        })
        return cfg


__all__ = ["LearnedWaveletPair1D_QMF"]
