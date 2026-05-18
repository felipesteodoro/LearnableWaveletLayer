"""
Gerador de sinais sintéticos para validação de técnicas de deep learning.

Este módulo gera sinais ruidosos e complexos que são ideais para avaliar
técnicas de processamento de sinais como wavelets.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import json


class SyntheticSignalGenerator:
    """
    Gerador de sinais sintéticos com múltiplas componentes para regressão.
    
    O sinal gerado inclui:
    - Tendência polinomial não-linear
    - Múltiplos harmônicos com frequências variáveis
    - Ruído gaussiano
    - Spikes aleatórios (outliers)
    - Mudanças de regime (não-estacionariedade)
    - Modulação de amplitude
    
    Ideal para testar técnicas de wavelets que são boas em:
    - Separar sinal de ruído
    - Detectar mudanças abruptas
    - Capturar features multi-escala
    """
    
    def __init__(
        self,
        n_samples: int = 50000,
        random_seed: int = 42,
        **kwargs
    ):
        """
        Args:
            n_samples: Número total de amostras
            random_seed: Semente para reprodutibilidade
            **kwargs: Parâmetros adicionais do sinal
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        
        # Parâmetros do sinal
        self.trend_degree = kwargs.get("trend_degree", 2)
        self.n_harmonics = kwargs.get("n_harmonics", 5)
        self.base_frequency = kwargs.get("base_frequency", 0.01)
        self.noise_level = kwargs.get("noise_level", 0.3)
        self.spike_probability = kwargs.get("spike_probability", 0.02)
        self.spike_magnitude = kwargs.get("spike_magnitude", 3.0)
        self.regime_changes = kwargs.get("regime_changes", 3)
        
        # Armazenar componentes
        self.components: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Any] = {}
        
    def _generate_trend(self, t: np.ndarray) -> np.ndarray:
        """Gera tendência polinomial suave."""
        t_norm = t / len(t)
        coeffs = self.rng.uniform(-1, 1, self.trend_degree + 1)
        trend = np.polyval(coeffs, t_norm)
        self.metadata["trend_coeffs"] = coeffs.tolist()
        return trend
    
    def _generate_harmonics(self, t: np.ndarray) -> np.ndarray:
        """Gera soma de harmônicos com frequências variáveis."""
        signal = np.zeros_like(t, dtype=np.float64)
        harmonics_info = []
        
        for i in range(self.n_harmonics):
            freq = self.base_frequency * (i + 1) * self.rng.uniform(0.8, 1.2)
            amplitude = self.rng.uniform(0.5, 2.0) / (i + 1)
            phase = self.rng.uniform(0, 2 * np.pi)
            
            harmonic = amplitude * np.sin(2 * np.pi * freq * t + phase)
            signal += harmonic
            
            harmonics_info.append({
                "frequency": freq,
                "amplitude": amplitude,
                "phase": phase
            })
        
        self.metadata["harmonics"] = harmonics_info
        return signal
    
    def _generate_chirp(self, t: np.ndarray) -> np.ndarray:
        """Gera um chirp (frequência variável no tempo)."""
        f0 = self.base_frequency * 0.5
        f1 = self.base_frequency * 5
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * len(t)))
        chirp = 0.5 * np.sin(phase)
        self.metadata["chirp"] = {"f0": f0, "f1": f1}
        return chirp
    
    def _generate_regime_changes(self, t: np.ndarray) -> np.ndarray:
        """Gera mudanças de regime (não-estacionariedade)."""
        signal = np.zeros_like(t, dtype=np.float64)
        regime_points = np.sort(self.rng.integers(0, len(t), self.regime_changes))
        regime_points = np.concatenate([[0], regime_points, [len(t)]])
        
        regimes_info = []
        for i in range(len(regime_points) - 1):
            start, end = regime_points[i], regime_points[i + 1]
            regime_amplitude = self.rng.uniform(0.5, 2.0)
            regime_freq_mult = self.rng.uniform(0.5, 2.0)
            
            t_segment = np.arange(end - start)
            segment_signal = regime_amplitude * np.sin(
                2 * np.pi * self.base_frequency * regime_freq_mult * t_segment
            )
            signal[start:end] = segment_signal
            
            regimes_info.append({
                "start": int(start),
                "end": int(end),
                "amplitude": regime_amplitude,
                "freq_mult": regime_freq_mult
            })
        
        self.metadata["regimes"] = regimes_info
        return signal
    
    def _generate_transients(self, t: np.ndarray) -> np.ndarray:
        """Gera transientes localizados (bons para wavelets detectarem)."""
        signal = np.zeros_like(t, dtype=np.float64)
        n_transients = max(1, self.n_samples // 5000)
        transient_positions = self.rng.integers(100, len(t) - 100, n_transients)
        
        transients_info = []
        for pos in transient_positions:
            width = self.rng.integers(10, 50)
            amplitude = self.rng.uniform(1, 3)
            
            # Wavelet-like transient (Mexican hat)
            x = np.linspace(-3, 3, width)
            transient = amplitude * (1 - x**2) * np.exp(-x**2 / 2)
            
            start = max(0, pos - width // 2)
            end = min(len(t), start + width)
            actual_width = end - start
            signal[start:end] += transient[:actual_width]
            
            transients_info.append({
                "position": int(pos),
                "width": width,
                "amplitude": amplitude
            })
        
        self.metadata["transients"] = transients_info
        return signal
    
    def _generate_noise(self, t: np.ndarray) -> np.ndarray:
        """Gera ruído gaussiano."""
        noise = self.noise_level * self.rng.standard_normal(len(t))
        self.metadata["noise_level"] = self.noise_level
        return noise
    
    def _generate_spikes(self, t: np.ndarray) -> np.ndarray:
        """Gera spikes aleatórios (outliers)."""
        spikes = np.zeros_like(t, dtype=np.float64)
        spike_mask = self.rng.random(len(t)) < self.spike_probability
        spike_values = self.rng.choice([-1, 1], size=spike_mask.sum()) * \
                       self.spike_magnitude * self.rng.uniform(0.5, 1.5, spike_mask.sum())
        spikes[spike_mask] = spike_values
        
        self.metadata["n_spikes"] = int(spike_mask.sum())
        self.metadata["spike_positions"] = np.where(spike_mask)[0].tolist()[:100]  # Salvar primeiros 100
        return spikes
    
    def _generate_amplitude_modulation(self, t: np.ndarray) -> np.ndarray:
        """Gera envoltória de modulação de amplitude."""
        mod_freq = self.base_frequency * 0.1
        modulation = 1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        self.metadata["modulation_freq"] = mod_freq
        return modulation
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera o sinal sintético completo.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) onde X é o sinal ruidoso
                                            e y é o sinal limpo (target)
        """
        t = np.arange(self.n_samples, dtype=np.float64)
        
        # Gerar componentes do sinal limpo
        self.components["trend"] = self._generate_trend(t)
        self.components["harmonics"] = self._generate_harmonics(t)
        self.components["chirp"] = self._generate_chirp(t)
        self.components["regimes"] = self._generate_regime_changes(t)
        self.components["transients"] = self._generate_transients(t)
        
        # Sinal limpo (target para regressão)
        clean_signal = (
            self.components["trend"] +
            self.components["harmonics"] +
            0.5 * self.components["chirp"] +
            0.3 * self.components["regimes"] +
            self.components["transients"]
        )
        
        # Modulação de amplitude
        modulation = self._generate_amplitude_modulation(t)
        clean_signal = clean_signal * modulation
        
        # Normalizar sinal limpo
        clean_signal = (clean_signal - clean_signal.mean()) / (clean_signal.std() + 1e-8)
        
        # Adicionar ruído e spikes
        self.components["noise"] = self._generate_noise(t)
        self.components["spikes"] = self._generate_spikes(t)
        
        # Sinal ruidoso (input)
        noisy_signal = clean_signal + self.components["noise"] + self.components["spikes"]
        
        # Armazenar estatísticas
        self.metadata["signal_stats"] = {
            "clean_mean": float(clean_signal.mean()),
            "clean_std": float(clean_signal.std()),
            "noisy_mean": float(noisy_signal.mean()),
            "noisy_std": float(noisy_signal.std()),
            "snr_db": float(10 * np.log10(
                np.var(clean_signal) / (np.var(self.components["noise"]) + 1e-8)
            ))
        }
        
        return noisy_signal, clean_signal
    
    def create_regression_dataset(
        self,
        sequence_length: int = 256,
        horizon: int = 1,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria dataset para regressão com janelas deslizantes.
        
        O objetivo é prever o valor limpo do sinal dado o histórico ruidoso.
        
        Args:
            sequence_length: Comprimento de cada sequência de entrada
            horizon: Passos à frente para prever
            stride: Passo entre janelas consecutivas
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) prontos para ML/DL
        """
        noisy_signal, clean_signal = self.generate()
        
        X, y = [], []
        for i in range(0, len(noisy_signal) - sequence_length - horizon + 1, stride):
            X.append(noisy_signal[i:i + sequence_length])
            # Target: valor limpo no final da janela
            y.append(clean_signal[i + sequence_length + horizon - 1])
        
        X = np.array(X)
        y = np.array(y)
        
        self.metadata["dataset_info"] = {
            "sequence_length": sequence_length,
            "horizon": horizon,
            "stride": stride,
            "n_sequences": len(X),
            "X_shape": X.shape,
            "y_shape": y.shape
        }
        
        return X, y
    
    def save(
        self,
        save_dir: Path,
        X: np.ndarray,
        y: np.ndarray,
        prefix: str = "synthetic"
    ) -> Dict[str, Path]:
        """
        Salva o dataset e metadados.
        
        Args:
            save_dir: Diretório para salvar
            X: Features
            y: Targets
            prefix: Prefixo dos arquivos
            
        Returns:
            Dict[str, Path]: Caminhos dos arquivos salvos
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Salvar arrays
        np.save(save_dir / f"{prefix}_X.npy", X)
        np.save(save_dir / f"{prefix}_y.npy", y)
        paths["X"] = save_dir / f"{prefix}_X.npy"
        paths["y"] = save_dir / f"{prefix}_y.npy"
        
        # Salvar componentes do sinal
        for name, component in self.components.items():
            np.save(save_dir / f"{prefix}_component_{name}.npy", component)
            paths[f"component_{name}"] = save_dir / f"{prefix}_component_{name}.npy"
        
        # Salvar metadados
        with open(save_dir / f"{prefix}_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        paths["metadata"] = save_dir / f"{prefix}_metadata.json"
        
        # Salvar também como CSV para fácil visualização
        df = pd.DataFrame({
            "noisy_signal": self.components.get("noise", np.zeros(1))[:1000],
            "clean_signal": self.components.get("trend", np.zeros(1))[:1000],
        })
        df.to_csv(save_dir / f"{prefix}_sample.csv", index=False)
        paths["sample_csv"] = save_dir / f"{prefix}_sample.csv"
        
        return paths
    
    @staticmethod
    def load(save_dir: Path, prefix: str = "synthetic") -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Carrega dataset salvo.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: X, y, metadata
        """
        save_dir = Path(save_dir)
        
        X = np.load(save_dir / f"{prefix}_X.npy")
        y = np.load(save_dir / f"{prefix}_y.npy")
        
        with open(save_dir / f"{prefix}_metadata.json", "r") as f:
            metadata = json.load(f)
        
        return X, y, metadata


class RandomWalkGenerator:
    """
    Gerador de random walk puro — sinal adversário para wavelets.

    Um random walk não tem estrutura multi-escala nem transientes:
    cada passo é independente dos anteriores (processo de Markov de ordem 0).

    Por que é adversário?
    ----------------------
    Wavelets são projetados para capturar estrutura de frequência e escala.
    Em um random walk, não há frequência dominante nem padrão de escala.
    Um modelo que usa wavelet NÃO deveria ter vantagem aqui sobre um modelo
    que usa a série bruta — se tiver, é evidência de overfitting ou confundidores.

    Uso: validar que a vantagem observada do wavelet em dados reais não é
    um artefato da pipeline, testando em dados onde não deveria existir vantagem.
    """

    def __init__(
        self,
        n_samples: int = 50000,
        random_seed: int = 42,
        drift: float = 0.0,       # tendência constante (retorno esperado por passo)
        volatility: float = 1.0,  # desvio padrão dos incrementos
    ):
        self.n_samples   = n_samples
        self.random_seed = random_seed
        self.drift       = drift
        self.volatility  = volatility
        self.rng = np.random.default_rng(random_seed)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera random walk com drift e volatilidade configuráveis.

        Returns:
            (X, y): X = random walk, y = próximo passo (previsão impossível = baseline)
            O target é o próximo incremento — por construção, não é previsível.
        """
        increments = self.drift + self.volatility * self.rng.standard_normal(self.n_samples)
        walk       = np.cumsum(increments)
        # Normaliza para mesma escala do SyntheticSignalGenerator
        walk       = (walk - walk.mean()) / (walk.std() + 1e-8)
        return walk, increments  # X = walk, y = incrementos (noise puro)

    def create_regression_dataset(
        self,
        sequence_length: int = 256,
        horizon: int = 1,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dataset de regressão com janelas deslizantes no random walk."""
        walk, increments = self.generate()
        X, y = [], []
        for i in range(0, len(walk) - sequence_length - horizon + 1, stride):
            X.append(walk[i:i + sequence_length])
            y.append(increments[i + sequence_length + horizon - 1])
        return np.array(X), np.array(y)


class RegimeOnlyGenerator:
    """
    Gerador de sinal apenas com mudanças de regime (sem harmônicos, sem transientes).

    Motivação adversária
    ---------------------
    Se a vantagem do wavelet em dados reais viesse apenas da detecção de
    regime-switching (bull/bear), então esse gerador deveria exibir uma vantagem
    similar — mas de forma mais "pura" e controlada.

    O sinal gerado alterna entre K regimes com médias e volatilidades distintas.
    As transições são abruptas (Markov chain) — como choques de mercado.

    Usos:
    1. Teste de sensibilidade: quanto da vantagem do wavelet vem de regime detection?
    2. Ablação: remover harmônicos do SyntheticSignalGenerator e ver impacto no wavelet
    """

    def __init__(
        self,
        n_samples: int = 50000,
        random_seed: int = 42,
        n_regimes: int = 3,
        regime_means: Optional[list] = None,      # médias de cada regime
        regime_stds: Optional[list] = None,       # volatilidades de cada regime
        transition_prob: float = 0.01,            # prob. de mudar de regime a cada passo
    ):
        self.n_samples       = n_samples
        self.random_seed     = random_seed
        self.n_regimes       = n_regimes
        self.transition_prob = transition_prob
        self.rng = np.random.default_rng(random_seed)

        # Defaults: regimes alternando entre bear (-0.5), neutral (0), bull (+0.5)
        self.regime_means = regime_means or np.linspace(-0.5, 0.5, n_regimes).tolist()
        self.regime_stds  = regime_stds  or [0.3 + 0.2 * i for i in range(n_regimes)]

    def _generate_regime_sequence(self) -> np.ndarray:
        """Gera sequência de regimes via Markov chain homogênea."""
        regimes = np.zeros(self.n_samples, dtype=int)
        regimes[0] = self.rng.integers(0, self.n_regimes)
        for t in range(1, self.n_samples):
            if self.rng.random() < self.transition_prob:
                # Transição para outro regime aleatório
                new_regime = self.rng.integers(0, self.n_regimes)
                regimes[t] = new_regime
            else:
                regimes[t] = regimes[t - 1]
        return regimes

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera sinal regime-only.

        Returns:
            (signal, regime_labels): signal = séries com regime variável,
                                     regime_labels = qual regime em cada passo
        """
        regimes = self._generate_regime_sequence()
        signal  = np.zeros(self.n_samples)

        for t in range(self.n_samples):
            r = regimes[t]
            signal[t] = self.regime_means[r] + self.regime_stds[r] * self.rng.standard_normal()

        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        return signal, regimes.astype(np.float32)

    def create_regression_dataset(
        self,
        sequence_length: int = 256,
        horizon: int = 1,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dataset de regressão: prevê o valor do sinal horizon passos à frente."""
        signal, _ = self.generate()
        X, y = [], []
        for i in range(0, len(signal) - sequence_length - horizon + 1, stride):
            X.append(signal[i:i + sequence_length])
            y.append(signal[i + sequence_length + horizon - 1])
        return np.array(X), np.array(y)


class WhiteNoiseGenerator:
    """
    Gerador de ruído branco puro — o caso mais adversário possível para wavelets.

    Ruído branco é i.i.d. por definição — cada amostra é independente.
    Qualquer transformada (wavelet ou não) preserva essa independência.
    Um modelo que aprende com wavelet NÃO deveria ter nenhuma vantagem aqui.

    Se a acurácia wavelet > acurácia raw em ruído branco, há um bug ou
    um confundidor severo na pipeline experimental.

    Distribuições disponíveis:
    - 'gaussian': ruído gaussiano padrão (default)
    - 'laplace':  ruído de Laplace — caudas pesadas como retornos financeiros
    - 'uniform':  ruído uniforme
    """

    DISTRIBUTIONS = ("gaussian", "laplace", "uniform")

    def __init__(
        self,
        n_samples: int = 50000,
        random_seed: int = 42,
        distribution: str = "gaussian",
        scale: float = 1.0,
    ):
        assert distribution in self.DISTRIBUTIONS, f"distribution must be one of {self.DISTRIBUTIONS}"
        self.n_samples    = n_samples
        self.random_seed  = random_seed
        self.distribution = distribution
        self.scale        = scale
        self.rng = np.random.default_rng(random_seed)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera ruído branco.

        Returns:
            (noise, noise): X e y são noise independente — sem relação causal.
            A target é outro vetor de noise (previsão impossível).
        """
        if self.distribution == "gaussian":
            noise = self.scale * self.rng.standard_normal(self.n_samples)
        elif self.distribution == "laplace":
            noise = self.rng.laplace(scale=self.scale, size=self.n_samples)
        else:  # uniform
            noise = self.rng.uniform(-self.scale, self.scale, size=self.n_samples)

        target = self.rng.standard_normal(self.n_samples)  # independente de noise
        return noise.astype(np.float64), target.astype(np.float64)

    def create_regression_dataset(
        self,
        sequence_length: int = 256,
        horizon: int = 1,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dataset adversário: X = janelas de ruído, y = próximo ruído (independente)."""
        noise, target = self.generate()
        X, y = [], []
        for i in range(0, len(noise) - sequence_length - horizon + 1, stride):
            X.append(noise[i:i + sequence_length])
            y.append(target[i + sequence_length + horizon - 1])
        return np.array(X), np.array(y)


class MultiScaleSyntheticGenerator(SyntheticSignalGenerator):
    """
    Extensão que gera sinais com características multi-escala bem definidas.
    Ideal para validar capacidade de wavelets de capturar features em diferentes escalas.
    """
    
    def __init__(self, n_scales: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_scales = n_scales
    
    def _generate_multiscale_components(self, t: np.ndarray) -> np.ndarray:
        """Gera componentes em múltiplas escalas de frequência."""
        signal = np.zeros_like(t, dtype=np.float64)
        scales_info = []
        
        for scale in range(self.n_scales):
            # Frequência cresce exponencialmente com a escala
            freq = self.base_frequency * (2 ** scale)
            amplitude = 1.0 / (scale + 1)  # Amplitude decresce
            
            # Adiciona componente oscilatório
            component = amplitude * np.sin(2 * np.pi * freq * t)
            
            # Adiciona transientes localizados nesta escala
            n_events = max(1, self.n_samples // (10000 * (scale + 1)))
            event_positions = self.rng.integers(0, len(t), n_events)
            event_width = max(10, 100 // (2 ** scale))
            
            for pos in event_positions:
                start = max(0, pos - event_width // 2)
                end = min(len(t), pos + event_width // 2)
                x = np.linspace(-2, 2, end - start)
                event = 0.5 * amplitude * np.exp(-x**2)
                component[start:end] += event
            
            signal += component
            scales_info.append({
                "scale": scale,
                "frequency": freq,
                "amplitude": amplitude,
                "n_events": n_events
            })
        
        self.metadata["multiscale"] = scales_info
        return signal
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gera sinal multi-escala."""
        t = np.arange(self.n_samples, dtype=np.float64)
        
        # Componentes padrão
        self.components["trend"] = self._generate_trend(t)
        self.components["multiscale"] = self._generate_multiscale_components(t)
        self.components["transients"] = self._generate_transients(t)
        
        # Sinal limpo
        clean_signal = (
            self.components["trend"] +
            self.components["multiscale"] +
            self.components["transients"]
        )
        
        # Normalizar
        clean_signal = (clean_signal - clean_signal.mean()) / (clean_signal.std() + 1e-8)
        
        # Adicionar ruído
        self.components["noise"] = self._generate_noise(t)
        self.components["spikes"] = self._generate_spikes(t)
        
        noisy_signal = clean_signal + self.components["noise"] + self.components["spikes"]
        
        # Estatísticas
        self.metadata["signal_stats"] = {
            "clean_mean": float(clean_signal.mean()),
            "clean_std": float(clean_signal.std()),
            "noisy_mean": float(noisy_signal.mean()),
            "noisy_std": float(noisy_signal.std()),
            "snr_db": float(10 * np.log10(
                np.var(clean_signal) / (np.var(self.components["noise"]) + 1e-8)
            ))
        }
        
        return noisy_signal, clean_signal


class MultivariateSyntheticGenerator:
    """
    Gera um sinal sintético multivariado (n_channels) com target univariado e
    **conteúdo espectral heterogêneo por canal**.

    Cada canal é construído por uma instância independente de
    `SyntheticSignalGenerator`, mas usando um conjunto distinto de parâmetros
    fornecido em `channel_profiles` (uma lista de dicionários `**kwargs`, um por
    canal). Isso permite, por exemplo, que o canal 0 contenha apenas baixas
    frequências e o canal 2 apenas altas — justificando filtros wavelet
    diferentes em cada canal.

    Target univariado **não-linear** (default):
        y[i] = w·clean_last
             + alpha * (clean_last[1] * clean_last[3])
             + beta  * (clean_last[2]**2 - clean_last[4]**2)
    onde `clean_last = clean[i + L + h - 1]` é o vetor de canais limpos no fim
    da janela. Pode ser substituído passando uma `target_fn(clean_last) -> float`.

    Saídas:
      - generate() -> (noisy[T,C], clean[T,C])
      - create_regression_dataset() -> (X[N,L,C], y[N])
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_channels: int = 5,
        channel_seeds: Optional[list] = None,
        channel_profiles: Optional[list] = None,
        target_linear_weights: Optional[list] = None,
        target_alpha: float = 0.3,
        target_beta: float = 0.2,
        target_fn=None,
    ):
        self.n_samples = int(n_samples)
        self.n_channels = int(n_channels)

        self.channel_seeds = (
            list(channel_seeds)
            if channel_seeds is not None
            else [42 + i for i in range(self.n_channels)]
        )
        if len(self.channel_seeds) != self.n_channels:
            raise ValueError(
                f"channel_seeds deve ter {self.n_channels} elementos, "
                f"recebeu {len(self.channel_seeds)}"
            )

        if channel_profiles is None:
            raise ValueError(
                "channel_profiles é obrigatório: forneça uma lista de "
                f"{self.n_channels} dicts com os **kwargs de cada canal."
            )
        if len(channel_profiles) != self.n_channels:
            raise ValueError(
                f"channel_profiles deve ter {self.n_channels} elementos, "
                f"recebeu {len(channel_profiles)}"
            )
        self.channel_profiles = [dict(p) for p in channel_profiles]

        if target_linear_weights is None:
            target_linear_weights = [1.0 / self.n_channels] * self.n_channels
        w = np.array(target_linear_weights, dtype=np.float64)
        if len(w) != self.n_channels:
            raise ValueError(
                f"target_linear_weights deve ter {self.n_channels} elementos, "
                f"recebeu {len(w)}"
            )
        self.target_linear_weights = w / (w.sum() + 1e-12)
        self.target_alpha = float(target_alpha)
        self.target_beta = float(target_beta)
        self.target_fn = target_fn  # se None, usa o default não-linear abaixo

        self.channel_generators: list = []
        self.metadata: Dict[str, Any] = {
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "channel_seeds": list(self.channel_seeds),
            "channel_profiles": [dict(p) for p in self.channel_profiles],
            "target": {
                "kind": "custom_fn" if target_fn is not None else "nonlinear_default",
                "linear_weights": self.target_linear_weights.tolist(),
                "alpha": self.target_alpha,
                "beta": self.target_beta,
                "formula": (
                    "y = w·clean_last "
                    "+ alpha * (clean_last[1] * clean_last[3]) "
                    "+ beta  * (clean_last[2]**2 - clean_last[4]**2)"
                ),
            },
        }

    def _default_target(self, clean_last: np.ndarray) -> float:
        """Target não-linear padrão (válido para n_channels >= 5)."""
        w = self.target_linear_weights.astype(np.float32)
        linear = float(clean_last @ w)
        cross = float(clean_last[1] * clean_last[3])
        sqdiff = float(clean_last[2] ** 2 - clean_last[4] ** 2)
        return linear + self.target_alpha * cross + self.target_beta * sqdiff

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera o sinal multivariado heterogêneo.

        Returns:
            (noisy, clean) com shape (n_samples, n_channels) cada.
        """
        noisy_cols, clean_cols = [], []
        per_channel_meta = []

        for ch_idx, (seed, profile) in enumerate(
            zip(self.channel_seeds, self.channel_profiles)
        ):
            gen = SyntheticSignalGenerator(
                n_samples=self.n_samples,
                random_seed=int(seed),
                **profile,
            )
            noisy, clean = gen.generate()
            noisy_cols.append(noisy)
            clean_cols.append(clean)
            self.channel_generators.append(gen)
            per_channel_meta.append({
                "channel": ch_idx,
                "seed": int(seed),
                "profile": dict(profile),
                "snr_db": gen.metadata["signal_stats"]["snr_db"],
                "n_spikes": gen.metadata.get("n_spikes", 0),
            })

        noisy = np.stack(noisy_cols, axis=-1)   # (T, C)
        clean = np.stack(clean_cols, axis=-1)   # (T, C)

        self.metadata["per_channel"] = per_channel_meta
        self.metadata["noisy_shape"] = list(noisy.shape)
        self.metadata["clean_shape"] = list(clean.shape)

        if self.n_channels > 1:
            corr = np.corrcoef(clean.T)
            self.metadata["clean_channel_correlation"] = corr.tolist()

        return noisy, clean

    def create_regression_dataset(
        self,
        sequence_length: int = 256,
        horizon: int = 1,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria dataset multivariado com janelas deslizantes e target univariado.

        Returns:
            X: (N, sequence_length, n_channels) sinal ruidoso
            y: (N,) target univariado (não-linear por padrão)
        """
        noisy, clean = self.generate()
        T = noisy.shape[0]
        L, h = int(sequence_length), int(horizon)

        n_total = T - L - h + 1
        if n_total <= 0:
            raise ValueError(
                f"n_samples={T} insuficiente para sequence_length={L}, horizon={h}"
            )
        starts = np.arange(0, n_total, int(stride))

        N = len(starts)
        C = self.n_channels
        X = np.empty((N, L, C), dtype=np.float32)
        y = np.empty((N,), dtype=np.float32)

        target_fn = self.target_fn if self.target_fn is not None else self._default_target

        for k, i in enumerate(starts):
            X[k] = noisy[i:i + L].astype(np.float32)
            clean_last = clean[i + L + h - 1]
            y[k] = float(target_fn(clean_last))

        self.metadata["dataset_info"] = {
            "sequence_length": L,
            "horizon": h,
            "stride": int(stride),
            "n_sequences": int(N),
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
        }
        self.metadata["target_stats"] = {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        }

        return X, y
