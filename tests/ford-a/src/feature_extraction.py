"""
Módulo de extração de features usando wavelets.

Implementa extração de features estatísticas dos coeficientes wavelet
para uso com modelos de machine learning clássicos.
"""
import numpy as np
import pywt
from typing import List, Dict, Tuple, Optional
from scipy import stats


class WaveletFeatureExtractor:
    """
    Extrator de features baseado em decomposição wavelet.

    Extrai features estatísticas dos coeficientes de aproximação e detalhe
    em múltiplos níveis de decomposição.
    """

    def __init__(
        self,
        wavelet: str = "db2",
        level: int = 2,
        mode: str = "symmetric",
        features: Optional[List[str]] = None,
    ):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

        self.available_features = [
            "mean", "std", "var", "max", "min", "range",
            "energy", "entropy", "skewness", "kurtosis",
            "median", "iqr", "mad", "rms", "peak_to_peak",
            "zero_crossings", "mean_crossings",
        ]
        self.features = features if features else self.available_features

    # ------------------------------------------------------------------
    def _compute_features(self, coeffs: np.ndarray) -> Dict[str, float]:
        """Calcula features estatísticas de um array de coeficientes."""
        feats: Dict[str, float] = {}
        if len(coeffs) == 0:
            return {f: 0.0 for f in self.features}

        if "mean" in self.features:
            feats["mean"] = float(np.mean(coeffs))
        if "std" in self.features:
            feats["std"] = float(np.std(coeffs))
        if "var" in self.features:
            feats["var"] = float(np.var(coeffs))
        if "max" in self.features:
            feats["max"] = float(np.max(coeffs))
        if "min" in self.features:
            feats["min"] = float(np.min(coeffs))
        if "range" in self.features:
            feats["range"] = float(np.max(coeffs) - np.min(coeffs))
        if "median" in self.features:
            feats["median"] = float(np.median(coeffs))
        if "energy" in self.features:
            feats["energy"] = float(np.sum(coeffs ** 2))
        if "entropy" in self.features:
            csq = coeffs ** 2
            cn = csq / (np.sum(csq) + 1e-10)
            feats["entropy"] = float(-np.sum(cn * np.log2(cn + 1e-10)))
        if "skewness" in self.features:
            feats["skewness"] = float(stats.skew(coeffs))
        if "kurtosis" in self.features:
            feats["kurtosis"] = float(stats.kurtosis(coeffs))
        if "iqr" in self.features:
            feats["iqr"] = float(np.percentile(coeffs, 75) - np.percentile(coeffs, 25))
        if "mad" in self.features:
            feats["mad"] = float(np.median(np.abs(coeffs - np.median(coeffs))))
        if "rms" in self.features:
            feats["rms"] = float(np.sqrt(np.mean(coeffs ** 2)))
        if "peak_to_peak" in self.features:
            feats["peak_to_peak"] = float(np.ptp(coeffs))
        if "zero_crossings" in self.features:
            feats["zero_crossings"] = float(np.sum(np.diff(np.sign(coeffs)) != 0))
        if "mean_crossings" in self.features:
            m = np.mean(coeffs)
            feats["mean_crossings"] = float(np.sum(np.diff(np.sign(coeffs - m)) != 0))
        return feats

    # ------------------------------------------------------------------
    def decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Decomposição wavelet multi-nível."""
        coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=self.level)
        return coeffs[0], coeffs[1:]

    def extract_features_single(self, signal: np.ndarray) -> np.ndarray:
        """Extrai features de um único sinal."""
        approx, details = self.decompose(signal)
        all_features = []
        for feat_name in sorted(self._compute_features(approx).keys()):
            all_features.append(self._compute_features(approx)[feat_name])
        for detail in details:
            df = self._compute_features(detail)
            for feat_name in sorted(df.keys()):
                all_features.append(df[feat_name])
        return np.array(all_features)

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extrai features de múltiplos sinais (n_samples, seq_len) → (n_samples, n_features)."""
        return np.array([self.extract_features_single(X[i]) for i in range(len(X))])

    def get_feature_names(self) -> List[str]:
        """Retorna nomes das features extraídas."""
        names = [f"approx_{f}" for f in sorted(self.features)]
        for level in range(self.level):
            names += [f"detail_{level+1}_{f}" for f in sorted(self.features)]
        return names

    def get_coefficients(self, X: np.ndarray) -> np.ndarray:
        """Retorna coeficientes wavelet concatenados."""
        coeffs_list = []
        for i in range(len(X)):
            approx, details = self.decompose(X[i])
            coeffs_list.append(np.concatenate([approx] + details))
        max_len = max(len(c) for c in coeffs_list)
        padded = np.zeros((len(coeffs_list), max_len))
        for i, c in enumerate(coeffs_list):
            padded[i, :len(c)] = c
        return padded

    def get_multilevel_coefficients(self, X: np.ndarray, align_length: bool = True) -> np.ndarray:
        """
        Retorna coeficientes como canais separados: (n_samples, max_length, n_levels+1).
        """
        from scipy import interpolate

        n_samples = len(X)
        n_channels = self.level + 1
        sample_approx, sample_details = self.decompose(X[0])
        max_len = max(len(sample_approx), max(len(d) for d in sample_details))
        result = np.zeros((n_samples, max_len, n_channels))

        for i in range(n_samples):
            approx, details = self.decompose(X[i])
            if align_length and len(approx) != max_len:
                x_old = np.linspace(0, 1, len(approx))
                x_new = np.linspace(0, 1, max_len)
                result[i, :, 0] = interpolate.interp1d(x_old, approx, kind='linear')(x_new)
            else:
                result[i, :len(approx), 0] = approx
            for j, detail in enumerate(details):
                if align_length and len(detail) != max_len:
                    x_old = np.linspace(0, 1, len(detail))
                    x_new = np.linspace(0, 1, max_len)
                    result[i, :, j + 1] = interpolate.interp1d(x_old, detail, kind='linear')(x_new)
                else:
                    result[i, :len(detail), j + 1] = detail
        return result


class StatisticalFeatureExtractor:
    """Extrator de features estatísticas diretamente do sinal (baseline)."""

    def __init__(self):
        self.feature_functions = {
            "mean": np.mean,
            "std": np.std,
            "var": np.var,
            "max": np.max,
            "min": np.min,
            "median": np.median,
            "skewness": stats.skew,
            "kurtosis": stats.kurtosis,
            "energy": lambda x: np.sum(x ** 2),
            "rms": lambda x: np.sqrt(np.mean(x ** 2)),
            "zcr": lambda x: np.sum(np.diff(np.sign(x)) != 0) / len(x),
        }

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        return np.array([
            [func(X[i]) for func in self.feature_functions.values()]
            for i in range(len(X))
        ])

    def get_feature_names(self) -> List[str]:
        return list(self.feature_functions.keys())
