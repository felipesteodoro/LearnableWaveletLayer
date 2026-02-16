"""
Módulo de extração de features usando wavelets.

Implementa extração de features estatísticas dos coeficientes wavelet
para uso com modelos de machine learning clássicos.
"""
import numpy as np
import pywt
from typing import List, Dict, Tuple, Optional, Union
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
        level: int = 4,
        mode: str = "symmetric",
        features: Optional[List[str]] = None
    ):
        """
        Args:
            wavelet: Nome da wavelet (db2, db4, haar, sym4, etc.)
            level: Número de níveis de decomposição
            mode: Modo de extensão de borda
            features: Lista de features a extrair (None = todas)
        """
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        
        self.available_features = [
            "mean", "std", "var", "max", "min", "range",
            "energy", "entropy", "skewness", "kurtosis",
            "median", "iqr", "mad", "rms", "peak_to_peak",
            "zero_crossings", "mean_crossings"
        ]
        
        self.features = features if features else self.available_features
        
    def _compute_features(self, coeffs: np.ndarray) -> Dict[str, float]:
        """
        Calcula features estatísticas de um array de coeficientes.
        """
        features = {}
        
        if len(coeffs) == 0:
            return {f: 0.0 for f in self.features}
        
        # Features básicas
        if "mean" in self.features:
            features["mean"] = float(np.mean(coeffs))
        if "std" in self.features:
            features["std"] = float(np.std(coeffs))
        if "var" in self.features:
            features["var"] = float(np.var(coeffs))
        if "max" in self.features:
            features["max"] = float(np.max(coeffs))
        if "min" in self.features:
            features["min"] = float(np.min(coeffs))
        if "range" in self.features:
            features["range"] = float(np.max(coeffs) - np.min(coeffs))
        if "median" in self.features:
            features["median"] = float(np.median(coeffs))
            
        # Energia e entropia
        if "energy" in self.features:
            features["energy"] = float(np.sum(coeffs ** 2))
        if "entropy" in self.features:
            # Shannon entropy dos coeficientes normalizados
            coeffs_sq = coeffs ** 2
            coeffs_norm = coeffs_sq / (np.sum(coeffs_sq) + 1e-10)
            features["entropy"] = float(-np.sum(coeffs_norm * np.log2(coeffs_norm + 1e-10)))
            
        # Momentos estatísticos
        if "skewness" in self.features:
            features["skewness"] = float(stats.skew(coeffs))
        if "kurtosis" in self.features:
            features["kurtosis"] = float(stats.kurtosis(coeffs))
            
        # Features robustas
        if "iqr" in self.features:
            features["iqr"] = float(np.percentile(coeffs, 75) - np.percentile(coeffs, 25))
        if "mad" in self.features:
            features["mad"] = float(np.median(np.abs(coeffs - np.median(coeffs))))
        if "rms" in self.features:
            features["rms"] = float(np.sqrt(np.mean(coeffs ** 2)))
        if "peak_to_peak" in self.features:
            features["peak_to_peak"] = float(np.ptp(coeffs))
            
        # Features de contagem
        if "zero_crossings" in self.features:
            features["zero_crossings"] = float(np.sum(np.diff(np.sign(coeffs)) != 0))
        if "mean_crossings" in self.features:
            mean_val = np.mean(coeffs)
            features["mean_crossings"] = float(np.sum(np.diff(np.sign(coeffs - mean_val)) != 0))
            
        return features
    
    def decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Realiza decomposição wavelet do sinal.
        
        Args:
            signal: Sinal 1D
            
        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: (aproximação, [detalhes])
        """
        coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=self.level)
        approx = coeffs[0]
        details = coeffs[1:]
        return approx, details
    
    def extract_features_single(self, signal: np.ndarray) -> np.ndarray:
        """
        Extrai features de um único sinal.
        
        Args:
            signal: Sinal 1D
            
        Returns:
            np.ndarray: Vetor de features
        """
        approx, details = self.decompose(signal)
        
        all_features = []
        
        # Features da aproximação
        approx_features = self._compute_features(approx)
        for feat_name in sorted(approx_features.keys()):
            all_features.append(approx_features[feat_name])
        
        # Features de cada nível de detalhe
        for level_idx, detail in enumerate(details):
            detail_features = self._compute_features(detail)
            for feat_name in sorted(detail_features.keys()):
                all_features.append(detail_features[feat_name])
        
        return np.array(all_features)
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extrai features de múltiplos sinais.
        
        Args:
            X: Array de sinais (n_samples, sequence_length)
            
        Returns:
            np.ndarray: Matriz de features (n_samples, n_features)
        """
        features_list = []
        for i in range(len(X)):
            features = self.extract_features_single(X[i])
            features_list.append(features)
        
        return np.array(features_list)
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nomes das features extraídas.
        """
        names = []
        
        # Features da aproximação
        for feat in sorted(self.features):
            names.append(f"approx_{feat}")
        
        # Features de cada nível de detalhe
        for level in range(self.level):
            for feat in sorted(self.features):
                names.append(f"detail_{level+1}_{feat}")
        
        return names
    
    def get_coefficients(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna coeficientes wavelet concatenados (para uso com DL).
        
        Args:
            X: Array de sinais (n_samples, sequence_length)
            
        Returns:
            np.ndarray: Coeficientes (n_samples, n_coeffs)
        """
        coeffs_list = []
        
        for i in range(len(X)):
            approx, details = self.decompose(X[i])
            # Concatenar todos os coeficientes
            all_coeffs = np.concatenate([approx] + details)
            coeffs_list.append(all_coeffs)
        
        # Padding para mesmo tamanho
        max_len = max(len(c) for c in coeffs_list)
        padded = np.zeros((len(coeffs_list), max_len))
        for i, c in enumerate(coeffs_list):
            padded[i, :len(c)] = c
        
        return padded
    
    def get_multilevel_coefficients(
        self, 
        X: np.ndarray,
        align_length: bool = True
    ) -> np.ndarray:
        """
        Retorna coeficientes como canais separados para cada nível.
        
        Formato: (n_samples, max_length, n_levels + 1)
        Onde os canais são [approx, detail_1, detail_2, ..., detail_L]
        
        Args:
            X: Array de sinais (n_samples, sequence_length)
            align_length: Se True, interpola para alinhar comprimentos
            
        Returns:
            np.ndarray: Coeficientes multi-canal
        """
        from scipy import interpolate
        
        n_samples = len(X)
        n_channels = self.level + 1
        
        # Encontrar comprimento máximo após decomposição
        sample_approx, sample_details = self.decompose(X[0])
        max_len = max(len(sample_approx), max(len(d) for d in sample_details))
        
        result = np.zeros((n_samples, max_len, n_channels))
        
        for i in range(n_samples):
            approx, details = self.decompose(X[i])
            
            # Aproximação
            if align_length and len(approx) != max_len:
                x_old = np.linspace(0, 1, len(approx))
                x_new = np.linspace(0, 1, max_len)
                f = interpolate.interp1d(x_old, approx, kind='linear')
                result[i, :, 0] = f(x_new)
            else:
                result[i, :len(approx), 0] = approx
            
            # Detalhes
            for j, detail in enumerate(details):
                if align_length and len(detail) != max_len:
                    x_old = np.linspace(0, 1, len(detail))
                    x_new = np.linspace(0, 1, max_len)
                    f = interpolate.interp1d(x_old, detail, kind='linear')
                    result[i, :, j + 1] = f(x_new)
                else:
                    result[i, :len(detail), j + 1] = detail
        
        return result


class StatisticalFeatureExtractor:
    """
    Extrator de features estatísticas diretamente do sinal (sem wavelet).
    Para comparação como baseline.
    """
    
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
        """
        Extrai features estatísticas de múltiplos sinais.
        """
        features_list = []
        for i in range(len(X)):
            signal = X[i]
            features = [func(signal) for func in self.feature_functions.values()]
            features_list.append(features)
        return np.array(features_list)
    
    def get_feature_names(self) -> List[str]:
        return list(self.feature_functions.keys())
