"""
Módulo de avaliação e gerenciamento de resultados.

Implementa métricas de regressão e armazenamento organizado de resultados.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    max_error,
)
import warnings


class RegressionEvaluator:
    """
    Avaliador completo para tarefas de regressão.
    
    Calcula múltiplas métricas e fornece análise estatística dos resultados.
    """
    
    def __init__(self):
        self.metrics_functions = {
            "mse": mean_squared_error,
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error,
            "r2": r2_score,
            "explained_var": explained_variance_score,
            "max_error": max_error,
        }
        
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calcula todas as métricas de regressão.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            prefix: Prefixo para nomes das métricas
            
        Returns:
            Dict[str, float]: Dicionário com métricas
        """
        results = {}
        
        for metric_name, metric_func in self.metrics_functions.items():
            try:
                key = f"{prefix}_{metric_name}" if prefix else metric_name
                results[key] = float(metric_func(y_true, y_pred))
            except Exception as e:
                warnings.warn(f"Erro ao calcular {metric_name}: {e}")
                results[key] = np.nan
        
        # MAPE com tratamento de zeros
        try:
            key = f"{prefix}_mape" if prefix else "mape"
            mask = y_true != 0
            if mask.sum() > 0:
                results[key] = float(mean_absolute_percentage_error(
                    y_true[mask], y_pred[mask]
                ))
            else:
                results[key] = np.nan
        except Exception:
            results[key] = np.nan
        
        # Métricas adicionais
        key = f"{prefix}_residual_std" if prefix else "residual_std"
        results[key] = float(np.std(y_true - y_pred))
        
        key = f"{prefix}_residual_mean" if prefix else "residual_mean"
        results[key] = float(np.mean(y_true - y_pred))
        
        return results
    
    def evaluate_cv(
        self,
        y_true_list: List[np.ndarray],
        y_pred_list: List[np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Avalia resultados de cross-validation.
        
        Returns:
            Dict com 'per_fold', 'mean', 'std' para cada métrica
        """
        fold_results = []
        
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            fold_metrics = self.evaluate(y_true, y_pred)
            fold_results.append(fold_metrics)
        
        # Compilar estatísticas
        all_metrics = fold_results[0].keys()
        summary = {}
        
        for metric in all_metrics:
            values = [fr[metric] for fr in fold_results if not np.isnan(fr[metric])]
            if values:
                summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "per_fold": values
                }
            else:
                summary[metric] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "per_fold": []
                }
        
        return summary
    
    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Cria tabela comparativa de múltiplos modelos.
        
        Args:
            results: Dict[nome_modelo] -> Dict[métrica] -> valor
            
        Returns:
            DataFrame com comparação
        """
        df = pd.DataFrame(results).T
        df.index.name = "Model"
        return df.round(6)


class ResultsManager:
    """
    Gerenciador de resultados para armazenamento organizado.
    """
    
    def __init__(self, base_dir: Path):
        """
        Args:
            base_dir: Diretório base para salvar resultados
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments_file = self.base_dir / "experiments_log.json"
        self.summary_file = self.base_dir / "results_summary.csv"
        
        # Carregar log existente ou criar novo
        if self.experiments_file.exists():
            with open(self.experiments_file, "r") as f:
                self.experiments_log = json.load(f)
        else:
            self.experiments_log = []
    
    def log_experiment(
        self,
        experiment_name: str,
        model_name: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Registra um experimento.
        
        Returns:
            str: ID do experimento
        """
        timestamp = datetime.now().isoformat()
        experiment_id = f"{experiment_name}_{model_name}_{timestamp.replace(':', '-')}"
        
        entry = {
            "id": experiment_id,
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "model_name": model_name,
            "metrics": metrics,
            "config": config,
            "additional_info": additional_info or {}
        }
        
        self.experiments_log.append(entry)
        
        # Salvar log atualizado
        with open(self.experiments_file, "w") as f:
            json.dump(self.experiments_log, f, indent=2, default=str)
        
        # Atualizar summary CSV
        self._update_summary()
        
        return experiment_id
    
    def _update_summary(self):
        """Atualiza arquivo CSV de resumo."""
        if not self.experiments_log:
            return
        
        rows = []
        for exp in self.experiments_log:
            row = {
                "id": exp["id"],
                "timestamp": exp["timestamp"],
                "experiment": exp["experiment_name"],
                "model": exp["model_name"],
            }
            # Adicionar métricas
            for metric, value in exp["metrics"].items():
                row[metric] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.summary_file, index=False)
    
    def save_predictions(
        self,
        experiment_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None
    ):
        """Salva predições de um experimento."""
        pred_dir = self.base_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        
        np.save(pred_dir / f"{experiment_id}_y_true.npy", y_true)
        np.save(pred_dir / f"{experiment_id}_y_pred.npy", y_pred)
        if X is not None:
            np.save(pred_dir / f"{experiment_id}_X.npy", X)
    
    def save_model_weights(
        self,
        experiment_id: str,
        model: Any,
        framework: str = "keras"
    ):
        """Salva pesos do modelo."""
        models_dir = self.base_dir / "model_weights"
        models_dir.mkdir(exist_ok=True)
        
        if framework == "keras":
            model.save(models_dir / f"{experiment_id}.keras")
        elif framework == "sklearn":
            import joblib
            joblib.dump(model, models_dir / f"{experiment_id}.joblib")
        elif framework == "xgboost":
            model.save_model(models_dir / f"{experiment_id}.json")
        elif framework == "lightgbm":
            model.booster_.save_model(models_dir / f"{experiment_id}.txt")
    
    def get_best_experiment(
        self,
        metric: str = "rmse",
        lower_is_better: bool = True,
        experiment_filter: Optional[str] = None
    ) -> Dict:
        """
        Retorna o melhor experimento baseado em uma métrica.
        """
        filtered = self.experiments_log
        
        if experiment_filter:
            filtered = [e for e in filtered if experiment_filter in e["experiment_name"]]
        
        if not filtered:
            return {}
        
        # Filtrar experimentos que têm a métrica
        filtered = [e for e in filtered if metric in e["metrics"]]
        
        if not filtered:
            return {}
        
        if lower_is_better:
            best = min(filtered, key=lambda x: x["metrics"][metric])
        else:
            best = max(filtered, key=lambda x: x["metrics"][metric])
        
        return best
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Retorna DataFrame com todos os experimentos."""
        if self.summary_file.exists():
            return pd.read_csv(self.summary_file)
        return pd.DataFrame()
    
    def get_comparison_table(
        self,
        metrics: List[str] = ["rmse", "mae", "r2"],
        group_by: str = "model"
    ) -> pd.DataFrame:
        """
        Cria tabela de comparação entre modelos ou experimentos.
        """
        df = self.get_summary_dataframe()
        
        if df.empty:
            return df
        
        # Agregar por modelo
        agg_dict = {m: ["mean", "std"] for m in metrics if m in df.columns}
        
        if not agg_dict:
            return df
        
        summary = df.groupby(group_by).agg(agg_dict).round(6)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        
        return summary


class CrossValidationManager:
    """
    Gerenciador de cross-validation para séries temporais.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        gap: int = 0
    ):
        """
        Args:
            n_splits: Número de folds
            test_size: Proporção do teste em cada fold
            gap: Gap entre treino e teste (para purged CV)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def time_series_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[tuple]:
        """
        Gera splits respeitando ordem temporal.
        
        Yields:
            Tuple[train_idx, test_idx]
        """
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)
        train_size = n_samples - test_size
        
        splits = []
        fold_size = train_size // self.n_splits
        
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)
            
            if test_end <= test_start:
                continue
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def expanding_window_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_train_size: float = 0.5
    ) -> List[tuple]:
        """
        Split com janela de treino expansiva.
        """
        n_samples = len(X)
        initial_size = int(n_samples * initial_train_size)
        test_size = int(n_samples * self.test_size)
        
        splits = []
        remaining = n_samples - initial_size
        step = remaining // self.n_splits
        
        for i in range(self.n_splits):
            train_end = initial_size + i * step
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)
            
            if test_end <= test_start:
                continue
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
