"""
Módulo de avaliação e gerenciamento de resultados para classificação binária.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss,
)
import warnings


class ClassificationEvaluator:
    """
    Avaliador para tarefas de classificação binária.
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Calcula métricas de classificação.

        Args:
            y_true: Rótulos reais (0/1)
            y_pred: Rótulos preditos (0/1)
            y_prob: Probabilidades da classe positiva (para AUC-ROC)
            prefix: Prefixo para nomes das métricas
        """
        results: Dict[str, float] = {}

        def _k(name: str) -> str:
            return f"{prefix}_{name}" if prefix else name

        results[_k("accuracy")] = float(accuracy_score(y_true, y_pred))
        results[_k("f1")] = float(f1_score(y_true, y_pred, zero_division=0))
        results[_k("precision")] = float(precision_score(y_true, y_pred, zero_division=0))
        results[_k("recall")] = float(recall_score(y_true, y_pred, zero_division=0))

        if y_prob is not None:
            try:
                results[_k("auc_roc")] = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                results[_k("auc_roc")] = np.nan
            try:
                results[_k("log_loss")] = float(log_loss(y_true, y_prob))
            except ValueError:
                results[_k("log_loss")] = np.nan
        else:
            results[_k("auc_roc")] = np.nan
            results[_k("log_loss")] = np.nan

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            results[_k("specificity")] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return results

    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        df = pd.DataFrame(results).T
        df.index.name = "Model"
        return df.round(6)


class ResultsManager:
    """Gerenciador de resultados para armazenamento organizado."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.base_dir / "experiments_log.json"
        self.summary_file = self.base_dir / "results_summary.csv"

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
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        timestamp = datetime.now().isoformat()
        experiment_id = f"{experiment_name}_{model_name}_{timestamp.replace(':', '-')}"
        entry = {
            "id": experiment_id,
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "model_name": model_name,
            "metrics": metrics,
            "config": config,
            "additional_info": additional_info or {},
        }
        self.experiments_log.append(entry)
        with open(self.experiments_file, "w") as f:
            json.dump(self.experiments_log, f, indent=2, default=str)
        self._update_summary()
        return experiment_id

    def _update_summary(self):
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
            for metric, value in exp["metrics"].items():
                row[metric] = value
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.summary_file, index=False)

    def save_predictions(self, experiment_id: str, y_true: np.ndarray,
                         y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None):
        pred_dir = self.base_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        np.save(pred_dir / f"{experiment_id}_y_true.npy", y_true)
        np.save(pred_dir / f"{experiment_id}_y_pred.npy", y_pred)
        if y_prob is not None:
            np.save(pred_dir / f"{experiment_id}_y_prob.npy", y_prob)

    def save_model_weights(self, experiment_id: str, model: Any,
                           framework: str = "keras"):
        models_dir = self.base_dir / "model_weights"
        models_dir.mkdir(exist_ok=True)
        if framework == "keras":
            model.save(models_dir / f"{experiment_id}.keras")
        elif framework == "sklearn":
            import joblib
            joblib.dump(model, models_dir / f"{experiment_id}.joblib")

    def get_best_experiment(self, metric: str = "accuracy",
                            higher_is_better: bool = True,
                            experiment_filter: Optional[str] = None) -> Dict:
        filtered = self.experiments_log
        if experiment_filter:
            filtered = [e for e in filtered if experiment_filter in e["experiment_name"]]
        filtered = [e for e in filtered if metric in e["metrics"]]
        if not filtered:
            return {}
        if higher_is_better:
            return max(filtered, key=lambda x: x["metrics"][metric])
        return min(filtered, key=lambda x: x["metrics"][metric])

    def get_summary_dataframe(self) -> pd.DataFrame:
        if self.summary_file.exists():
            return pd.read_csv(self.summary_file)
        return pd.DataFrame()
