"""
Módulo de avaliação e gerenciamento de resultados para classificação
MULTICLASSE (UWaveGestureLibrary — 8 gestos).
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
)


class ClassificationEvaluator:
    """
    Avaliador para tarefas de classificação multiclasse.

    Métricas (macro pondera todas as classes igualmente):
      accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, auc_ovr.
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Calcula métricas de classificação multiclasse.

        Args:
            y_true:  rótulos reais (inteiros 0..n_classes-1)
            y_pred:  rótulos preditos (inteiros 0..n_classes-1)
            y_proba: matriz de probabilidades (n_samples, n_classes) para AUC-OvR
            prefix:  prefixo para nomes das métricas (ex: "test")
        """
        results: Dict[str, float] = {}

        def _k(name: str) -> str:
            return f"{prefix}_{name}" if prefix else name

        results[_k("accuracy")] = float(accuracy_score(y_true, y_pred))
        results[_k("f1_macro")] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        results[_k("f1_weighted")] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        results[_k("precision_macro")] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        results[_k("recall_macro")] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

        # AUC one-vs-rest (macro). Requer matriz de probabilidade e todas as
        # classes presentes em y_true; caso contrário → NaN.
        if y_proba is not None:
            try:
                results[_k("auc_ovr")] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
            except ValueError:
                results[_k("auc_ovr")] = float("nan")
        else:
            results[_k("auc_ovr")] = float("nan")

        return results

    @staticmethod
    def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
        """Matriz de confusão n_classes × n_classes."""
        return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

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
                         y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None):
        pred_dir = self.base_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        np.save(pred_dir / f"{experiment_id}_y_true.npy", y_true)
        np.save(pred_dir / f"{experiment_id}_y_pred.npy", y_pred)
        if y_proba is not None:
            np.save(pred_dir / f"{experiment_id}_y_proba.npy", y_proba)

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
