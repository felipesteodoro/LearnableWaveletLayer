"""
Classification and financial metrics for experiment results.
Mirrors tests/synthetic/src/evaluation.py (RegressionEvaluator → ClassificationEvaluator).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import quantstats as qs
except ImportError:
    qs = None

try:
    import empyrical
except ImportError:
    empyrical = None


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class ClassificationEvaluator:
    """Compute classification metrics for 3-class buy/sell/hold predictions."""

    CLASS_NAMES = ["sell", "hold", "buy"]

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "",
    ) -> dict[str, float]:
        p = f"{prefix}_" if prefix else ""

        # Per-class F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        f1_dict = {
            f"{p}f1_sell": float(f1_per_class[0]),
            f"{p}f1_hold": float(f1_per_class[1]),
            f"{p}f1_buy":  float(f1_per_class[2]),
        }

        # ROC-AUC (one-vs-rest, needs probability — skip if not available)
        try:
            y_true_oh = np.eye(3)[y_true.astype(int)]
            y_pred_oh = np.eye(3)[y_pred.astype(int)]
            roc_auc = float(roc_auc_score(y_true_oh, y_pred_oh, multi_class="ovr"))
        except Exception:
            roc_auc = float("nan")

        return {
            f"{p}accuracy":        float(accuracy_score(y_true, y_pred)),
            f"{p}f1_macro":        float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
            f"{p}f1_weighted":     float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            f"{p}precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            f"{p}recall_macro":    float(recall_score(y_true, y_pred, average="macro",  zero_division=0)),
            f"{p}mcc":             float(matthews_corrcoef(y_true, y_pred)),
            f"{p}roc_auc_ovr":     roc_auc,
            **f1_dict,
        }

    @staticmethod
    def evaluate_cv(
        y_true_list: list[np.ndarray],
        y_pred_list: list[np.ndarray],
    ) -> dict[str, float]:
        """Aggregate per-fold metrics (mean ± std)."""
        fold_metrics = [
            ClassificationEvaluator.evaluate(yt, yp, prefix="")
            for yt, yp in zip(y_true_list, y_pred_list)
        ]
        aggregated: dict[str, float] = {}
        for key in fold_metrics[0]:
            values = [m[key] for m in fold_metrics]
            aggregated[key]           = float(np.nanmean(values))
            aggregated[f"{key}_std"]  = float(np.nanstd(values))
        return aggregated


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

class FinancialMetrics:
    """Compute trading strategy metrics from a returns series."""

    TRADING_DAYS = 252

    @classmethod
    def compute(
        cls,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free: float = 0.0,
    ) -> dict[str, float]:
        r = strategy_returns.fillna(0).replace([np.inf, -np.inf], 0)

        if len(r) < 5:
            return cls._empty()

        ann_factor = cls.TRADING_DAYS

        # Cumulative return
        cum_ret   = float((1 + r).prod() - 1)
        n_years   = len(r) / ann_factor
        cagr      = float((1 + cum_ret) ** (1 / max(n_years, 1e-6)) - 1) if cum_ret > -1 else -1.0

        # Volatility
        vol_ann = float(r.std() * np.sqrt(ann_factor))

        # Sharpe
        excess = r - risk_free / ann_factor
        sharpe = float(excess.mean() / excess.std() * np.sqrt(ann_factor)) if excess.std() > 0 else 0.0

        # Sortino
        downside = r[r < 0]
        sortino_denom = float(downside.std() * np.sqrt(ann_factor)) if len(downside) > 1 else 1e-9
        sortino = float(r.mean() * ann_factor / sortino_denom) if sortino_denom > 0 else 0.0

        # Max drawdown
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        mdd = float(drawdown.min())

        # Calmar
        calmar = float(cagr / abs(mdd)) if mdd != 0 else 0.0

        # Win rate & profit factor
        wins   = r[r > 0]
        losses = r[r < 0]
        win_rate      = float(len(wins) / len(r[r != 0])) if len(r[r != 0]) > 0 else 0.0
        gross_profit  = float(wins.sum())
        gross_loss    = float(abs(losses.sum()))
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        avg_win       = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss      = float(losses.mean()) if len(losses) > 0 else 0.0

        # VaR / CVaR (95%)
        var_95  = float(np.percentile(r, 5))
        cvar_95 = float(r[r <= var_95].mean()) if len(r[r <= var_95]) > 0 else var_95

        result = {
            "sharpe":         sharpe,
            "sortino":        sortino,
            "calmar":         calmar,
            "max_drawdown":   mdd,
            "cagr":           cagr,
            "volatility_ann": vol_ann,
            "win_rate":       win_rate,
            "profit_factor":  profit_factor,
            "avg_win":        avg_win,
            "avg_loss":       avg_loss,
            "var_95":         var_95,
            "cvar_95":        cvar_95,
            "total_return":   cum_ret,
        }

        # Alpha / Beta vs benchmark
        if benchmark_returns is not None and len(benchmark_returns) == len(r):
            bm = benchmark_returns.fillna(0)
            cov_matrix = np.cov(r, bm)
            beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 0 else 0.0
            alpha = float(r.mean() - beta * bm.mean()) * ann_factor
            result["beta"]  = beta
            result["alpha"] = alpha

        return result

    @classmethod
    def aggregate_cv(cls, results_list: list[dict]) -> dict[str, float]:
        """Mean ± std across folds."""
        aggregated: dict[str, float] = {}
        for key in results_list[0]:
            values = [r[key] for r in results_list if not np.isnan(r.get(key, float("nan")))]
            if values:
                aggregated[f"fin_{key}"]      = float(np.mean(values))
                aggregated[f"fin_{key}_std"]  = float(np.std(values))
        return aggregated

    @classmethod
    def _empty(cls) -> dict[str, float]:
        keys = ["sharpe","sortino","calmar","max_drawdown","cagr","volatility_ann",
                "win_rate","profit_factor","avg_win","avg_loss","var_95","cvar_95","total_return"]
        return {k: float("nan") for k in keys}


# ---------------------------------------------------------------------------
# Results manager (mirrors synthetic ResultsManager)
# ---------------------------------------------------------------------------

class ResultsManager:
    """Persist experiment results in the same structure as DL jobs."""

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, ticker: str, model_name: str, mode: str) -> Path:
        d = self.results_dir / ticker / f"{model_name}_{mode}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log_experiment(
        self,
        ticker: str,
        model_name: str,
        mode: str,
        ml_metrics: dict,
        financial_metrics: dict,
        config: dict,
        extra: Optional[dict] = None,
    ) -> Path:
        job_dir = self._job_dir(ticker, model_name, mode)
        record = {
            "ticker":      ticker,
            "model_name":  model_name,
            "mode":        mode,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config":      config,
            "ml_metrics":  ml_metrics,
            "fin_metrics": financial_metrics,
            **(extra or {}),
        }
        metrics_file = job_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(record, f, indent=2, default=str)
        return metrics_file

    def save_predictions(
        self,
        ticker: str,
        model_name: str,
        mode: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_idx: int = -1,
    ) -> None:
        job_dir = self._job_dir(ticker, model_name, mode)
        suffix  = f"_fold{fold_idx}" if fold_idx >= 0 else ""
        np.savez(
            job_dir / f"predictions{suffix}.npz",
            y_true=y_true,
            y_pred=y_pred,
        )

    def load_all_results(self) -> pd.DataFrame:
        """Aggregate all metrics.json files into a summary DataFrame."""
        records = []
        for metrics_file in self.results_dir.rglob("metrics.json"):
            try:
                with open(metrics_file) as f:
                    data = json.load(f)
                flat = {
                    "ticker":     data["ticker"],
                    "model_name": data["model_name"],
                    "mode":       data["mode"],
                    **{f"ml_{k}": v for k, v in data.get("ml_metrics", {}).items()},
                    **{f"fin_{k}": v for k, v in data.get("fin_metrics", {}).items()},
                }
                records.append(flat)
            except Exception:
                pass
        return pd.DataFrame(records)

    def get_best(self, metric: str = "ml_f1_macro", top_n: int = 10) -> pd.DataFrame:
        df = self.load_all_results()
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found.")
        return df.nlargest(top_n, metric)
