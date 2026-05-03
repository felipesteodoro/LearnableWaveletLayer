"""
Classification and financial metrics for experiment results.
Mirrors tests/synthetic/src/evaluation.py (RegressionEvaluator → ClassificationEvaluator).

Inclui:
- ClassificationEvaluator: acurácia, F1, MCC, ROC-AUC OvR
- FinancialMetrics: Sharpe, Sortino, Calmar, MDD, CAGR, VaR, CVaR
- WilcoxonComparison: teste estatístico para comparar modos (raw/db4/learned_wavelet)
- ResultsManager: persistência e carregamento de resultados de experimentos
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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
    """Compute classification metrics for 2-class or 3-class predictions."""

    CLASS_NAMES = ["sell", "hold", "buy"]

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "",
        y_proba: np.ndarray | None = None,
    ) -> dict[str, float]:
        p = f"{prefix}_" if prefix else ""

        # Detectar número de classes a partir dos dados
        unique_labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        is_binary = len(unique_labels) <= 2

        # Per-class F1 — usa apenas as classes presentes
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
        if is_binary:
            # Meta-labeling binário: 0=down(sell), 1=up(buy); hold=0.0
            f1_dict = {
                f"{p}f1_sell": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
                f"{p}f1_hold": 0.0,
                f"{p}f1_buy":  float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            }
        else:
            f1_dict = {
                f"{p}f1_sell": float(f1_per_class[0]),
                f"{p}f1_hold": float(f1_per_class[1]),
                f"{p}f1_buy":  float(f1_per_class[2]),
            }

        # ROC-AUC — binário: AUC padrão; 3-class: one-vs-rest
        roc_auc = float("nan")
        if y_proba is not None:
            try:
                if is_binary:
                    # Para binário usa apenas probabilidade da classe positiva (col 1)
                    proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    roc_auc = float(roc_auc_score(y_true, proba_pos))
                else:
                    y_true_oh = np.eye(3)[y_true.astype(int)]
                    roc_auc = float(roc_auc_score(y_true_oh, y_proba, multi_class="ovr"))
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
        y_proba_list: list[np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Aggregate per-fold metrics (mean ± std)."""
        fold_metrics = [
            ClassificationEvaluator.evaluate(
                yt, yp,
                prefix="",
                y_proba=yprob if y_proba_list is not None else None,
            )
            for yt, yp, yprob in zip(
                y_true_list,
                y_pred_list,
                y_proba_list if y_proba_list is not None else [None] * len(y_true_list),
            )
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
        return_horizon: float = 1.0,
    ) -> dict[str, float]:
        """
        Parameters
        ----------
        return_horizon : average number of trading days each return observation spans.
            1.0 for daily returns (default).
            For event-based returns (multi-day holds), pass avg_event_duration so
            annualization and risk-free scaling use the effective number of
            observations per year instead of assuming daily data.
        """
        r = strategy_returns.fillna(0).replace([np.inf, -np.inf], 0)
        r = r.clip(lower=-0.99)  # previne (1+r) <= 0 no cumprod (overflow)

        if len(r) < 5:
            return cls._empty()

        horizon = max(float(return_horizon), 1e-9)
        ann_factor = cls.TRADING_DAYS / horizon

        # Risk-free per observation (scaled for multi-day event returns)
        rf_per_obs = risk_free * horizon / cls.TRADING_DAYS

        # Cumulative return
        cum_ret   = float((1 + r).prod() - 1)
        n_years   = len(r) / ann_factor
        cagr      = float((1 + cum_ret) ** (1 / max(n_years, 1e-6)) - 1) if cum_ret > -1 else -1.0

        # Volatility
        vol_ann = float(r.std() * np.sqrt(ann_factor))

        # Sharpe
        excess = r - rf_per_obs
        sharpe = float(excess.mean() / excess.std() * np.sqrt(ann_factor)) if excess.std() > 0 else 0.0

        # Sortino: (R_p - R_f) / σ_downside
        daily_rf = rf_per_obs
        downside = r[r < daily_rf]
        sortino_denom = float(downside.std() * np.sqrt(ann_factor)) if len(downside) > 1 else 1e-9
        sortino = float((r.mean() - daily_rf) * ann_factor / sortino_denom) if sortino_denom > 0 else 0.0

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

        # Alpha de Jensen vs benchmark: α = (R_p - R_f) - β(R_bm - R_f)
        if benchmark_returns is not None and len(benchmark_returns) == len(r):
            bm = benchmark_returns.fillna(0)
            cov_matrix = np.cov(r, bm)
            beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 0 else 0.0
            alpha = float((r.mean() - rf_per_obs) - beta * (bm.mean() - rf_per_obs)) * ann_factor
            result["beta"]  = beta
            result["alpha"] = alpha

            # BH Sharpe: Sharpe do benchmark (buy-and-hold) com mesma rf_per_obs
            bh_excess = bm - rf_per_obs
            result["bh_sharpe"] = (
                float(bh_excess.mean() / bh_excess.std() * np.sqrt(ann_factor))
                if bh_excess.std() > 0 else 0.0
            )

        return result

    @classmethod
    def aggregate_cv(cls, results_list: list[dict]) -> dict[str, float]:
        """Mean ± std across folds."""
        aggregated: dict[str, float] = {}
        for key in results_list[0]:
            values = [r[key] for r in results_list if not np.isnan(r.get(key, float("nan")))]
            if values:
                aggregated[key]           = float(np.mean(values))
                aggregated[f"{key}_std"]  = float(np.std(values))
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

    def _job_dir(self, ticker: str, model_name: str, mode: str, feature_mode: str = "features") -> Path:
        d = self.results_dir / feature_mode / ticker / f"{model_name}_{mode}"
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
        feature_mode: str = "features",
        extra: Optional[dict] = None,
    ) -> Path:
        job_dir = self._job_dir(ticker, model_name, mode, feature_mode)
        record = {
            "ticker":       ticker,
            "model_name":   model_name,
            "mode":         mode,
            "feature_mode": feature_mode,
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config":       config,
            "ml_metrics":   ml_metrics,
            "fin_metrics":  financial_metrics,
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

    @staticmethod
    def _norm_fin_key(k: str) -> str:
        """
        Normalise a fin_metrics key to a uniform `fin_*` column name.

        Handles two on-disk formats:
          Legacy (aggregate_cv used to add fin_ prefix itself):
            fin_sharpe       → fin_sharpe
            oos_fin_sharpe   → fin_oos_sharpe
            oos_f1_macro     → fin_oos_f1_macro
          New (aggregate_cv returns raw keys, load_all_results adds fin_):
            sharpe           → fin_sharpe
            oos_sharpe       → fin_oos_sharpe
            oos_f1_macro     → fin_oos_f1_macro
        """
        # Legacy: oos_fin_* → rewrite to oos_* before adding fin_
        if k.startswith("oos_fin_"):
            k = "oos_" + k[len("oos_fin_"):]
        # Strip leading fin_ to avoid double prefix (legacy IS metrics)
        elif k.startswith("fin_"):
            k = k[len("fin_"):]
        return f"fin_{k}"

    def load_all_results(self) -> pd.DataFrame:
        """Aggregate all metrics.json files into a summary DataFrame.

        Handles both legacy format (fin_metrics keys already contain fin_ prefix)
        and new format (fin_metrics keys are raw, e.g. 'sharpe').
        """
        records = []
        for metrics_file in self.results_dir.rglob("metrics.json"):
            try:
                with open(metrics_file) as f:
                    data = json.load(f)
                flat = {
                    "ticker":       data["ticker"],
                    "model_name":   data["model_name"],
                    "mode":         data["mode"],
                    "feature_mode": data.get("feature_mode", "features"),
                    "best_fold":    data.get("best_fold"),
                    "best_fold_sharpe": data.get("best_fold_sharpe"),
                    "retrain_oos":  data.get("retrain_oos", False),
                    **{f"ml_{k}": v for k, v in data.get("ml_metrics", {}).items()},
                    **{self._norm_fin_key(k): v for k, v in data.get("fin_metrics", {}).items()},
                }
                # Merge meta_metrics.json if it exists alongside metrics.json
                meta_file = metrics_file.parent / "meta_metrics.json"
                if meta_file.exists():
                    try:
                        with open(meta_file) as mf:
                            meta = json.load(mf)
                        flat.update({f"meta_{k}": v for k, v in meta.items()
                                     if not isinstance(v, (dict, list))})
                    except Exception:
                        pass
                records.append(flat)
            except Exception:
                pass
        return pd.DataFrame(records)

    def get_best(self, metric: str = "ml_f1_macro", top_n: int = 10) -> pd.DataFrame:
        df = self.load_all_results()
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found.")
        return df.nlargest(top_n, metric)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test para comparação entre modos
# ---------------------------------------------------------------------------

class WilcoxonComparison:
    """
    Teste de Wilcoxon signed-rank para comparar dois modos de experimento.

    Por que Wilcoxon e não t-test?
    --------------------------------
    O t-test assume que as diferenças de performance entre modos seguem uma
    distribuição normal. Em finanças, as métricas (Sharpe, F1) costumam ser
    assimétricas e com caudas pesadas — a premissa de normalidade é violada.

    O teste de Wilcoxon é não-paramétrico: compara os postos (ranks) das
    diferenças em vez dos valores absolutos. Ele requer apenas que as diferenças
    sejam simétricas em torno da mediana, uma suposição muito mais fraca.

    Uso padrão: comparar 'learned_wavelet' vs 'raw' em todos os folds/ativos.
    Se p-value < 0.05: rejeita H0 (diferença estatisticamente significativa).

    Limitações:
    - Requer pares de observações (mesmos folds/ativos nos dois modos)
    - Com poucos pares (< 10), o poder do teste é baixo
    - Não distingue magnitude do efeito — use effect_size para isso
    """

    @staticmethod
    def compare(
        scores_a: list[float] | np.ndarray,
        scores_b: list[float] | np.ndarray,
        alternative: str = "two-sided",
    ) -> dict:
        """
        Compara duas sequências de scores com o teste de Wilcoxon.

        H0: mediana(scores_a - scores_b) = 0
        H1 (two-sided): mediana ≠ 0
        H1 (greater): mediana(a) > mediana(b)

        Parameters
        ----------
        scores_a    : scores do modo A (ex: learned_wavelet) — one per fold/ticker
        scores_b    : scores do modo B (ex: raw)
        alternative : 'two-sided' | 'greater' | 'less'

        Returns
        -------
        dict com: statistic, p_value, significant (bool, α=0.05),
                  median_diff, mean_diff, effect_size (rank-biserial correlation)
        """
        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)

        # Remove pares onde qualquer valor é NaN
        valid = ~(np.isnan(a) | np.isnan(b))
        a, b = a[valid], b[valid]

        if len(a) < 3:
            # Pares insuficientes para o teste
            return {
                "statistic":   float("nan"),
                "p_value":     float("nan"),
                "significant": False,
                "median_diff": float(np.median(a - b)) if len(a) > 0 else float("nan"),
                "mean_diff":   float(np.mean(a - b))   if len(a) > 0 else float("nan"),
                "effect_size": float("nan"),
                "n_pairs":     int(len(a)),
                "note":        "Insufficient pairs for Wilcoxon test (need >= 3)",
            }

        try:
            stat, p_val = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        except Exception as e:
            return {
                "statistic": float("nan"), "p_value": float("nan"),
                "significant": False, "note": str(e),
            }

        # Effect size: rank-biserial correlation ∈ [-1, +1]
        # |r| = 0.1 pequeno, 0.3 médio, 0.5 grande (Cohen, 1988)
        n = len(a)
        effect_size = 1 - (2 * stat) / (n * (n + 1) / 2) if n > 0 else float("nan")

        return {
            "statistic":   float(stat),
            "p_value":     float(p_val),
            "significant": bool(p_val < 0.05),
            "median_diff": float(np.median(a - b)),
            "mean_diff":   float(np.mean(a - b)),
            "effect_size": float(effect_size),
            "n_pairs":     int(n),
        }

    @staticmethod
    def compare_modes(
        results_df: pd.DataFrame,
        metric: str = "ml_f1_macro",
        mode_col: str = "mode",
        baseline_mode: str = "raw",
        comparison_modes: Optional[list[str]] = None,
        group_col: Optional[str] = "ticker",
    ) -> pd.DataFrame:
        """
        Compara múltiplos modos wavelet vs um baseline usando Wilcoxon.

        Agrupa por ticker (ou outro group_col) para obter pares de observações,
        então aplica Wilcoxon para cada par (comparison_mode vs baseline_mode).

        Parameters
        ----------
        results_df        : DataFrame com colunas [mode_col, group_col, metric]
        metric            : nome da coluna de métrica a comparar
        mode_col          : coluna que identifica o modo (ex: "mode")
        baseline_mode     : modo de referência (ex: "raw")
        comparison_modes  : modos a comparar vs baseline (default: todos exceto baseline)
        group_col         : coluna de agrupamento para criar pares (ex: "ticker")
                           Se None, usa os índices como pares

        Returns
        -------
        DataFrame com uma linha por comparison_mode, colunas do dict de WilcoxonComparison
        """
        if comparison_modes is None:
            comparison_modes = [m for m in results_df[mode_col].unique() if m != baseline_mode]

        records = []
        for cmp_mode in comparison_modes:
            if group_col and group_col in results_df.columns:
                # Cria pares alinhados por grupo (ticker)
                base_df = results_df[results_df[mode_col] == baseline_mode].set_index(group_col)[metric]
                cmp_df  = results_df[results_df[mode_col] == cmp_mode].set_index(group_col)[metric]

                # Alinha por grupos comuns
                common_groups = base_df.index.intersection(cmp_df.index)
                scores_base = base_df.loc[common_groups].values
                scores_cmp  = cmp_df.loc[common_groups].values
            else:
                # Sem agrupamento: usa todos os valores como pares na ordem
                scores_base = results_df[results_df[mode_col] == baseline_mode][metric].values
                scores_cmp  = results_df[results_df[mode_col] == cmp_mode][metric].values
                min_len = min(len(scores_base), len(scores_cmp))
                scores_base = scores_base[:min_len]
                scores_cmp  = scores_cmp[:min_len]

            result = WilcoxonComparison.compare(scores_cmp, scores_base, alternative="two-sided")
            records.append({
                "comparison":   f"{cmp_mode}_vs_{baseline_mode}",
                "mode_a":       cmp_mode,
                "mode_b":       baseline_mode,
                "metric":       metric,
                **result,
            })

        return pd.DataFrame(records)

    @staticmethod
    def summary_table(
        results_df: pd.DataFrame,
        metrics: Optional[list[str]] = None,
        mode_col: str = "mode",
        group_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Tabela resumo de todas as comparações Wilcoxon para múltiplas métricas.

        Útil para o relatório final: mostra quais métricas são significativamente
        diferentes entre modos wavelet e o baseline raw.
        """
        if metrics is None:
            metrics = ["ml_f1_macro", "ml_accuracy", "ml_mcc", "ml_roc_auc_ovr"]
            # Filtra apenas métricas que existem no DataFrame
            metrics = [m for m in metrics if m in results_df.columns]

        all_records = []
        for metric in metrics:
            comparison_df = WilcoxonComparison.compare_modes(
                results_df, metric=metric, mode_col=mode_col, group_col=group_col
            )
            all_records.append(comparison_df)

        return pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
