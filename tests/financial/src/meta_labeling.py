"""
Meta-Labeling Pipeline (Lopez de Prado, Advances in Financial Machine Learning, Cap. 3)

Meta-labeling é uma técnica de 2 estágios:

  Estágio 1 — Modelo primário (primary model):
    Prediz a DIREÇÃO do movimento: buy (2), sell (0), hold (1).
    Pode ser qualquer classificador (ex: DL com wavelet, RandomForest, etc.)
    O modelo primário gera predições binárias direccionais.

  Estágio 2 — Meta-modelo (meta model):
    Aprende a prever SE o modelo primário vai acertar ou errar.
    Input: features originais + predição do modelo primário (como feature extra)
    Output: binário — 0 (primário vai errar) ou 1 (primário vai acertar)

  Sinal final:
    - Se meta-modelo prediz 1 (primário correto): executa trade na direção do primário
    - Se meta-modelo prediz 0 (primário errado): não executa (tamanho de posição = 0)
    - Opcionalmente: usa a probabilidade do meta-modelo como tamanho de posição fracionário

Vantagens:
  1. Separa a decisão de QUANDO apostar da decisão de QUAL DIREÇÃO apostar
  2. Pode usar um modelo simples e interpretável como primário
  3. Reduz a frequência de trading (filtra sinais de baixa confiança)
  4. Melhora o Sharpe sem comprometer a acurácia direcional

Referência: Lopez de Prado (2018), "Advances in Financial Machine Learning", Cap. 3.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

_FINANCIAL_DIR = Path(__file__).parent.parent
_ROOT = _FINANCIAL_DIR.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_FINANCIAL_DIR))

from utils.validation import PurgedKFold  # noqa: E402


# ---------------------------------------------------------------------------
# Binary meta-labels
# ---------------------------------------------------------------------------

def make_meta_labels(
    y_true: np.ndarray,
    y_pred_primary: np.ndarray,
) -> np.ndarray:
    """
    Converte labels originais (0/1/2) em meta-labels binários.

    Meta-label = 1 quando o modelo primário acertou, 0 quando errou.

    Esta conversão é feita no conjunto de treino do meta-modelo:
    o meta-modelo aprende padrões de quando o primário falha.
    Em produção, o meta-modelo recebe features + predição do primário
    e decide se vale a pena executar o trade.
    """
    return (y_pred_primary == y_true).astype(np.int32)


# ---------------------------------------------------------------------------
# Meta-Labeling Pipeline
# ---------------------------------------------------------------------------

class MetaLabelingPipeline:
    """
    Pipeline completo de meta-labeling para 3 classes (sell/hold/buy).

    Workflow
    --------
    1. Treina o modelo primário no conjunto IS com PurgedKFold
    2. Obtém predições out-of-fold do primário (para evitar overfitting do meta-modelo)
    3. Constrói features do meta-modelo: features originais + predição primária
    4. Treina o meta-modelo nas predições out-of-fold
    5. No OOS: aplica primário → aplica meta-modelo → decide posição

    Parameters
    ----------
    primary_model : classificador sklearn (ou qualquer objeto com fit/predict_proba)
        Modelo primário de direção. Deve ter predict() e predict_proba().
        Default: RandomForestClassifier balanceado.

    meta_model : classificador sklearn (binário)
        Meta-modelo de confiança. Binário: 1=execute, 0=não execute.
        Default: RandomForestClassifier balanceado.

    n_folds : int
        Número de folds no PurgedKFold para gerar predições out-of-fold.

    meta_threshold : float
        Probabilidade mínima do meta-modelo para executar o trade.
        Default 0.5 → execute se meta-modelo prediz probabilidade > 50% de acerto.

    use_fractional_sizing : bool
        Se True, tamanho de posição = probabilidade do meta-modelo (fracionário).
        Se False, posição binária: 0 ou 1 (threshold).
    """

    def __init__(
        self,
        primary_model: Optional[BaseEstimator] = None,
        meta_model: Optional[BaseEstimator] = None,
        n_folds: int = 5,
        pct_embargo: float = 0.01,
        meta_threshold: float = 0.5,
        use_fractional_sizing: bool = True,
    ):
        self.primary_model = primary_model or RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self.meta_model = meta_model or RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=10,
            class_weight="balanced",  # meta-labels são desbalanceados (mais acertos que erros)
            n_jobs=-1,
            random_state=42,
        )
        self.n_folds            = n_folds
        self.pct_embargo        = pct_embargo
        self.meta_threshold     = meta_threshold
        self.use_fractional_sizing = use_fractional_sizing

        # Estado pós-treino
        self._primary_fitted   = None
        self._meta_fitted      = None
        self._scaler           = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        t1: Optional[pd.Series] = None,
    ) -> "MetaLabelingPipeline":
        """
        Treina o pipeline completo de 2 estágios com out-of-fold cross-fitting.

        O cross-fitting (out-of-fold) é essencial para o meta-modelo:
        se o meta-modelo visse as predições do primário no mesmo dado usado
        para treinar o primário, aprenderia os padrões de overfitting do primário,
        não os padrões genuínos de quando o primário falha.

        Parameters
        ----------
        X   : (N, F) features 2D (amostras já flattenadas, sem janela temporal)
        y   : (N,) labels 0/1/2
        t1  : pd.Series com event-end para PurgedKFold (opcional)
        """
        logger.info("MetaLabeling: treinando primário com out-of-fold (%d folds)", self.n_folds)

        n = len(X)
        # Escala features: RobustScaler é robusto a outliers financeiros
        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Configura PurgedKFold
        if t1 is None:
            # Fallback: t1 = posição + 10 (sem lookahead por 10 dias)
            t1 = pd.Series(np.arange(n) + 10, index=pd.RangeIndex(n))

        pkf = PurgedKFold(
            n_splits=self.n_folds,
            t1=t1,
            pct_embargo=self.pct_embargo,
        )

        # Predições out-of-fold do modelo primário
        oof_preds   = np.full(n, -1, dtype=np.int32)
        oof_probas  = np.zeros((n, 3), dtype=np.float32)

        for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(X_scaled, y)):
            logger.debug("  Fold %d/%d", fold_idx + 1, self.n_folds)

            # Treina cópia fresca do modelo primário (evita vazamento entre folds)
            model_fold = clone(self.primary_model)
            model_fold.fit(X_scaled[train_idx], y[train_idx])

            oof_preds[test_idx]  = model_fold.predict(X_scaled[test_idx])
            if hasattr(model_fold, "predict_proba"):
                oof_probas[test_idx] = model_fold.predict_proba(X_scaled[test_idx])

        # Máscara de amostras que têm predição OOF válida
        valid_mask = oof_preds >= 0

        # Meta-labels binários: 1 = primário acertou, 0 = primário errou
        meta_y = make_meta_labels(y[valid_mask], oof_preds[valid_mask])

        # Features do meta-modelo: features originais + predição primária one-hot
        # A predição primária como feature adicional diz ao meta-modelo
        # "dado que o primário disse X, qual a probabilidade de ele estar certo?"
        primary_pred_onehot = np.eye(3)[oof_preds[valid_mask].astype(int)]  # (N, 3)
        X_meta = np.concatenate([X_scaled[valid_mask], primary_pred_onehot], axis=1)

        logger.info("MetaLabeling: treinando meta-modelo (balance: %d acertos / %d erros)",
                    meta_y.sum(), (meta_y == 0).sum())

        self._meta_fitted = clone(self.meta_model)
        self._meta_fitted.fit(X_meta, meta_y)

        # Treina modelo primário final em todos os dados IS
        self._primary_fitted = clone(self.primary_model)
        self._primary_fitted.fit(X_scaled, y)

        logger.info("MetaLabeling: treinamento concluído")
        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica pipeline de 2 estágios.

        Returns
        -------
        y_pred_primary  : (N,) predições do modelo primário (0/1/2)
        meta_proba      : (N,) probabilidade do meta-modelo de acerto
        final_positions : (N,) posições finais após filtro meta
            - Se use_fractional_sizing=True: posição ∈ [-1, +1] escalada por meta_proba
            - Se use_fractional_sizing=False: posição binária {-1, 0, +1}
        """
        if self._primary_fitted is None or self._meta_fitted is None:
            raise RuntimeError("Pipeline não treinado. Chame fit() primeiro.")

        X_scaled = self._scaler.transform(X)

        # Estágio 1: direção do primário
        y_pred_primary = self._primary_fitted.predict(X_scaled)

        # Features do meta-modelo
        primary_pred_onehot = np.eye(3)[y_pred_primary.astype(int)]
        X_meta = np.concatenate([X_scaled, primary_pred_onehot], axis=1)

        # Estágio 2: confiança do meta-modelo
        meta_proba = self._meta_fitted.predict_proba(X_meta)[:, 1]  # prob. de acerto

        # Posição final
        # Mapeamento: buy(2)→+1, hold(1)→0, sell(0)→-1
        direction = np.where(y_pred_primary == 2, 1.0,
                    np.where(y_pred_primary == 0, -1.0, 0.0))

        if self.use_fractional_sizing:
            # Posição fracionária: |posição| = meta_proba × sinal_da_direção
            # Valores próximos de 0.5 ficam perto de 0 (sem confiança), 1.0 é posição plena
            final_positions = direction * meta_proba
        else:
            # Posição binária: execute apenas se meta_proba > threshold
            execute_mask    = meta_proba > self.meta_threshold
            final_positions = direction * execute_mask.astype(float)

        return y_pred_primary, meta_proba, final_positions

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: Optional[np.ndarray] = None,
        transaction_cost: float = 0.001,
    ) -> dict:
        """
        Avaliação completa: métricas do primário + do meta-modelo + financeiras.

        Parameters
        ----------
        X               : features de teste
        y               : labels verdadeiros
        returns         : retornos diários alinhados (para backtest)
        transaction_cost: custo por trade
        """
        y_pred_primary, meta_proba, final_positions = self.predict(X)

        # Métricas do modelo primário (direção)
        primary_metrics = {
            "primary_accuracy":        float(accuracy_score(y, y_pred_primary)),
            "primary_f1_macro":        float(f1_score(y, y_pred_primary, average="macro", zero_division=0)),
            "primary_f1_weighted":     float(f1_score(y, y_pred_primary, average="weighted", zero_division=0)),
            "primary_precision_macro": float(precision_score(y, y_pred_primary, average="macro", zero_division=0)),
            "primary_recall_macro":    float(recall_score(y, y_pred_primary, average="macro", zero_division=0)),
        }

        # Métricas do meta-modelo (qual a taxa de acerto do filtro?)
        meta_y_true = make_meta_labels(y, y_pred_primary)
        meta_pred   = (meta_proba > self.meta_threshold).astype(int)
        try:
            meta_roc_auc = float(roc_auc_score(meta_y_true, meta_proba))
        except Exception:
            meta_roc_auc = float("nan")

        meta_metrics = {
            "meta_accuracy":   float(accuracy_score(meta_y_true, meta_pred)),
            "meta_f1":         float(f1_score(meta_y_true, meta_pred, zero_division=0)),
            "meta_roc_auc":    meta_roc_auc,
            "meta_avg_proba":  float(meta_proba.mean()),
            # Cobertura: fração de amostras onde o meta-modelo decide executar
            "meta_coverage":   float((meta_pred == 1).mean()),
        }

        # Métricas financeiras (backtest com posições filtradas pelo meta-modelo)
        fin_metrics = {}
        if returns is not None:
            from src.backtest import simulate_strategy
            from src.evaluation import FinancialMetrics

            # Simula com posição fracionária se disponível
            if self.use_fractional_sizing:
                r = pd.Series(returns)
                # Posição fracionária: aplica shift de 1 dia (evita look-ahead no retorno)
                strat_returns = pd.Series(final_positions).shift(1).fillna(0) * r
                # Custo de transação proporcional à variação de posição
                pos_changes = pd.Series(final_positions).diff().abs().fillna(0)
                strat_returns -= pos_changes * transaction_cost
            else:
                # Posição discreta: usa simulate_strategy padrão
                strat_returns = simulate_strategy(
                    (np.sign(final_positions) * 2 + 2 - np.sign(np.abs(final_positions))).astype(int),
                    returns=returns,
                    transaction_cost=transaction_cost,
                )

            fin_metrics = {
                f"meta_{k}": v
                for k, v in FinancialMetrics.compute(strat_returns).items()
            }

        return {**primary_metrics, **meta_metrics, **fin_metrics}


# ---------------------------------------------------------------------------
# Convenience wrapper para integração com FinancialExperimentPipeline
# ---------------------------------------------------------------------------

def run_meta_labeling_experiment(
    ticker: str,
    primary_predictions_path: Path,
    features_array: np.ndarray,
    labels_array: np.ndarray,
    returns_array: np.ndarray,
    t1: Optional[pd.Series] = None,
    n_folds: int = 5,
    meta_threshold: float = 0.5,
    use_fractional_sizing: bool = True,
    results_dir: Optional[Path] = None,
) -> dict:
    """
    Executa experimento de meta-labeling para um ticker.

    Carrega predições do modelo primário (geradas pelo FinancialExperimentPipeline),
    treina meta-modelo e avalia na divisão OOS.

    Parameters
    ----------
    ticker                    : código do ativo
    primary_predictions_path  : diretório com predictions_fold*.npz do modelo primário
    features_array            : features 2D (N, F) — sem janelas, série original
    labels_array              : labels (N,) — série original
    returns_array             : retornos diários (N,)
    t1                        : event-end para PurgedKFold
    n_folds                   : folds no meta-modelo
    meta_threshold            : limiar de confiança do meta-modelo
    use_fractional_sizing     : posição fracionária ou binária
    results_dir               : onde salvar resultados
    """
    logger.info("MetaLabeling experiment: %s", ticker)

    # Agrega predições OOF de todos os folds do primário
    pred_files = sorted(primary_predictions_path.glob("predictions_fold*.npz"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files in {primary_predictions_path}")

    oof_preds  = []
    oof_true   = []
    for pf in pred_files:
        data = np.load(pf)
        oof_preds.append(data["y_pred"])
        oof_true.append(data["y_true"])

    # Usa apenas os dados correspondentes às predições do primário
    n_used = sum(len(p) for p in oof_preds)
    if n_used > len(features_array):
        n_used = len(features_array)

    X  = features_array[:n_used]
    y  = labels_array[:n_used]
    ret = returns_array[:n_used]
    if t1 is not None:
        t1 = t1.iloc[:n_used]

    pipeline = MetaLabelingPipeline(
        n_folds=n_folds,
        meta_threshold=meta_threshold,
        use_fractional_sizing=use_fractional_sizing,
    )
    pipeline.fit(X, y, t1=t1)

    # Avaliação no conjunto de teste (último 20% temporal)
    n_test = max(30, int(0.2 * n_used))
    X_te, y_te, ret_te = X[-n_test:], y[-n_test:], ret[-n_test:]
    metrics = pipeline.evaluate(X_te, y_te, returns=ret_te)

    if results_dir is not None:
        import json
        results_dir = Path(results_dir) / ticker / "meta_labeling"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "metrics.json", "w") as f:
            json.dump({"ticker": ticker, **metrics}, f, indent=2, default=str)

    logger.info("MetaLabeling %s: primary_acc=%.3f, meta_coverage=%.2f, meta_sharpe=%.3f",
                ticker,
                metrics.get("primary_accuracy", float("nan")),
                metrics.get("meta_coverage", float("nan")),
                metrics.get("meta_sharpe", float("nan")))

    return metrics


__all__ = [
    "MetaLabelingPipeline",
    "make_meta_labels",
    "run_meta_labeling_experiment",
]
