"""
FinancialExperimentPipeline — runs a full PurgedKFold experiment for one job.
Called by experiment_runner.py (GPU queue) and by notebook 03_dl_experiments.

Estrutura IS/OOS:
  - IS (In-Sample): dados até (total - test_years), usados para PurgedKFold
  - OOS (Out-of-Sample): últimos test_years, avaliados em n_oos_windows janelas
    walk-forward sem retreino (zero-leak: modelo do último fold IS é usado)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

_FINANCIAL_DIR = Path(__file__).parent.parent
_ROOT = _FINANCIAL_DIR.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_FINANCIAL_DIR))

from utils.validation import PurgedKFold  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_windows(
    X: np.ndarray,
    y: np.ndarray,
    returns: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window via stride_tricks — O(1) de memória extra (view, não cópia).

    Returns
    -------
    X_win    : (N-seq_len, seq_len, F)
    y_win    : (N-seq_len,)
    ret_win  : (N-seq_len,)  — log-returns alinhados ao dia de predição
    idx_win  : (N-seq_len,)  — posições inteiras no array original (para t1)
    """
    n = len(X)
    # sliding_window_view: (N-seq_len+1, F, seq_len) — sem cópias
    view = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
    X_win   = view[:n - seq_len].transpose(0, 2, 1).copy()  # (N-seq_len, seq_len, F)
    idx_win = np.arange(seq_len, n)                          # posição do label no array original
    y_win   = y[idx_win]
    ret_win = returns[idx_win]
    return X_win, y_win, ret_win, idx_win


def _is_oos_split(
    n_total: int,
    trading_days_per_year: int = 252,
    test_years: float = 6.0,
    seq_len: int = 60,
    n_folds: int = 5,
    min_windows_per_fold: int = 10,
) -> int:
    """
    Retorna o índice de corte IS/OOS garantindo IS com janelas suficientes.

    Tudo antes do índice é IS (In-Sample), tudo a partir dele é OOS.
    Se o dataset é pequeno (ex: SUZB3.SA com ~1164 amostras), reduz o OOS
    até que IS tenha ao menos `seq_len + n_folds * min_windows_per_fold` linhas.

    Parameters
    ----------
    n_total              : total de observações diárias
    trading_days_per_year: dias de trading por ano (default 252)
    test_years           : anos preferidos para OOS (default 6)
    seq_len              : comprimento da janela (default 60)
    n_folds              : número de folds CV (default 5)
    min_windows_per_fold : mínimo de janelas por fold (default 10)

    Returns
    -------
    int: índice de início do OOS no array original
    """
    # Mínimo de linhas IS para ter janelas suficientes em todos os folds
    min_is = seq_len + n_folds * min_windows_per_fold

    oos_size = int(trading_days_per_year * test_years)
    # OOS não pode deixar IS sem linhas mínimas
    oos_size = min(oos_size, n_total - min_is)
    if oos_size <= 0:
        # Dataset muito pequeno: sem OOS, usa tudo como IS
        logger.warning(
            "Dataset com apenas %d amostras — insuficiente para IS/OOS split com "
            "test_years=%.1f. Usando IS=%.0f%% / OOS=%.0f%%.",
            n_total, test_years, 70, 30,
        )
        oos_size = max(1, int(n_total * 0.30))

    return max(min_is, n_total - oos_size)


def _compute_event_returns(
    daily_returns: np.ndarray,
    t1: "pd.Series | None",
    dates: "pd.Index",
    max_hold: int = 10,
) -> tuple[np.ndarray, float]:
    """
    Para cada dia d, retorna o retorno simples composto de d+1 até t1[d].

    Recebe `daily_returns` em retorno SIMPLES (não log). Combina via produto:
        R_evento = prod(1 + r_k, k=d+1..t1) - 1

    Isso alinha o backtest com o significado do label triple-barrier: um label "buy"
    significa que a barreira superior foi atingida em algum momento até t1 — não
    necessariamente no dia seguinte. Usar o retorno composto até t1 captura o P&L
    real da trade se ela for executada no dia d e encerrada em t1[d].

    Leakage prevention: `dates` deve conter apenas as datas do segmento atual
    (IS ou OOS). Eventos cujo t1 cai fora desse segmento são truncados no último
    dia do segmento via fallback `end_pos = min(i + max_hold, n - 1)`.

    Returns
    -------
    event_returns : array com retornos acumulados por evento
    avg_duration  : duração média dos eventos em dias de trading (para escalar RF no Sharpe)
    """
    n = len(dates)
    event_rets = daily_returns.copy().astype(np.float32)

    if t1 is None or t1.isna().all():
        return event_rets, 1.0

    date_to_pos = {d: i for i, d in enumerate(dates)}
    durations: list[int] = []

    for i, date in enumerate(dates):
        if date not in t1.index or pd.isna(t1.loc[date]):
            durations.append(1)
            continue

        t1_date = t1.loc[date]
        if t1_date in date_to_pos:
            end_pos = date_to_pos[t1_date]
        else:
            # t1 cai além do segmento — trunca ao último dia disponível
            end_pos = min(i + max_hold, n - 1)

        if end_pos > i:
            # Retorno composto simples: prod(1+r) - 1  (não soma de log-returns)
            event_rets[i] = float(np.prod(1.0 + daily_returns[i + 1: end_pos + 1]) - 1.0)
            durations.append(end_pos - i + 1)
        else:
            event_rets[i] = 0.0
            durations.append(1)

    avg_duration = float(np.mean(durations)) if durations else 1.0
    return event_rets, avg_duration


def _class_weights(y: np.ndarray) -> dict:
    """
    Pesos de classe balanceados para combater desbalanceamento sell/hold/buy.
    Em dados financeiros, 'hold' tende a ser muito mais frequente que 'buy'/'sell'.
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _replace_inf(X: np.ndarray) -> np.ndarray:
    """
    Substitui ±Inf por NaN e depois NaN pela mediana da coluna.
    Necessário para features como obv_roc_10 que produzem Inf por divisão por zero
    (ex: OBV=0 no denominador). RobustScaler rejeita Inf com ValueError.
    """
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    col_medians = np.nanmedian(X.reshape(-1, X.shape[-1]), axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[2] if X.ndim == 3
                          else np.where(nan_mask)[1])
    return X


def _scale(X_train, X_val, X_test):
    """
    RobustScaler por feature: robusto a outliers (usa mediana/IQR).
    Fit apenas em treino — nunca em val/test para evitar data leakage.
    Inf são substituídos pela mediana da coluna antes de escalar.
    """
    X_train = _replace_inf(X_train)
    X_val   = _replace_inf(X_val)
    X_test  = _replace_inf(X_test)

    orig_shape_train = X_train.shape
    orig_shape_val   = X_val.shape
    orig_shape_test  = X_test.shape
    n_feat = X_train.shape[-1]

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(orig_shape_train)
    X_v_s  = scaler.transform(X_val.reshape(-1, n_feat)).reshape(orig_shape_val)
    X_te_s = scaler.transform(X_test.reshape(-1, n_feat)).reshape(orig_shape_test)
    return X_tr_s, X_v_s, X_te_s, scaler


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class FinancialExperimentPipeline:
    """
    Full experiment for one (ticker, model_name, mode) job.

    Fluxo de dados
    --------------
    1. load_data() → X, y, returns, t1  (série temporal completa)
    2. IS/OOS split: IS = X[:cut], OOS = X[cut:]
    3. make_windows(IS) → janelas deslizantes para PurgedKFold
    4. PurgedKFold nos dados IS:
       - Purga: remove amostras de treino com t1 sobreposto ao período de teste
       - Embargo: gap entre treino e teste para evitar autocorrelação residual
    5. Para cada fold: escala, treina, avalia métricas IS
    6. Avaliação OOS: aplica o modelo do último fold IS nos dados OOS
       em n_oos_windows janelas walk-forward sem retreino

    Parameters
    ----------
    ticker      : código do ativo (e.g. "PETR4.SA")
    model_name  : CNN | LSTM | CNN_LSTM | Transformer
    mode        : raw | db4 | learned_wavelet
    config      : config dict mesclado (de experiment_config ou notebook)
    results_dir : diretório para salvar metrics.json e predições
    """

    def __init__(
        self,
        ticker: str,
        model_name: str,
        mode: str,
        config: Optional[dict] = None,
        results_dir: Optional[str] = None,
        feature_mode: str = "features",
    ):
        self.ticker       = ticker
        self.model_name   = model_name
        self.mode         = mode
        self.feature_mode = feature_mode

        from config.experiment_config import (
            DL_MODELS_CONFIG,
            DL_TRAINING_CONFIG,
            FEATURE_CONFIG,
            LABELING_CONFIG,
            LEARNED_WAVELET_CONFIG,
            VALIDATION_CONFIG,
            BACKTEST_CONFIG,
        )
        # Ordem de merge: valores mais específicos sobrescrevem os gerais.
        # FEATURE_CONFIG e DL_TRAINING_CONFIG fornecem defaults seguros.
        # (config or {}) deve conter APENAS overrides explícitos por job —
        # não deve repetir valores já definidos no experiment_config, pois
        # vem por último e sobrescreveria os defaults corretos.
        self.cfg = {
            "sequence_length": FEATURE_CONFIG["sequence_length"],  # garante sempre o valor certo
            **DL_TRAINING_CONFIG,
            **LEARNED_WAVELET_CONFIG,
            **DL_MODELS_CONFIG.get(model_name, {}),
            **VALIDATION_CONFIG,
            **LABELING_CONFIG,
            **BACKTEST_CONFIG,
            **(config or {}),
        }

        self.results_dir = Path(results_dir or (_FINANCIAL_DIR / "results"))
        self.models_dir  = self.results_dir / "saved_models"

    # ── Public ──────────────────────────────────────────────────────────────

    @property
    def job_key(self) -> str:
        return f"{self.model_name}_{self.mode}"

    @property
    def job_results_dir(self) -> Path:
        return self.results_dir / self.feature_mode / self.ticker / self.job_key

    def is_done(self) -> bool:
        return (self.job_results_dir / "metrics.json").exists()

    def run(self) -> dict:
        import tensorflow as tf
        # Semente global para reprodutibilidade end-to-end
        tf.random.set_seed(42)

        if self.is_done():
            logger.info("Already done: %s / %s — skipping.", self.ticker, self.job_key)
            with open(self.job_results_dir / "metrics.json") as f:
                return json.load(f)

        logger.info("Starting: %s / %s", self.ticker, self.job_key)
        t0 = time.time()

        # 1. Carrega série temporal completa
        X, y, returns, t1 = self._load_data()

        # 2. Split IS/OOS: IS para PurgedKFold, OOS para avaliação final
        seq_len = self.cfg["sequence_length"]
        cut = _is_oos_split(
            n_total=len(X),
            test_years=self.cfg.get("test_years", 6.0),
            seq_len=seq_len,
            n_folds=self.cfg.get("n_folds", 5),
        )
        X_is, y_is, t1_is     = X[:cut], y[:cut], (t1.iloc[:cut] if t1 is not None else None)
        X_oos, y_oos, t1_oos  = X[cut:], y[cut:], (t1.iloc[cut:] if t1 is not None else None)

        # Retornos por evento: retorno acumulado de d até t1[d] (backtest multi-day).
        # Computados separadamente em IS e OOS para evitar leakage na fronteira.
        # Fallback para retornos diários quando use_event_returns=False ou t1 ausente.
        use_ev = self.cfg.get("use_event_returns", True)
        horizon = self.cfg.get("time_horizon", 10)
        if use_ev and t1 is not None and hasattr(self, "_dates"):
            is_dates  = self._dates[:cut]
            oos_dates = self._dates[cut:]
            ret_is,  avg_dur_is  = _compute_event_returns(returns[:cut],  t1_is,  is_dates,  horizon)
            ret_oos, avg_dur_oos = _compute_event_returns(returns[cut:],  t1_oos, oos_dates, horizon)
        else:
            ret_is  = returns[:cut]
            ret_oos = returns[cut:]
            avg_dur_is = avg_dur_oos = 1.0
        # Durações médias dos eventos por segmento → escalam o custo de capital no Sharpe
        self._avg_event_duration     = avg_dur_is
        self._avg_oos_event_duration = avg_dur_oos

        logger.info("  IS: %d amostras, OOS: %d amostras (split em %d)", cut, len(X) - cut, cut)

        # 3. Janelas deslizantes sobre IS
        seq_len = self.cfg["sequence_length"]
        X_win, y_win, ret_win, idx_win = _make_windows(X_is, y_is, ret_is, seq_len)

        # t1 alinhado às janelas IS (para PurgedKFold)
        t1_win = self._align_t1(t1_is, idx_win)

        cw  = _class_weights(y_win)
        pkf = self._build_cv(t1_win, len(y_win))

        ml_fold_metrics:  list[dict] = []
        fin_fold_metrics: list[dict] = []
        # (fold_idx, model, scaler) — best Sharpe fold selected after loop
        fold_models: list[tuple] = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(pkf.split(X_win, y_win)):
            logger.info("  Fold %d/%d", fold_idx + 1, self.cfg["n_folds"])

            # Libera memória GPU entre folds — importante em runs longos (25 tickers × 4 modelos)
            tf.keras.backend.clear_session()
            tf.random.set_seed(42)

            # Divisão train/val interna (últimos val_split% do treino)
            n_val      = max(1, int(len(train_val_idx) * self.cfg.get("val_split", 0.15)))
            train_idx  = train_val_idx[:-n_val]
            val_idx    = train_val_idx[-n_val:]

            X_tr, X_v, X_te, scaler = _scale(X_win[train_idx], X_win[val_idx], X_win[test_idx])
            y_tr  = y_win[train_idx]
            y_v   = y_win[val_idx]
            y_te  = y_win[test_idx]

            model = self._build_model(X_tr.shape[1:])
            callbacks = self._callbacks(fold_idx)

            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_v, y_v),
                class_weight=cw,
                epochs=self.cfg["epochs"],
                batch_size=self.cfg["batch_size"],
                callbacks=callbacks,
                verbose=0,
            )
            best_val_loss = float(min(history.history.get("val_loss", [float("inf")])))

            y_proba     = model.predict(X_te, verbose=0)  # (N_test, 3) softmax
            y_pred      = np.argmax(y_proba, axis=1)
            # Predições no conjunto de validação — salvas para meta-labeling
            y_proba_val = model.predict(X_v, verbose=0)

            from src.evaluation import ClassificationEvaluator, FinancialMetrics
            from src.backtest import simulate_strategy

            ml_metrics = ClassificationEvaluator.evaluate(y_te, y_pred, y_proba=y_proba)
            strat_ret  = simulate_strategy(
                y_pred,
                returns=ret_win[test_idx],
                transaction_cost=self.cfg.get("transaction_cost", 0.001),
                allow_short=self.cfg.get("allow_short", True),
                position_lag=0 if use_ev else 1,
            )
            bh_ret      = pd.Series(ret_win[test_idx])
            fin_metrics = FinancialMetrics.compute(
                strat_ret, bh_ret,
                risk_free=self.cfg.get("annual_risk_free", 0.0),
                return_horizon=self._avg_event_duration,
            )

            ml_fold_metrics.append(ml_metrics)
            fin_fold_metrics.append(fin_metrics)

            self._save_predictions(
                y_true=y_te, y_pred=y_pred, y_proba=y_proba,
                filename=f"predictions_fold{fold_idx}.npz",
                X_val=X_v, y_val=y_v, y_proba_val=y_proba_val,
                ret_test=ret_win[test_idx],
            )
            fold_models.append((fold_idx, model, scaler, best_val_loss))

        # 4. Seleciona o melhor fold por Sharpe IS e avalia OOS
        # Scaler fit no IS inteiro (para consistência com OOS — não só no último fold)
        n_feat = X_win.shape[-1] if len(X_win) > 0 else 0
        full_is_scaler = RobustScaler()
        if n_feat > 0:
            full_is_scaler.fit(_replace_inf(X_win).reshape(-1, n_feat))

        # Best fold: menor val_loss de validação (critério de treino, sem snooping financeiro)
        best_fold_idx = len(fold_models) - 1  # default = último fold
        if fold_models:
            val_losses = [(i, fm[3]) for i, fm in enumerate(fold_models) if np.isfinite(fm[3])]
            if val_losses:
                best_fold_idx = min(val_losses, key=lambda x: x[1])[0]
        best_fold_val_loss = fold_models[best_fold_idx][3] if fold_models else float("nan")
        best_fold_sharpe = fin_fold_metrics[best_fold_idx].get("sharpe", float("nan")) if fin_fold_metrics else float("nan")

        # Carrega modelo do melhor fold do checkpoint em disco.
        # custom_objects é necessário para classes não registradas no serializer
        # padrão do Keras (SinusoidalPositionalEncoding, TransformerBlock, wavelets).
        best_model_obj = None
        if fold_models:
            best_keras_path = (
                self.models_dir / self.feature_mode / self.ticker
                / self.job_key / f"fold_{best_fold_idx}.keras"
            )
            if best_keras_path.exists():
                try:
                    from models.dl_utils import SinusoidalPositionalEncoding, TransformerBlock
                    from models.LWT.fixed_db4_dwt import FixedDb4DWT1D
                    from models.LWT.learned_wavelet_dwt_qmf import LearnedWaveletDWT1D_QMF
                    best_model_obj = tf.keras.models.load_model(
                        str(best_keras_path),
                        custom_objects={
                            "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
                            "TransformerBlock": TransformerBlock,
                            "FixedDb4DWT1D": FixedDb4DWT1D,
                            "LearnedWaveletDWT1D_QMF": LearnedWaveletDWT1D_QMF,
                        },
                    )
                except Exception as e:
                    logger.warning("load_model falhou (%s) — usando modelo em memória.", e)
                    _, best_model_obj, _, _ = fold_models[best_fold_idx]
            else:
                # Fallback: modelo ainda em memória (fold_models[best_fold_idx])
                _, best_model_obj, _, _ = fold_models[best_fold_idx]
            logger.info(
                "  Best fold: %d/%d  Sharpe=%.3f",
                best_fold_idx + 1, len(fold_models), best_fold_sharpe,
            )

        oos_metrics = {}
        if best_model_obj is not None and n_feat > 0 and len(X_oos) > seq_len:
            if self.cfg.get("retrain_oos", False):
                oos_metrics = self._evaluate_oos_retrain(
                    X_is, y_is, ret_is, X_oos, y_oos, ret_oos
                )
            else:
                best_model_scaler = (best_model_obj, full_is_scaler)
                oos_metrics = self._evaluate_oos(best_model_scaler, X_oos, y_oos, ret_oos, t1_oos)

        from src.evaluation import FinancialMetrics, ResultsManager
        agg_ml  = _aggregate_dicts(ml_fold_metrics)
        agg_fin = FinancialMetrics.aggregate_cv(fin_fold_metrics)
        # agg_fin has raw keys (e.g. 'sharpe'). Add fin_ prefix only for the return
        # value so notebook cell 3 can still filter on k.startswith('fin_').
        # The JSON stores raw keys; load_all_results adds fin_ prefix once cleanly.
        agg_fin_prefixed = {f"fin_{k}": v for k, v in agg_fin.items()}

        elapsed = time.time() - t0
        rm = ResultsManager(self.results_dir)
        rm.log_experiment(
            ticker=self.ticker,
            model_name=self.model_name,
            mode=self.mode,
            feature_mode=self.feature_mode,
            ml_metrics=agg_ml,
            financial_metrics={**agg_fin, **{f"oos_{k}": v for k, v in oos_metrics.items()}},
            config=self.cfg,
            extra={
                "elapsed_seconds": elapsed,
                "n_folds": self.cfg["n_folds"],
                "best_fold": best_fold_idx,
                "best_fold_val_loss": best_fold_val_loss,
                "best_fold_sharpe": best_fold_sharpe,
                "retrain_oos": self.cfg.get("retrain_oos", False),
                "avg_event_duration_is":  round(getattr(self, "_avg_event_duration", 1.0), 2),
                "avg_event_duration_oos": round(getattr(self, "_avg_oos_event_duration", 1.0), 2),
                "fold_ml_metrics": ml_fold_metrics,
                "fold_fin_metrics": fin_fold_metrics,
            },
        )

        logger.info("Done: %s / %s  (%.1fs)", self.ticker, self.job_key, elapsed)
        return {**agg_ml, **agg_fin_prefixed, **{f"oos_{k}": v for k, v in oos_metrics.items()}}

    # ── Private ─────────────────────────────────────────────────────────────

    def _load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series | None]:
        from src.data_loader import load_labels, load_t1

        if self.feature_mode == "ohlcv":
            from src.data_loader import load_ohlcv_signals
            features = load_ohlcv_signals(self.ticker)
        else:
            from src.data_loader import load_processed
            features = load_processed(self.ticker)

        labels = load_labels(self.ticker)
        t1     = load_t1(self.ticker)

        # Alinha features e labels pelo índice temporal
        common   = features.index.intersection(labels.index)
        features = features.loc[common]
        labels   = labels.loc[common]

        X = features.values.astype(np.float32)
        y = labels.values.astype(np.int32)

        # Retornos simples diários para backtest — expm1(log_return) = pct_change exato.
        # Todas as métricas financeiras (Sharpe, CAGR, MDD) assumem retorno simples.
        if self.feature_mode == "ohlcv":
            returns = np.expm1(features["log_return"].values).astype(np.float32)
        else:
            returns = (
                np.expm1(features["log_return_1"].values).astype(np.float32)
                if "log_return_1" in features.columns
                else np.zeros(len(X), dtype=np.float32)
            )

        # Alinha t1 ao índice comum
        if t1 is not None:
            t1 = t1.reindex(common)

        # Guarda o DatetimeIndex completo para que run() possa computar
        # event_returns por segmento (IS/OOS) sem leakage na fronteira.
        self._dates = common

        return X, y, returns, t1

    def _align_t1(self, t1: pd.Series | None, idx_win: np.ndarray) -> pd.Series:
        """
        Constrói Series t1 com RangeIndex alinhada às janelas windowed.

        t1 representa o fim do evento (event-end) para PurgedKFold.
        PurgedKFold usa RangeIndex inteiro para as janelas, portanto t1 também
        deve estar em coordenadas de índice de janela (0..n_windows-1).

        Quando t1 real está disponível como timestamps:
          - Para cada janela i com label na posição IS `idx_win[i]`,
            o evento termina em `t1.iloc[idx_win[i]]` (data).
          - Convertemos essa data para posição inteira no array IS via searchsorted,
            depois subtraímos seq_len para obter o índice de janela correspondente.

        Fallback: se t1 real não estiver disponível, assume que cada evento
        termina `time_horizon` janelas após o início (estimativa conservadora).
        """
        seq_len  = self.cfg["sequence_length"]
        horizon  = self.cfg.get("time_horizon", 10)
        n_windows = len(idx_win)

        if t1 is not None and not t1.isna().all():
            # t1.index é o DatetimeIndex do período IS (datas originais).
            # Para cada evento que termina em t1.iloc[idx_win[i]], encontramos
            # a posição inteira via searchsorted e convertemos para índice de janela.
            is_index   = t1.index                         # DatetimeIndex IS
            t1_dates   = pd.to_datetime(t1.iloc[idx_win].values)
            pos_in_is  = is_index.searchsorted(t1_dates, side="left")
            # Clipa para [0, n_windows-1]: eventos que terminam além do IS ficam na última janela
            t1_values  = np.clip(pos_in_is - seq_len, 0, n_windows - 1).astype(np.int64)
        else:
            # Fallback inteiro: evento termina horizon janelas após o início da janela
            t1_values = np.minimum(np.arange(n_windows) + horizon, n_windows - 1)

        return pd.Series(t1_values, index=pd.RangeIndex(n_windows))

    def _build_cv(self, t1_win: pd.Series, n_samples: int) -> PurgedKFold:
        """
        Cria PurgedKFold com embargo calculado como fração do dataset.

        pct_embargo = embargo_days / n_samples garante que o gap absoluto seja
        aproximadamente embargo_days, independentemente do tamanho do dataset.
        """
        pct_embargo = self.cfg["embargo_days"] / n_samples
        return PurgedKFold(
            n_splits=self.cfg["n_folds"],
            t1=t1_win,
            pct_embargo=pct_embargo,
        )

    def _build_model(self, input_shape: tuple):
        from src.models import build_model
        return build_model(
            model_name=self.model_name,
            mode=self.mode,
            input_shape=input_shape,
            n_classes=self.cfg.get("n_classes", 2),
            cfg=self.cfg,
        )

    def _callbacks(self, fold_idx: int) -> list:
        from src.models import get_callbacks
        model_path = (
            self.models_dir / self.feature_mode / self.ticker / self.job_key / f"fold_{fold_idx}.keras"
        )
        return get_callbacks(
            model_path=model_path,
            early_patience=self.cfg["early_stopping_patience"],
            lr_patience=self.cfg["reduce_lr_patience"],
            lr_factor=self.cfg.get("reduce_lr_factor", 0.5),
            min_lr=self.cfg.get("min_lr", 1e-6),
        )

    def _evaluate_oos_retrain(
        self,
        X_is: np.ndarray,
        y_is: np.ndarray,
        ret_is: np.ndarray,
        X_oos: np.ndarray,
        y_oos: np.ndarray,
        ret_oos: np.ndarray,
    ) -> dict:
        """
        OOS walk-forward com retreino expanding-window.

        Para cada janela OOS `i`:
          - IS expandido = IS_original ∪ OOS[0 : w_i_start]  (expanding window)
          - Refit scaler no IS expandido completo (sem leakage)
          - Retreina modelo do zero com early stopping (val = últimos 15%)
          - Prediz apenas a janela OOS atual — sem overlap, sem leakage

        Ativado via RETRAIN_OOS=1 (config["retrain_oos"] = True).
        Custo: ~n_oos_windows × custo de treino normal.
        """
        import tensorflow as tf
        from src.evaluation import ClassificationEvaluator, FinancialMetrics
        from src.backtest import simulate_strategy

        seq_len = self.cfg["sequence_length"]
        n_oos_windows = self.cfg.get("n_oos_windows", 2)

        X_oos_win, y_oos_win, ret_oos_win, _ = _make_windows(X_oos, y_oos, ret_oos, seq_len)
        if len(X_oos_win) == 0:
            return {}

        window_size = len(X_oos_win) // n_oos_windows
        if window_size < 5:
            windows = [slice(0, len(X_oos_win))]
        else:
            windows = [slice(i * window_size, (i + 1) * window_size) for i in range(n_oos_windows)]

        # raw OOS cut points (original array, before windowing)
        raw_window_size = len(X_oos) // n_oos_windows if n_oos_windows > 0 else len(X_oos)

        oos_ml_metrics:  list[dict] = []
        oos_fin_metrics: list[dict] = []
        cw = _class_weights(y_oos_win)  # approximate; recomputed per window in loop

        for w_idx, w_slice in enumerate(windows):
            # Expanding IS: IS_original + raw OOS data up to this window
            raw_oos_end = min(w_idx * raw_window_size, len(X_oos))
            X_exp = np.concatenate([X_is, X_oos[:raw_oos_end]], axis=0) if raw_oos_end > 0 else X_is
            y_exp = np.concatenate([y_is, y_oos[:raw_oos_end]], axis=0) if raw_oos_end > 0 else y_is
            ret_exp = np.concatenate([ret_is, ret_oos[:raw_oos_end]], axis=0) if raw_oos_end > 0 else ret_is

            # Rolling IS: limita a janela de treino ao último rolling_train_years * 252 dias
            if self.cfg.get("oos_protocol", "expanding") == "rolling":
                roll_size = int(self.cfg.get("rolling_train_years", 3.0) * 252)
                if len(X_exp) > roll_size:
                    X_exp   = X_exp[-roll_size:]
                    y_exp   = y_exp[-roll_size:]
                    ret_exp = ret_exp[-roll_size:]

            X_exp_win, y_exp_win, _, _ = _make_windows(X_exp, y_exp, ret_exp, seq_len)
            if len(X_exp_win) < 10:
                continue  # não há dados suficientes para treinar

            n_feat = X_exp_win.shape[-1]
            n_val  = max(1, int(len(X_exp_win) * self.cfg.get("val_split", 0.15)))
            n_train = len(X_exp_win) - n_val

            X_tr_raw = X_exp_win[:n_train]
            X_v_raw  = X_exp_win[n_train:]
            y_tr_exp = y_exp_win[:n_train]
            y_v_exp  = y_exp_win[n_train:]

            # Refit scaler no IS expandido — sem leakage de OOS
            scaler_exp = RobustScaler()
            X_tr_s = scaler_exp.fit_transform(_replace_inf(X_tr_raw).reshape(-1, n_feat)).reshape(X_tr_raw.shape)
            X_v_s  = scaler_exp.transform(_replace_inf(X_v_raw).reshape(-1, n_feat)).reshape(X_v_raw.shape)

            tf.keras.backend.clear_session()
            tf.random.set_seed(42)
            model_exp = self._build_model(X_tr_s.shape[1:])
            cw_exp = _class_weights(y_tr_exp)

            # Usa temporary path para não sobrescrever checkpoints IS
            retrain_path = (
                self.models_dir / self.feature_mode / self.ticker
                / self.job_key / f"retrain_w{w_idx}.keras"
            )
            retrain_path.parent.mkdir(parents=True, exist_ok=True)
            from src.models import get_callbacks
            cb = get_callbacks(
                model_path=retrain_path,
                early_patience=self.cfg["early_stopping_patience"],
                lr_patience=self.cfg["reduce_lr_patience"],
                lr_factor=self.cfg.get("reduce_lr_factor", 0.5),
                min_lr=self.cfg.get("min_lr", 1e-6),
            )
            model_exp.fit(
                X_tr_s, y_tr_exp,
                validation_data=(X_v_s, y_v_exp),
                class_weight=cw_exp,
                epochs=self.cfg["epochs"],
                batch_size=self.cfg["batch_size"],
                callbacks=cb,
                verbose=0,
            )

            # Predict on this OOS window
            X_w = X_oos_win[w_slice]
            y_w = y_oos_win[w_slice]
            r_w = ret_oos_win[w_slice]

            X_w_scaled = scaler_exp.transform(_replace_inf(X_w).reshape(-1, n_feat)).reshape(X_w.shape)
            y_proba = model_exp.predict(X_w_scaled, verbose=0)
            y_pred  = np.argmax(y_proba, axis=1)

            ml_m = ClassificationEvaluator.evaluate(y_w, y_pred, y_proba=y_proba)
            strat_ret = simulate_strategy(
                y_pred, returns=r_w,
                transaction_cost=self.cfg.get("transaction_cost", 0.001),
                allow_short=self.cfg.get("allow_short", True),
                position_lag=0 if self.cfg.get("use_event_returns", True) else 1,
            )
            fin_m = FinancialMetrics.compute(
                strat_ret, pd.Series(r_w),
                risk_free=self.cfg.get("annual_risk_free", 0.0),
                return_horizon=getattr(self, "_avg_event_duration", 1.0),
            )

            oos_ml_metrics.append(ml_m)
            oos_fin_metrics.append(fin_m)
            logger.info("  OOS retrain window %d/%d done", w_idx + 1, len(windows))

        if not oos_ml_metrics:
            return {}
        agg_ml  = _aggregate_dicts(oos_ml_metrics)
        agg_fin = FinancialMetrics.aggregate_cv(oos_fin_metrics)
        return {**agg_ml, **agg_fin}

    def _evaluate_oos(
        self,
        last_model_scaler: tuple,
        X_oos: np.ndarray,
        y_oos: np.ndarray,
        ret_oos: np.ndarray,
        t1_oos: pd.Series | None,
    ) -> dict:
        """
        Avalia o modelo do melhor fold IS em janelas walk-forward no OOS.

        Walk-forward: divide o OOS em n_oos_windows janelas temporais,
        avalia sequencialmente sem retreino. Isso simula o uso real do modelo
        onde ele foi treinado em dados históricos e usado em dados futuros.

        O scaler fit no IS inteiro é reutilizado — sem leakage de OOS.
        Para retreino expanding-window, use RETRAIN_OOS=1 (config['retrain_oos']).
        """
        import tensorflow as tf
        from src.evaluation import ClassificationEvaluator, FinancialMetrics
        from src.backtest import simulate_strategy

        model, scaler = last_model_scaler
        seq_len = self.cfg["sequence_length"]
        n_oos_windows = self.cfg.get("n_oos_windows", 2)

        X_win, y_win, ret_win, _ = _make_windows(X_oos, y_oos, ret_oos, seq_len)

        if len(X_win) == 0:
            return {}

        # Divide OOS em n_oos_windows janelas temporais iguais
        window_size = len(X_win) // n_oos_windows
        if window_size < 5:
            # OOS muito curto — avalia tudo de uma vez
            windows = [slice(0, len(X_win))]
        else:
            windows = [slice(i * window_size, (i + 1) * window_size) for i in range(n_oos_windows)]

        oos_ml_metrics:  list[dict] = []
        oos_fin_metrics: list[dict] = []
        all_strat_rets:  list[pd.Series] = []
        all_bh_rets:     list[pd.Series] = []
        all_y_true:      list[np.ndarray] = []
        all_y_pred:      list[np.ndarray] = []
        all_y_proba:     list[np.ndarray] = []

        n_feat = X_win.shape[-1]
        for w_idx, w_slice in enumerate(windows):
            X_w = X_win[w_slice]
            y_w = y_win[w_slice]
            r_w = ret_win[w_slice]

            # Aplica scaler IS no OOS (sem refit — evita leakage)
            X_w_scaled = scaler.transform(X_w.reshape(-1, n_feat)).reshape(X_w.shape)

            y_proba = model.predict(X_w_scaled, verbose=0)
            y_pred  = np.argmax(y_proba, axis=1)

            strat_ret = simulate_strategy(
                y_pred, returns=r_w,
                transaction_cost=self.cfg.get("transaction_cost", 0.001),
                allow_short=self.cfg.get("allow_short", True),
                position_lag=0 if self.cfg.get("use_event_returns", True) else 1,
            )

            # Métricas por janela (para _std de consistência entre janelas)
            ml_metrics  = ClassificationEvaluator.evaluate(y_w, y_pred, y_proba=y_proba)
            _oos_rh = getattr(self, "_avg_oos_event_duration", getattr(self, "_avg_event_duration", 1.0))
            fin_metrics = FinancialMetrics.compute(
                strat_ret, pd.Series(r_w),
                risk_free=self.cfg.get("annual_risk_free", 0.0),
                return_horizon=_oos_rh,
            )
            oos_ml_metrics.append(ml_metrics)
            oos_fin_metrics.append(fin_metrics)

            self._save_predictions(
                y_true=y_w, y_pred=y_pred, y_proba=y_proba,
                filename=f"predictions_oos_w{w_idx}.npz",
                ret_test=r_w, strat_ret=strat_ret,
            )

            # Acumula séries completas para métricas sobre o OOS inteiro
            all_strat_rets.append(strat_ret)
            all_bh_rets.append(pd.Series(r_w))
            all_y_true.append(y_w)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)

        # Métricas headline calculadas sobre o OOS concatenado completo.
        # mean(Sharpe_janela) ≠ Sharpe(retornos_concatenados) — o segundo é o
        # número correto para um walk-forward pois reflete a curva de equity real.
        full_strat_ret = pd.concat(all_strat_rets, ignore_index=True)
        full_bh_ret    = pd.concat(all_bh_rets,    ignore_index=True)
        full_y_true    = np.concatenate(all_y_true)
        full_y_pred    = np.concatenate(all_y_pred)
        full_y_proba   = np.concatenate(all_y_proba)

        global_ml  = ClassificationEvaluator.evaluate(full_y_true, full_y_pred, y_proba=full_y_proba)
        _oos_rh = getattr(self, "_avg_oos_event_duration", getattr(self, "_avg_event_duration", 1.0))
        global_fin = FinancialMetrics.compute(
            full_strat_ret, full_bh_ret,
            risk_free=self.cfg.get("annual_risk_free", 0.0),
            return_horizon=_oos_rh,
        )

        self._save_predictions(
            y_true=full_y_true, y_pred=full_y_pred, y_proba=full_y_proba,
            filename="predictions_oos_full.npz",
            ret_test=np.asarray(full_bh_ret), strat_ret=np.asarray(full_strat_ret),
        )

        # _std vem da variação entre janelas (consistência do modelo ao longo do tempo)
        window_std_ml  = {k: v for k, v in _aggregate_dicts(oos_ml_metrics).items() if k.endswith("_std")}
        window_std_fin = {k: v for k, v in FinancialMetrics.aggregate_cv(oos_fin_metrics).items() if k.endswith("_std")}

        return {**global_ml, **window_std_ml, **global_fin, **window_std_fin}

    def _save_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        filename: str,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        y_proba_val: np.ndarray | None = None,
        ret_test: np.ndarray | None = None,
        strat_ret: np.ndarray | None = None,
    ) -> None:
        out_dir = self.job_results_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = dict(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )
        if X_val is not None:
            arrays["X_val"] = X_val
        if y_val is not None:
            arrays["y_val"] = y_val
        if y_proba_val is not None:
            arrays["y_proba_val"] = y_proba_val
        if ret_test is not None:
            arrays["ret_test"] = ret_test
        if strat_ret is not None:
            arrays["strat_ret"] = np.asarray(strat_ret)
        np.savez_compressed(out_dir / filename, **arrays)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _aggregate_dicts(dicts: list[dict]) -> dict:
    """Mean ± std across fold dicts (skip non-numeric)."""
    if not dicts:
        return {}
    result = {}
    for key in dicts[0]:
        try:
            values = [float(d[key]) for d in dicts if not np.isnan(float(d[key]))]
            result[key]          = float(np.mean(values)) if values else float("nan")
            result[f"{key}_std"] = float(np.std(values))  if values else float("nan")
        except (TypeError, ValueError):
            result[key] = dicts[0][key]
    return result
