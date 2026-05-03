"""
GlobalFinancialModel — modelo cross-asset treinado em todos os 25 ativos B3.

Motivação
---------
Treinar um modelo separado por ativo (25 modelos) tem duas desvantagens:
  1. Cada modelo vê apenas ~10-15 anos de dados de um único ativo → overfitting
  2. Padrões comuns ao mercado brasileiro (beta, momentum, correlações setoriais)
     são aprendidos 25 vezes de forma redundante

O modelo global:
  - Recebe features de qualquer ativo + embedding do ticker
  - Aprende padrões compartilhados (ex: efeito momentum em todo o mercado)
  - E padrões específicos por ativo via o embedding
  - Treina em 25× mais dados → melhor generalização
  - Um único modelo a manter em produção

Arquitetura
-----------
  Input: (B, seq_len, n_features)       — janela temporal de features
  Ticker embedding: (n_tickers, emb_dim) — lookup por ID do ativo

  x = wavelet_frontend(input)            — (B, seq_len, proj_dim)
  e = embedding_lookup(ticker_id)        — (B, emb_dim)
  e = tile_along_time(e)                 — (B, seq_len, emb_dim)
  x = concat([x, e], axis=-1)            — (B, seq_len, proj_dim + emb_dim)
  x = backbone(x)                        — (B, hidden)
  output = Dense(3, softmax)             — (B, 3)

O ticker embedding permite que o modelo aprenda:
  "Este padrão de features é bullish para VALE3, mas bearish para COGN3"
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


# ---------------------------------------------------------------------------
# Keras model com ticker embedding
# ---------------------------------------------------------------------------

def build_global_model(
    n_tickers: int,
    input_shape: tuple,        # (seq_len, n_features)
    mode: str = "learned_wavelet",
    backbone: str = "CNN",
    n_classes: int = 3,
    emb_dim: int = 16,
    cfg: Optional[dict] = None,
):
    """
    Constrói modelo global com embedding de ticker.

    Parameters
    ----------
    n_tickers   : número de ativos distintos (tamanho do vocabulário de embedding)
    input_shape : (seq_len, n_features)
    mode        : raw | db4 | learned_wavelet
    backbone    : CNN | LSTM | CNN_LSTM | Transformer
    emb_dim     : dimensão do embedding de ticker
                  Valores menores (8-16) forçam o modelo a aprender
                  representações compactas e generalizáveis por ativo.
                  Valores maiores permitem mais especificidade por ativo.
    cfg         : dict de configuração (de experiment_config)
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers
    except ImportError:
        raise ImportError("TensorFlow not found.")

    from src.models import _apply_wavelet_frontend, _BACKBONES

    if cfg is None:
        from config.experiment_config import (
            DL_MODELS_CONFIG, DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG
        )
        cfg = {**DL_MODELS_CONFIG.get(backbone, {}), **DL_TRAINING_CONFIG, **LEARNED_WAVELET_CONFIG}

    # --- Input 1: janela de features ---
    features_input = keras.Input(shape=input_shape, name="features")
    x = _apply_wavelet_frontend(features_input, mode, cfg)

    # --- Input 2: ID inteiro do ticker ---
    ticker_input = keras.Input(shape=(1,), dtype=tf.int32, name="ticker_id")

    # Embedding: (B, 1, emb_dim) → (B, emb_dim)
    # L2 regularização no embedding evita que ativos raros aprendam representações esparsas
    emb = layers.Embedding(
        input_dim=n_tickers,
        output_dim=emb_dim,
        embeddings_regularizer=regularizers.l2(1e-4),
        name="ticker_embedding",
    )(ticker_input)                   # (B, 1, emb_dim)
    emb = layers.Flatten()(emb)       # (B, emb_dim)

    # Replica o embedding ao longo do eixo temporal para concatenar com features
    # Usamos uma Layer customizada em vez de Lambda para evitar problemas com
    # KerasTensors no modo simbólico (Lambda + tf.shape em Keras functional API).
    class TileEmbedding(layers.Layer):
        def call(self, inputs):
            emb_vec, feat = inputs
            # emb_vec: (B, emb_dim), feat: (B, seq_len, proj_dim)
            seq = tf.shape(feat)[1]
            return tf.tile(tf.expand_dims(emb_vec, axis=1), [1, seq, 1])

    emb_tiled = TileEmbedding(name="tile_embedding")([emb, x])

    # Concatena features processadas + embedding do ativo
    x = layers.Concatenate(axis=-1, name="features_with_embedding")([x, emb_tiled])

    # Backbone compartilhado (aprende padrões comuns a todos os ativos)
    backbone_fn = _BACKBONES.get(backbone)
    if backbone_fn is None:
        raise ValueError(f"Unknown backbone: {backbone}")
    x = backbone_fn(x, cfg)

    # Cabeça de classificação
    output = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(
        inputs=[features_input, ticker_input],
        outputs=output,
        name=f"GlobalModel_{backbone}_{mode}",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.get("learning_rate", 1e-3)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Dataset builder — agrega todos os ativos em um único array
# ---------------------------------------------------------------------------

def build_global_dataset(
    tickers: list[str],
    seq_len: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """
    Carrega e agrega dados de todos os ativos.

    Retorna arrays globais onde cada amostra sabe de qual ativo veio
    através do ticker_ids array.

    Returns
    -------
    X_global       : (N_total, seq_len, n_features)
    y_global       : (N_total,)
    ret_global     : (N_total,)
    ticker_ids     : (N_total,) int — índice do ticker para embedding
    dates_global   : (N_total,) — data de cada amostra (para alinhamento temporal)
    ticker_to_id   : dict mapeando ticker_str → int_id
    """
    from src.data_loader import load_processed, load_labels
    from tests.financial.src.pipeline import _make_windows

    ticker_to_id: dict[str, int] = {t: i for i, t in enumerate(tickers)}

    X_parts, y_parts, ret_parts, tid_parts = [], [], [], []
    n_features = None

    for ticker in tickers:
        try:
            features = load_processed(ticker)
            labels   = load_labels(ticker)
        except Exception as e:
            logger.warning("Skipping %s: %s", ticker, e)
            continue

        common   = features.index.intersection(labels.index)
        features = features.loc[common]
        labels   = labels.loc[common]

        X_raw = features.values.astype(np.float32)
        y_raw = labels.values.astype(np.int32)
        ret_raw = (
            np.expm1(features["log_return_1"].values).astype(np.float32)
            if "log_return_1" in features.columns
            else np.zeros(len(X_raw), dtype=np.float32)
        )

        if n_features is None:
            n_features = X_raw.shape[1]
        elif X_raw.shape[1] != n_features:
            # Ativos com número diferente de features são pulados para manter shape uniforme
            logger.warning("Skipping %s: %d features != expected %d", ticker, X_raw.shape[1], n_features)
            continue

        # Escala cada ativo independentemente para remover diferenças de escala inter-ativos
        scaler = RobustScaler()
        X_raw  = scaler.fit_transform(X_raw)

        X_win, y_win, ret_win, _ = _make_windows(X_raw, y_raw, ret_raw, seq_len)

        tid_win = np.full(len(X_win), ticker_to_id[ticker], dtype=np.int32)

        X_parts.append(X_win)
        y_parts.append(y_win)
        ret_parts.append(ret_win)
        tid_parts.append(tid_win)

        logger.info("  Loaded %s: %d windows", ticker, len(X_win))

    if not X_parts:
        raise RuntimeError("Nenhum ativo carregado com sucesso.")

    X_global   = np.concatenate(X_parts, axis=0).astype(np.float32)
    y_global   = np.concatenate(y_parts, axis=0).astype(np.int32)
    ret_global = np.concatenate(ret_parts, axis=0).astype(np.float32)
    tid_global = np.concatenate(tid_parts, axis=0).astype(np.int32)

    logger.info("Dataset global: %d amostras, %d features, %d ativos",
                len(X_global), n_features or 0, len(X_parts))

    return X_global, y_global, ret_global, tid_global, ticker_to_id


# ---------------------------------------------------------------------------
# GlobalFinancialPipeline
# ---------------------------------------------------------------------------

class GlobalFinancialPipeline:
    """
    Pipeline do modelo global cross-asset.

    Treina um único DL model em todos os 25 ativos simultaneamente,
    usando ticker embedding para capturar heterogeneidade entre ativos.

    Comparação com experimento individual
    --------------------------------------
    - Individual: 25 modelos × (3 modos × 4 backbones) = 300 jobs
    - Global: 1 modelo × (3 modos × 4 backbones) = 12 jobs
      → Análise de se o modelo global supera a média dos modelos individuais

    Parameters
    ----------
    tickers    : lista de ativos
    mode       : raw | db4 | learned_wavelet
    backbone   : CNN | LSTM | CNN_LSTM | Transformer
    emb_dim    : dimensão do embedding de ticker
    config     : dict de configuração
    results_dir: onde salvar métricas
    """

    def __init__(
        self,
        tickers: list[str],
        mode: str = "learned_wavelet",
        backbone: str = "CNN",
        emb_dim: int = 16,
        config: Optional[dict] = None,
        results_dir: Optional[str] = None,
    ):
        self.tickers     = tickers
        self.mode        = mode
        self.backbone    = backbone
        self.emb_dim     = emb_dim

        from config.experiment_config import (
            DL_MODELS_CONFIG, DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG, VALIDATION_CONFIG, BACKTEST_CONFIG
        )
        self.cfg = {
            **DL_TRAINING_CONFIG,
            **LEARNED_WAVELET_CONFIG,
            **DL_MODELS_CONFIG.get(backbone, {}),
            **VALIDATION_CONFIG,
            **BACKTEST_CONFIG,
            **(config or {}),
        }
        self.results_dir = Path(results_dir or (_FINANCIAL_DIR / "results" / "global"))

    @property
    def job_key(self) -> str:
        return f"Global_{self.backbone}_{self.mode}"

    def is_done(self) -> bool:
        return (self.results_dir / f"{self.job_key}_metrics.json").exists()

    def run(self) -> dict:
        import tensorflow as tf
        tf.random.set_seed(42)

        if self.is_done():
            logger.info("Already done: %s — skipping.", self.job_key)
            with open(self.results_dir / f"{self.job_key}_metrics.json") as f:
                return json.load(f)

        logger.info("Starting global experiment: %s", self.job_key)
        t0 = time.time()

        # 1. Constrói dataset global
        seq_len = self.cfg["sequence_length"]
        X_global, y_global, ret_global, tid_global, ticker_to_id = build_global_dataset(
            self.tickers, seq_len=seq_len,
        )

        n_tickers  = len(ticker_to_id)
        n_features = X_global.shape[-1]

        logger.info("Dataset global: N=%d, tickers=%d, features=%d", len(X_global), n_tickers, n_features)

        # 2. IS/OOS temporal split (por posição absoluta no array global)
        # NOTA: como misturamos ativos, o split temporal aqui usa 80/20 simples.
        # O PurgedKFold por ativo individual é mais rigoroso — aqui fazemos uma
        # avaliação complementar no espírito de cross-asset generalization.
        n_total   = len(X_global)
        n_is      = int(0.8 * n_total)
        X_is, y_is, ret_is, tid_is   = X_global[:n_is], y_global[:n_is], ret_global[:n_is], tid_global[:n_is]
        X_oos, y_oos, ret_oos, tid_oos = X_global[n_is:], y_global[n_is:], ret_global[n_is:], tid_global[n_is:]

        # 3. Pesos de classe globais
        classes = np.unique(y_is)
        class_weights_arr = compute_class_weight("balanced", classes=classes, y=y_is)
        cw = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}

        # 4. Split train/val interno
        n_val      = max(1, int(n_is * 0.15))
        train_idx  = slice(0, n_is - n_val)
        val_idx    = slice(n_is - n_val, n_is)

        X_tr, y_tr, tid_tr = X_is[train_idx], y_is[train_idx], tid_is[train_idx]
        X_v,  y_v,  tid_v  = X_is[val_idx],   y_is[val_idx],   tid_is[val_idx]

        # 5. Constrói e treina modelo global
        tf.keras.backend.clear_session()
        model = build_global_model(
            n_tickers=n_tickers,
            input_shape=(seq_len, n_features),
            mode=self.mode,
            backbone=self.backbone,
            emb_dim=self.emb_dim,
            cfg=self.cfg,
        )

        from src.models import get_callbacks
        callbacks = get_callbacks(
            model_path=self.results_dir / f"{self.job_key}_best.keras",
            early_patience=self.cfg["early_stopping_patience"],
            lr_patience=self.cfg["reduce_lr_patience"],
        )

        # O modelo recebe [features, ticker_ids] como input
        model.fit(
            [X_tr, tid_tr], y_tr,
            validation_data=([X_v, tid_v], y_v),
            class_weight=cw,
            epochs=self.cfg["epochs"],
            batch_size=self.cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        # 6. Avaliação IS (validação interna)
        y_proba_is  = model.predict([X_is, tid_is], verbose=0)
        y_pred_is   = np.argmax(y_proba_is, axis=1)

        # 7. Avaliação OOS
        y_proba_oos = model.predict([X_oos, tid_oos], verbose=0)
        y_pred_oos  = np.argmax(y_proba_oos, axis=1)

        from src.evaluation import ClassificationEvaluator, FinancialMetrics
        from src.backtest import simulate_strategy

        is_ml  = ClassificationEvaluator.evaluate(y_is,  y_pred_is,  y_proba=y_proba_is,  prefix="is")
        oos_ml = ClassificationEvaluator.evaluate(y_oos, y_pred_oos, y_proba=y_proba_oos, prefix="oos")

        strat_oos = simulate_strategy(
            y_pred_oos, returns=ret_oos,
            transaction_cost=self.cfg.get("transaction_cost", 0.001),
        )
        oos_fin = FinancialMetrics.compute(strat_oos, pd.Series(ret_oos))
        oos_fin = {f"oos_{k}": v for k, v in oos_fin.items()}

        # 8. Breakdown por ativo (OOS)
        per_ticker_metrics = {}
        for ticker, tid in ticker_to_id.items():
            mask = tid_oos == tid
            if mask.sum() < 5:
                continue
            m = ClassificationEvaluator.evaluate(y_oos[mask], y_pred_oos[mask])
            per_ticker_metrics[ticker] = m

        elapsed = time.time() - t0
        results = {
            "job_key":            self.job_key,
            "n_tickers":          n_tickers,
            "n_train":            n_is,
            "n_oos":              len(X_oos),
            "elapsed_seconds":    elapsed,
            **is_ml, **oos_ml, **oos_fin,
        }

        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / f"{self.job_key}_metrics.json", "w") as f:
            json.dump({**results, "per_ticker": per_ticker_metrics}, f, indent=2, default=str)

        logger.info("Global done: %s  acc_oos=%.3f  sharpe_oos=%.3f  (%.1fs)",
                    self.job_key,
                    results.get("oos_accuracy", float("nan")),
                    oos_fin.get("oos_sharpe", float("nan")),
                    elapsed)
        return results

    def run_all_modes_and_backbones(
        self,
        modes: list[str] | None = None,
        backbones: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Executa o modelo global para todas as combinações modo × backbone.

        Permite responder: qual frontend wavelet + backbone generaliza melhor
        em 25 ativos simultaneamente?
        """
        from config.experiment_config import MODES, DL_MODELS
        modes     = modes     or MODES
        backbones = backbones or DL_MODELS

        records = []
        for backbone in backbones:
            for mode in modes:
                logger.info("Running global: backbone=%s, mode=%s", backbone, mode)
                self.backbone = backbone
                self.mode     = mode

                try:
                    result = self.run()
                    records.append({
                        "backbone": backbone,
                        "mode":     mode,
                        **result,
                    })
                except Exception as e:
                    logger.error("Failed %s/%s: %s", backbone, mode, e)
                    records.append({"backbone": backbone, "mode": mode, "error": str(e)})

        return pd.DataFrame(records)


__all__ = [
    "GlobalFinancialPipeline",
    "build_global_model",
    "build_global_dataset",
]
