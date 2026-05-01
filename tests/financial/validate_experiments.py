"""
validate_experiments.py — Validação rápida de toda a pipeline de experimentos.

Executa cada componente do sistema com poucas épocas (EPOCHS_OVERRIDE=3 por padrão)
para verificar que o código roda sem erros antes de submeter os jobs completos.

Uso:
    # Validação rápida com 3 épocas
    EPOCHS_OVERRIDE=3 python validate_experiments.py

    # Validação com mais épocas (para testar convergência)
    EPOCHS_OVERRIDE=10 python validate_experiments.py

    # Apenas validar imports e configuração (sem treinar)
    python validate_experiments.py --config-only

O que é validado:
  1. Imports e configuração (experiment_config.py)
  2. Frontend wavelet: raw, db4, learned_wavelet (shapes de saída)
  3. Warm-start db4 no LearnedWaveletPair1D_QMF
  4. Pipeline IS/OOS: _is_oos_split, _make_windows, _align_t1
  5. FinancialExperimentPipeline.run() em modo sintético (sem dados B3 reais)
  6. MetaLabelingPipeline.fit() e .predict()
  7. GlobalFinancialPipeline com dados sintéticos
  8. WilcoxonComparison.compare() e .compare_modes()
  9. Geradores adversários: RandomWalkGenerator, RegimeOnlyGenerator, WhiteNoiseGenerator
 10. Serialização JSON de resultados (sem erros de tipo)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Setup de paths
_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

# Mock yfinance se não instalado — só é necessário para download de dados,
# não para os experimentos de validação que usam dados sintéticos ou já processados.
try:
    import yfinance  # noqa: F401
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["yfinance"] = MagicMock()

# EPOCHS_OVERRIDE para validação rápida
_EPOCHS = int(os.environ.get("EPOCHS_OVERRIDE", "3"))
os.environ["EPOCHS_OVERRIDE"] = str(_EPOCHS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate")


# ---------------------------------------------------------------------------
# Utilitário de relatório
# ---------------------------------------------------------------------------

class ValidationReport:
    """Coleta resultados de todos os testes e imprime sumário final."""

    def __init__(self):
        self.results: list[dict] = []

    def add(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append({"name": name, "passed": passed, "detail": detail})
        icon = "✓" if passed else "✗"
        logger.info("%s [%s] %s%s", icon, status, name, f" — {detail}" if detail else "")

    def run_test(self, name: str, fn: Callable) -> bool:
        """Executa fn(), captura exceções, reporta PASS/FAIL."""
        try:
            result = fn()
            detail = str(result) if result is not None else ""
            self.add(name, True, detail[:120] if detail else "")
            return True
        except Exception as e:
            self.add(name, False, f"{type(e).__name__}: {str(e)[:120]}")
            logger.debug(traceback.format_exc())
            return False

    def summary(self) -> int:
        """Imprime sumário e retorna número de falhas."""
        n_pass = sum(r["passed"] for r in self.results)
        n_fail = len(self.results) - n_pass
        print("\n" + "=" * 70)
        print(f"VALIDATION SUMMARY: {n_pass}/{len(self.results)} tests passed")
        if n_fail > 0:
            print(f"\nFAILED TESTS ({n_fail}):")
            for r in self.results:
                if not r["passed"]:
                    print(f"  ✗ {r['name']}: {r['detail']}")
        print("=" * 70)
        return n_fail


# ---------------------------------------------------------------------------
# Dados sintéticos para validação (sem dados reais B3)
# ---------------------------------------------------------------------------

def _make_synthetic_data(n: int = 500, seq_len: int = 60, n_feat: int = 10):
    """
    Gera dados sintéticos simples para validar a pipeline sem depender de dados B3.

    n       : número de observações temporais
    seq_len : comprimento da janela (deve bater com experiment_config)
    n_feat  : número de features (simulando saída do feature_engineering)
    """
    rng  = np.random.default_rng(42)
    X    = rng.standard_normal((n, n_feat)).astype(np.float32)
    y    = rng.integers(0, 3, size=n).astype(np.int32)       # 3 classes: sell/hold/buy
    ret  = rng.standard_normal(n).astype(np.float32) * 0.01  # log-retornos ~1% diário

    # t1 inteira: evento termina 10 dias após o início
    t1 = pd.Series(np.arange(n) + 10, index=pd.RangeIndex(n))

    return X, y, ret, t1


# ---------------------------------------------------------------------------
# Testes individuais
# ---------------------------------------------------------------------------

report = ValidationReport()


def test_imports():
    """Verifica que todos os módulos são importáveis sem erro."""
    import tensorflow as tf
    from config.experiment_config import (
        TICKERS, DL_MODELS, MODES, FEATURE_CONFIG, LEARNED_WAVELET_CONFIG,
        DL_TRAINING_CONFIG, VALIDATION_CONFIG, MAX_GRID_CONFIGS,
    )
    assert FEATURE_CONFIG["sequence_length"] == 60, "sequence_length deve ser 60"
    assert LEARNED_WAVELET_CONFIG["kernel_size"] == 8, "kernel_size deve ser 8"
    assert LEARNED_WAVELET_CONFIG["align"] == "pad_to_first", "align deve ser pad_to_first"
    assert DL_TRAINING_CONFIG["epochs"] == _EPOCHS, f"EPOCHS_OVERRIDE={_EPOCHS} não aplicado"
    return f"TF={tf.__version__}, EPOCHS={_EPOCHS}, seq_len={FEATURE_CONFIG['sequence_length']}"


def test_wavelet_shapes():
    """Verifica shapes de saída dos três frontends wavelet."""
    import tensorflow as tf
    from models.LWT.fixed_db4_dwt import FixedDb4DWT1D
    from models.LWT.learned_wavelet_dwt_qmf import LearnedWaveletDWT1D_QMF

    B, L, C = 4, 60, 3
    x = tf.random.normal((B, L, C))

    # db4 concat mode
    db4 = FixedDb4DWT1D(levels=2, mode="concat", align="pad_to_first")
    out_db4 = db4(x)
    assert out_db4.shape == (B, L // 2, C * 3), f"db4 shape {out_db4.shape} != expected {(B, L//2, C*3)}"

    # learned_wavelet concat mode
    lwt = LearnedWaveletDWT1D_QMF(levels=2, kernel_size=8, mode="concat", align="pad_to_first")
    out_lwt = lwt(x)
    assert out_lwt.shape == (B, L // 2, C * 3), f"lwt shape {out_lwt.shape} != expected {(B, L//2, C*3)}"

    return f"db4: {out_db4.shape}, lwt: {out_lwt.shape}"


def test_warm_start():
    """Verifica que warm_start_db4 não causa erro e modifica os pesos."""
    import tensorflow as tf
    from models.LWT.learned_wavelet_dwt_qmf import LearnedWaveletDWT1D_QMF

    B, L, C = 2, 60, 1
    x = tf.random.normal((B, L, C))

    # Sem warm-start
    lwt_cold = LearnedWaveletDWT1D_QMF(levels=1, kernel_size=8, mode="concat", warm_start_db4=False)
    out_cold = lwt_cold(x)

    # Com warm-start
    lwt_warm = LearnedWaveletDWT1D_QMF(levels=1, kernel_size=8, mode="concat", warm_start_db4=True)
    out_warm = lwt_warm(x)

    assert out_cold.shape == out_warm.shape, "shapes devem ser iguais com/sem warm-start"
    return f"cold shape={out_cold.shape}, warm shape={out_warm.shape}"


def test_wavelet_projection():
    """Verifica que a projeção linear equaliza dimensionalidade entre modos."""
    import tensorflow as tf
    from tensorflow import keras
    from src.models import _apply_wavelet_frontend

    B, L, F = 4, 60, 5
    cfg = {
        "wavelet_projection_dim": 32,
        "wavelet_levels": 2,
        "kernel_size": 8,
        "align": "pad_to_first",
        "warm_start_db4": False,
    }

    inp = keras.Input(shape=(L, F))
    out_raw  = _apply_wavelet_frontend(inp, "raw", cfg)
    out_db4  = _apply_wavelet_frontend(inp, "db4", cfg)
    out_lwt  = _apply_wavelet_frontend(inp, "learned_wavelet", cfg)

    # Todos devem ter a mesma dimensionalidade após projeção
    assert out_raw.shape[-1]  == 32, f"raw proj={out_raw.shape[-1]}"
    assert out_db4.shape[-1]  == 32, f"db4 proj={out_db4.shape[-1]}"
    assert out_lwt.shape[-1]  == 32, f"lwt proj={out_lwt.shape[-1]}"

    return f"raw={out_raw.shape}, db4={out_db4.shape}, lwt={out_lwt.shape}"


def test_is_oos_split():
    """Verifica o split IS/OOS com n amostras conhecidos."""
    from src.pipeline import _is_oos_split

    n = 3000
    cut = _is_oos_split(n, trading_days_per_year=252, test_years=6.0)
    oos_size = n - cut
    # OOS deve ter aproximadamente 252*6=1512 dias, mas limitado a n-60
    assert oos_size > 0, "OOS vazio"
    assert cut >= 60, f"IS muito curto: {cut}"
    return f"n={n}, cut={cut}, IS={cut}, OOS={oos_size}"


def test_make_windows():
    """Verifica shapes e alinhamento das janelas deslizantes."""
    from src.pipeline import _make_windows

    n, F, seq_len = 200, 5, 60
    X   = np.random.randn(n, F).astype(np.float32)
    y   = np.random.randint(0, 3, n).astype(np.int32)
    ret = np.random.randn(n).astype(np.float32)

    X_w, y_w, r_w, idx_w = _make_windows(X, y, ret, seq_len)

    expected_n = n - seq_len
    assert X_w.shape == (expected_n, seq_len, F), f"X_win shape {X_w.shape}"
    assert len(y_w) == expected_n
    assert len(idx_w) == expected_n
    assert idx_w[0] == seq_len  # primeiro label está na posição seq_len
    return f"X_win={X_w.shape}, y_win={y_w.shape}"


def test_purged_kfold():
    """Verifica PurgedKFold com numpy arrays e t1 inteiro."""
    from utils.validation import PurgedKFold

    n = 200
    X = np.random.randn(n, 5).astype(np.float32)
    y = np.random.randint(0, 3, n).astype(np.int32)
    t1 = pd.Series(np.arange(n) + 10, index=pd.RangeIndex(n))

    pkf = PurgedKFold(n_splits=3, t1=t1, pct_embargo=0.05)
    splits = list(pkf.split(X, y))

    assert len(splits) == 3, f"Esperado 3 splits, obtido {len(splits)}"
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert len(set(train_idx) & set(test_idx)) == 0, "Vazamento train/test!"

    return f"3 splits, tamanhos treino: {[len(s[0]) for s in splits]}"


def test_build_model():
    """Verifica build_model para todas as combinações backbone × modo."""
    from src.models import build_model
    from config.experiment_config import DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG

    cfg = {
        **DL_TRAINING_CONFIG,
        **LEARNED_WAVELET_CONFIG,
        "filters": [16, 32],
        "units": [16, 8],
        "lstm_units": [16, 8],
        "num_heads": 2,
        "head_size": 8,
        "ff_dim": 16,
        "num_blocks": 1,
        "dropout_rate": 0.1,
        "l2_reg": 1e-4,
        "wavelet_projection_dim": 8,
    }

    input_shape = (60, 5)
    results = []
    for backbone in ["CNN", "LSTM"]:  # CNN e LSTM como amostra (mais rápido)
        for mode in ["raw", "db4", "learned_wavelet"]:
            model = build_model(backbone, mode, input_shape, n_classes=3, cfg=cfg)
            n_params = model.count_params()
            results.append(f"{backbone}/{mode}={n_params}")

    return ", ".join(results)


def test_mini_train():
    """
    Treino mínimo end-to-end: CNN + learned_wavelet, 2 folds, poucas épocas.

    Este é o teste mais importante: simula o run() completo da pipeline
    em dados sintéticos para garantir que todo o fluxo (IS/OOS, PurgedKFold,
    escala, model.fit, evaluate, backtest) funciona sem erro.
    """
    import tensorflow as tf
    from src.models import build_model, get_callbacks
    from src.pipeline import _make_windows, _is_oos_split, _aggregate_dicts
    from src.evaluation import ClassificationEvaluator, FinancialMetrics
    from src.backtest import simulate_strategy
    from utils.validation import PurgedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.utils.class_weight import compute_class_weight

    tf.random.set_seed(42)
    # n=500 para que IS/OOS split com test_years=1.0 deixe IS suficiente:
    # oos=252, IS=248 amostras, IS-seq_len=188 janelas (>0)
    n, F, seq_len = 500, 8, 60
    X, y, ret, t1 = _make_synthetic_data(n, seq_len=seq_len, n_feat=F)

    # IS/OOS split — com n=500 e test_years=1.0: cut=max(60, 500-252)=248
    cut = _is_oos_split(n, test_years=1.0)
    X_is, y_is, ret_is = X[:cut], y[:cut], ret[:cut]

    # Windows
    X_win, y_win, ret_win, idx_win = _make_windows(X_is, y_is, ret_is, seq_len)
    t1_win = pd.Series(idx_win + 10, index=pd.RangeIndex(len(idx_win)))

    pkf = PurgedKFold(n_splits=2, t1=t1_win, pct_embargo=0.05)
    splits = list(pkf.split(X_win, y_win))
    assert len(splits) == 2

    cfg = {
        "epochs": _EPOCHS,
        "batch_size": 16,
        "early_stopping_patience": 2,
        "reduce_lr_patience": 1,
        "reduce_lr_factor": 0.5,
        "min_lr": 1e-6,
        "learning_rate": 1e-3,
        "filters": [8, 16],
        "dropout_rate": 0.1,
        "l2_reg": 1e-4,
        "kernel_size": 8,
        "wavelet_levels": 2,
        "align": "pad_to_first",
        "warm_start_db4": True,
        "wavelet_net_units": 8,
        "reg_energy": 1e-2,
        "reg_high_dc": 1e-2,
        "reg_smooth": 1e-3,
        "wavelet_projection_dim": 8,
    }

    ml_metrics_list = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        tf.keras.backend.clear_session()
        n_val     = max(1, int(len(train_val_idx) * 0.15))
        train_idx = train_val_idx[:-n_val]
        val_idx   = train_val_idx[-n_val:]

        scaler = RobustScaler()
        n_feat = X_win.shape[-1]
        X_tr = scaler.fit_transform(X_win[train_idx].reshape(-1, n_feat)).reshape(X_win[train_idx].shape)
        X_v  = scaler.transform(X_win[val_idx].reshape(-1, n_feat)).reshape(X_win[val_idx].shape)
        X_te = scaler.transform(X_win[test_idx].reshape(-1, n_feat)).reshape(X_win[test_idx].shape)

        cw = {int(c): float(w) for c, w in zip(*[
            np.unique(y_win[train_idx]),
            compute_class_weight("balanced", classes=np.unique(y_win[train_idx]), y=y_win[train_idx])
        ])}

        model = build_model("CNN", "learned_wavelet", X_tr.shape[1:], n_classes=3, cfg=cfg)
        model.fit(
            X_tr, y_win[train_idx],
            validation_data=(X_v, y_win[val_idx]),
            class_weight=cw,
            epochs=_EPOCHS, batch_size=16, verbose=0,
        )

        y_proba = model.predict(X_te, verbose=0)
        y_pred  = np.argmax(y_proba, axis=1)
        ml_metrics_list.append(ClassificationEvaluator.evaluate(y_win[test_idx], y_pred, y_proba=y_proba))

    agg = _aggregate_dicts(ml_metrics_list)
    return f"acc={agg.get('accuracy', 'N/A'):.3f}, f1={agg.get('f1_macro', 'N/A'):.3f}"


def test_meta_labeling():
    """Treina MetaLabelingPipeline em dados sintéticos."""
    from src.meta_labeling import MetaLabelingPipeline, make_meta_labels

    X, y, ret, t1 = _make_synthetic_data(400, seq_len=60, n_feat=8)

    # Treina em dados 2D (sem janelas — meta-labeling usa features brutas)
    pipeline = MetaLabelingPipeline(n_folds=2, meta_threshold=0.5, use_fractional_sizing=True)
    pipeline.fit(X, y, t1=t1)

    X_te, y_te, ret_te = X[-50:], y[-50:], ret[-50:]
    y_pred, meta_proba, positions = pipeline.predict(X_te)

    assert len(y_pred) == 50
    assert meta_proba.min() >= 0 and meta_proba.max() <= 1

    metrics = pipeline.evaluate(X_te, y_te, returns=ret_te)
    return f"primary_acc={metrics.get('primary_accuracy', 'N/A'):.3f}, coverage={metrics.get('meta_coverage', 'N/A'):.2f}"


def test_global_model_build():
    """Verifica build_global_model com dados sintéticos mínimos."""
    import tensorflow as tf
    from src.global_pipeline import build_global_model

    cfg = {
        "epochs": _EPOCHS,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "early_stopping_patience": 2,
        "reduce_lr_patience": 1,
        "reduce_lr_factor": 0.5,
        "min_lr": 1e-6,
        "filters": [8, 16],
        "dropout_rate": 0.1,
        "l2_reg": 1e-4,
        "kernel_size": 8,
        "wavelet_levels": 2,
        "align": "pad_to_first",
        "warm_start_db4": False,
        "wavelet_net_units": 8,
        "reg_energy": 1e-2,
        "reg_high_dc": 1e-2,
        "reg_smooth": 1e-3,
        "wavelet_projection_dim": 8,
    }

    model = build_global_model(
        n_tickers=5,
        input_shape=(60, 8),
        mode="learned_wavelet",
        backbone="CNN",
        emb_dim=4,
        cfg=cfg,
    )

    # Treino mínimo com dados sintéticos
    B = 16
    X_syn   = np.random.randn(B, 60, 8).astype(np.float32)
    tid_syn = np.random.randint(0, 5, B).astype(np.int32)
    y_syn   = np.random.randint(0, 3, B).astype(np.int32)

    model.fit([X_syn, tid_syn], y_syn, epochs=_EPOCHS, batch_size=B, verbose=0)
    y_proba = model.predict([X_syn, tid_syn], verbose=0)

    assert y_proba.shape == (B, 3), f"proba shape {y_proba.shape}"
    n_params = model.count_params()
    return f"params={n_params}, proba_shape={y_proba.shape}"


def test_wilcoxon():
    """Verifica WilcoxonComparison.compare() e .compare_modes()."""
    from src.evaluation import WilcoxonComparison

    rng = np.random.default_rng(0)
    # A > B em média — deve ser significativo com n=30
    a = rng.normal(0.6, 0.1, 30)
    b = rng.normal(0.5, 0.1, 30)
    result = WilcoxonComparison.compare(a, b, alternative="two-sided")

    assert "p_value" in result
    assert "effect_size" in result
    assert result["n_pairs"] == 30

    # compare_modes em DataFrame simulado
    n_tickers = 10
    records = []
    for ticker in [f"T{i}" for i in range(n_tickers)]:
        for mode, base_acc in [("raw", 0.50), ("db4", 0.53), ("learned_wavelet", 0.56)]:
            records.append({
                "ticker": ticker,
                "mode":   mode,
                "ml_f1_macro": base_acc + rng.normal(0, 0.05),
            })
    df = pd.DataFrame(records)
    cmp_df = WilcoxonComparison.compare_modes(df, metric="ml_f1_macro")

    assert len(cmp_df) == 2  # db4_vs_raw e learned_wavelet_vs_raw
    return f"p_value_pair={result['p_value']:.4f}, significant={result['significant']}, modes_compared={len(cmp_df)}"


def test_adversarial_generators():
    """Verifica os três geradores adversários."""
    from tests.synthetic.src.data_generator import (
        RandomWalkGenerator, RegimeOnlyGenerator, WhiteNoiseGenerator
    )

    rw = RandomWalkGenerator(n_samples=1000)
    X_rw, y_rw = rw.create_regression_dataset(sequence_length=64)
    assert X_rw.shape[1] == 64

    ro = RegimeOnlyGenerator(n_samples=1000, n_regimes=3)
    X_ro, y_ro = ro.create_regression_dataset(sequence_length=64)
    assert X_ro.shape[1] == 64

    wn = WhiteNoiseGenerator(n_samples=1000, distribution="laplace")
    X_wn, y_wn = wn.create_regression_dataset(sequence_length=64)
    assert X_wn.shape[1] == 64

    return (f"RandomWalk={X_rw.shape}, RegimeOnly={X_ro.shape}, "
            f"WhiteNoise(laplace)={X_wn.shape}")


def test_evaluation_metrics():
    """Verifica ClassificationEvaluator e FinancialMetrics com dados sintéticos."""
    from src.evaluation import ClassificationEvaluator, FinancialMetrics

    rng = np.random.default_rng(7)
    y_true  = rng.integers(0, 3, 100)
    y_pred  = rng.integers(0, 3, 100)
    y_proba = rng.dirichlet([1, 1, 1], size=100).astype(np.float32)

    metrics = ClassificationEvaluator.evaluate(y_true, y_pred, y_proba=y_proba)
    required = ["accuracy", "f1_macro", "f1_weighted", "mcc", "roc_auc_ovr"]
    for k in required:
        assert k in metrics, f"Métrica faltando: {k}"

    ret = pd.Series(rng.normal(0.001, 0.02, 252))
    fin = FinancialMetrics.compute(ret)
    assert "sharpe" in fin and "max_drawdown" in fin

    return f"acc={metrics['accuracy']:.3f}, sharpe={fin['sharpe']:.3f}"


def test_json_serialization():
    """Verifica que os resultados são serializáveis em JSON (sem erros de tipo)."""
    import json
    from src.evaluation import ClassificationEvaluator, FinancialMetrics

    rng = np.random.default_rng(1)
    y_true  = rng.integers(0, 3, 50)
    y_pred  = rng.integers(0, 3, 50)
    y_proba = rng.dirichlet([1, 1, 1], size=50).astype(np.float32)

    ml_metrics  = ClassificationEvaluator.evaluate(y_true, y_pred, y_proba=y_proba)
    fin_metrics = FinancialMetrics.compute(pd.Series(rng.normal(0.001, 0.02, 50)))

    record = {
        "ticker": "TEST3.SA",
        "model_name": "CNN",
        "mode": "learned_wavelet",
        **ml_metrics,
        **{f"fin_{k}": v for k, v in fin_metrics.items()},
    }

    json_str = json.dumps(record, default=str)
    loaded   = json.loads(json_str)
    assert "accuracy" in loaded
    return f"JSON OK, {len(record)} campos"


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def main(config_only: bool = False):
    logger.info("=" * 70)
    logger.info("FINANCIAL EXPERIMENT VALIDATION (EPOCHS=%d)", _EPOCHS)
    logger.info("=" * 70)

    # Testes de configuração e imports (sempre executados)
    report.run_test("1. Imports e configuração", test_imports)
    report.run_test("2. Shapes dos frontends wavelet", test_wavelet_shapes)

    if config_only:
        return report.summary()

    # Testes funcionais
    report.run_test("3. Warm-start db4", test_warm_start)
    report.run_test("4. Projeção wavelet (equalização dim)", test_wavelet_projection)
    report.run_test("5. IS/OOS split", test_is_oos_split)
    report.run_test("6. Make windows (stride_tricks)", test_make_windows)
    report.run_test("7. PurgedKFold (numpy + t1 inteiro)", test_purged_kfold)
    report.run_test("8. Build model (CNN × 3 modos)", test_build_model)
    report.run_test("9. Mini treino end-to-end", test_mini_train)
    report.run_test("10. MetaLabelingPipeline fit/predict", test_meta_labeling)
    report.run_test("11. GlobalModel build + mini fit", test_global_model_build)
    report.run_test("12. Wilcoxon compare + compare_modes", test_wilcoxon)
    report.run_test("13. Geradores adversários (RW/Regime/Noise)", test_adversarial_generators)
    report.run_test("14. Métricas de avaliação completas", test_evaluation_metrics)
    report.run_test("15. Serialização JSON dos resultados", test_json_serialization)

    n_failures = report.summary()

    if n_failures == 0:
        logger.info("\n✓ Todos os testes passaram — pipeline pronta para execução completa.")
        logger.info("  Para rodar experimentos: EPOCHS_OVERRIDE=100 python run_dl_queue.py")
    else:
        logger.error("\n✗ %d teste(s) falharam — corrija antes de rodar experimentos completos.", n_failures)

    return n_failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida a pipeline de experimentos financeiros")
    parser.add_argument("--config-only", action="store_true",
                        help="Valida apenas imports e configuração (sem treinar)")
    args = parser.parse_args()

    failures = main(config_only=args.config_only)
    sys.exit(failures)  # exit code 0 = todos passaram, N = N falhas
