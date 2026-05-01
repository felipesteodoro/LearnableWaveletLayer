"""
Validação rápida dos experimentos Ford A (classificação binária).

Executa 12 testes com EPOCHS_OVERRIDE=3 para verificar que todo o código
funciona corretamente antes de rodar os experimentos completos.

Uso:
    EPOCHS_OVERRIDE=3 python validate_experiments.py
    EPOCHS_OVERRIDE=3 python validate_experiments.py --config-only
"""
import os
import sys
import argparse
import traceback
import time

# Configura epochs override para smoke-test rápido
os.environ.setdefault("EPOCHS_OVERRIDE", "3")

# Adiciona raiz do projeto ao path para que imports de 'models/' e 'tests/' funcionem
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_MODELS_DIR = os.path.join(_ROOT, 'models')
_SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
_CFG_DIR = os.path.join(os.path.dirname(__file__), 'config')

for _p in [_ROOT, _MODELS_DIR, _SRC_DIR, _CFG_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================================
# RELATÓRIO DE VALIDAÇÃO
# ============================================================================

class ValidationReport:
    def __init__(self):
        self.results = []

    def run_test(self, name, fn):
        print(f"  ► {name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            print(f"PASS ({elapsed:.1f}s)")
            self.results.append((name, True, None))
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"FAIL ({elapsed:.1f}s)")
            tb = traceback.format_exc()
            self.results.append((name, False, tb))

    def summary(self):
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"RESULTADO: {passed}/{total} testes passaram")
        print(f"{'='*60}")
        if passed < total:
            for name, ok, tb in self.results:
                if not ok:
                    print(f"\n[FAIL] {name}")
                    print(tb)
        return passed == total


report = ValidationReport()


# ============================================================================
# TESTE 1 — Importação do config
# ============================================================================
def test_config_imports():
    import experiment_config as cfg
    assert hasattr(cfg, "LEARNED_WAVELET_CONFIG")
    assert hasattr(cfg, "DL_MODELS_CONFIG")
    assert hasattr(cfg, "FORDA_CONFIG")
    assert cfg.FORDA_CONFIG["sequence_length"] == 500
    assert cfg.FORDA_CONFIG["n_features"] == 1
    # warmup deve estar desativado para este experimento
    assert cfg.DL_MODELS_CONFIG["Transformer"]["use_warmup"] is False
    assert cfg.LEARNED_WAVELET_MODELS_CONFIG["Transformer"]["use_warmup"] is False


report.run_test("config_imports", test_config_imports)


# ============================================================================
# TESTE 2 — Importação dos módulos de wavelet
# ============================================================================
def test_wavelet_imports():
    from LWT import LearnedWaveletDWT1D_QMF
    from LWT import LearnedWaveletPair1D_QMF
    assert LearnedWaveletDWT1D_QMF is not None
    assert LearnedWaveletPair1D_QMF is not None


report.run_test("wavelet_imports", test_wavelet_imports)


# ============================================================================
# TESTE 3 — Shape de saída do wavelet (seq_len=500, kernel_size=32, levels=2)
# ============================================================================
def test_wavelet_output_shape():
    import math
    import numpy as np
    import tensorflow as tf
    from LWT import LearnedWaveletDWT1D_QMF

    seq_len = 500
    batch = 4
    x = tf.random.normal([batch, seq_len, 1])

    dwt = LearnedWaveletDWT1D_QMF(levels=2, kernel_size=32, mode="concat",
                                   align="pad_to_first")
    out = dwt(x)
    # pad_to_first alinha todas as sub-bandas ao comprimento de D1 = ceil(seq_len/2)
    expected_len = math.ceil(seq_len / 2)
    # com levels=2 e mode="concat": 3 sub-bandas (A2, D2, D1) × 1 canal = 3 canais
    assert out.shape[0] == batch,         f"batch errado: {out.shape[0]}"
    assert out.shape[1] == expected_len,  f"len errado: {out.shape[1]} != {expected_len}"
    assert out.shape[2] == 3,             f"canais errados: {out.shape[2]}"


report.run_test("wavelet_output_shape", test_wavelet_output_shape)


# ============================================================================
# TESTE 4 — Warm-start db4 funciona sem erro
# ============================================================================
def test_warm_start_db4():
    import numpy as np
    import tensorflow as tf
    from LWT import LearnedWaveletDWT1D_QMF

    seq_len = 500
    batch = 2
    x = tf.random.normal([batch, seq_len, 1])

    # warm_start_db4=True deve inicializar filtro próximo ao db4
    dwt = LearnedWaveletDWT1D_QMF(levels=2, kernel_size=32, mode="concat",
                                   align="pad_to_first", warm_start_db4=True)
    out = dwt(x)
    assert not tf.reduce_any(tf.math.is_nan(out)).numpy(), "NaN na saída com warm_start"


report.run_test("warm_start_db4", test_warm_start_db4)


# ============================================================================
# TESTE 5 — LEARNED_WAVELET_CONFIG propagado corretamente para os modelos
# ============================================================================
def test_learned_wavelet_config_values():
    import experiment_config as cfg
    lwc = cfg.LEARNED_WAVELET_CONFIG
    assert lwc["align"] == "pad_to_first", f"align errado: {lwc['align']}"
    assert lwc["warm_start_db4"] is True, "warm_start_db4 deve ser True"
    assert lwc["normalize_low"] == "l2", f"normalize_low errado: {lwc['normalize_low']}"
    assert lwc["kernel_size"] == 32


report.run_test("learned_wavelet_config_values", test_learned_wavelet_config_values)


# ============================================================================
# TESTE 6 — Construção dos modelos DL (raw)
# ============================================================================
def test_build_raw_models():
    from models import (
        create_cnn_model, create_lstm_model,
        create_cnn_lstm_model, create_transformer_model,
    )
    input_shape = (500, 1)
    cnn   = create_cnn_model(input_shape)
    lstm  = create_lstm_model(input_shape)
    cnlst = create_cnn_lstm_model(input_shape)
    trans = create_transformer_model(input_shape, params={"use_warmup": False})
    # todos devem compilar com saída (None, 1) e sigmoid
    for m in [cnn, lstm, cnlst, trans]:
        assert m.output_shape == (None, 1), f"shape errada: {m.output_shape}"


report.run_test("build_raw_models", test_build_raw_models)


# ============================================================================
# TESTE 7 — Construção dos modelos com wavelet aprendido
# ============================================================================
def test_build_learned_wavelet_models():
    import experiment_config as cfg
    from models import (
        create_learned_wavelet_cnn_model,
        create_learned_wavelet_lstm_model,
        create_learned_wavelet_cnn_lstm_model,
        create_learned_wavelet_transformer_model,
    )
    input_shape = (500, 1)
    wc = cfg.LEARNED_WAVELET_CONFIG

    m1 = create_learned_wavelet_cnn_model(input_shape, wavelet_config=wc)
    m2 = create_learned_wavelet_lstm_model(input_shape, wavelet_config=wc)
    m3 = create_learned_wavelet_cnn_lstm_model(input_shape, wavelet_config=wc)
    m4 = create_learned_wavelet_transformer_model(
        input_shape, wavelet_config=wc,
        transformer_params={"use_warmup": False}
    )
    for m in [m1, m2, m3, m4]:
        assert m.output_shape == (None, 1), f"shape errada: {m.output_shape}"


report.run_test("build_learned_wavelet_models", test_build_learned_wavelet_models)


# ============================================================================
# TESTE 8 — align e warm_start_db4 são lidos do config pelo modelo
# ============================================================================
def test_wavelet_config_passed_to_layer():
    import experiment_config as cfg
    from models import create_learned_wavelet_cnn_model
    from LWT.learned_wavelet_dwt_qmf import LearnedWaveletDWT1D_QMF

    input_shape = (500, 1)
    wc = dict(cfg.LEARNED_WAVELET_CONFIG)  # cópia

    model = create_learned_wavelet_cnn_model(input_shape, wavelet_config=wc)

    # Procura a camada LearnedWaveletDWT1D_QMF no modelo
    wl_layers = [l for l in model.layers if isinstance(l, LearnedWaveletDWT1D_QMF)]
    assert len(wl_layers) == 1, "Esperado exatamente 1 camada LearnedWaveletDWT1D_QMF"
    wl = wl_layers[0]
    assert wl.align == "pad_to_first", f"align errado: {wl.align}"
    assert wl.warm_start_db4 is True, "warm_start_db4 deve ser True"


report.run_test("wavelet_config_passed_to_layer", test_wavelet_config_passed_to_layer)


# ============================================================================
# TESTE 9 — Transformer sem warmup usa lr fixo (não schedule)
# ============================================================================
def test_transformer_no_warmup():
    from models import create_transformer_model
    import tensorflow as tf

    model = create_transformer_model(
        (500, 1),
        params={"use_warmup": False, "learning_rate": 0.001}
    )
    # Com use_warmup=False, o optimizer deve usar lr float, não schedule
    lr = model.optimizer.learning_rate
    # Deve ser float ou tf.Variable (não um Schedule)
    assert not isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule), \
        "Warmup schedule ainda ativo com use_warmup=False"


report.run_test("transformer_no_warmup", test_transformer_no_warmup)


# ============================================================================
# TESTE 10 — Mini-treino end-to-end (CNN + learned_wavelet, dados sintéticos)
# ============================================================================
def test_mini_train_learned_wavelet_cnn():
    import numpy as np
    import tensorflow as tf
    import experiment_config as cfg
    from models import create_learned_wavelet_cnn_model

    # Dados sintéticos: séries temporais de comprimento 500 (univariado)
    np.random.seed(42)
    n = 100
    seq_len = 500
    X = np.random.randn(n, seq_len, 1).astype(np.float32)
    y = (np.random.rand(n) > 0.5).astype(np.float32)

    wc = cfg.LEARNED_WAVELET_CONFIG
    model = create_learned_wavelet_cnn_model(
        (seq_len, 1),
        wavelet_config=wc,
        cnn_params={"filters": [32], "learning_rate": 0.001}
    )

    history = model.fit(X, y, epochs=3, batch_size=16, verbose=0,
                        validation_split=0.2)
    assert "loss" in history.history
    assert len(history.history["loss"]) == 3
    # loss deve ser finita
    assert all(not (l != l) for l in history.history["loss"]), "NaN na loss"


report.run_test("mini_train_learned_wavelet_cnn", test_mini_train_learned_wavelet_cnn)


# ============================================================================
# TESTE 11 — Mini-treino LSTM com wavelet (verifica backward pass)
# ============================================================================
def test_mini_train_learned_wavelet_lstm():
    import numpy as np
    import experiment_config as cfg
    from models import create_learned_wavelet_lstm_model

    np.random.seed(0)
    n = 80
    seq_len = 500
    X = np.random.randn(n, seq_len, 1).astype(np.float32)
    y = (np.random.rand(n) > 0.5).astype(np.float32)

    wc = cfg.LEARNED_WAVELET_CONFIG
    model = create_learned_wavelet_lstm_model(
        (seq_len, 1),
        wavelet_config=wc,
        lstm_params={"units": [32, 16], "learning_rate": 0.001}
    )
    history = model.fit(X, y, epochs=3, batch_size=16, verbose=0,
                        validation_split=0.2)
    loss_vals = history.history["loss"]
    assert all(not (l != l) for l in loss_vals), "NaN na loss"


report.run_test("mini_train_learned_wavelet_lstm", test_mini_train_learned_wavelet_lstm)


# ============================================================================
# TESTE 12 — Métricas de avaliação (accuracy, f1, auc)
# ============================================================================
def test_evaluation_metrics():
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.4, 0.9, 0.1])
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)

    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= auc <= 1.0


report.run_test("evaluation_metrics", test_evaluation_metrics)


# ============================================================================
# SUMÁRIO FINAL
# ============================================================================
ok = report.summary()
sys.exit(0 if ok else 1)
