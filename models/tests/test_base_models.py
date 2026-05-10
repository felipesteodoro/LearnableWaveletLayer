"""
Smoke tests for models/base_models.py and the three experiment wrappers.

Verifies that every (model_name, mode, task) combination builds and produces
the correct output shape without actually training.

Run from the project root:
    python models/tests/test_base_models.py
"""
import os
import sys

# Ensure models/ is on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODELS_DIR = os.path.join(_ROOT, "models")
for _p in (_MODELS_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # suppress TF noise

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from base_models import build_model, get_callbacks, get_distribute_strategy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH = 4
SEQ_LEN = 64      # short enough for fast tests
N_FEAT = 1
INPUT_SHAPE = (SEQ_LEN, N_FEAT)

BACKBONES = ["CNN", "LSTM", "CNN_LSTM", "Transformer", "MLP"]
MODES_WAVELET = ["raw", "learned_wavelet", "learned_wavelet_no_warmup"]
TASKS = ["regression", "binary", "multiclass"]

# Minimal config that works for all backbone types
BASE_CFG = {
    # CNN
    "filters": [8, 16],
    "kernel_sizes": [3, 3],
    "pool_sizes": [2, 2],
    "dense_units": [16],
    # LSTM
    "units": [8, 4],
    "recurrent_dropout": 0.0,
    # CNN_LSTM
    "cnn_filters": [8, 16],
    "cnn_kernel_sizes": [3, 3],
    "lstm_units": [8, 4],
    # Transformer
    "head_size": 4,
    "num_heads": 2,
    "ff_dim": 16,
    "num_transformer_blocks": 1,
    "mlp_units": [16],
    # MLP
    # mlp_units already set above
    # Wavelet
    "levels": 2,
    "kernel_size": 4,
    "wavelet_net_units": 8,
    "reg_energy": 1e-2,
    "reg_high_dc": 1e-2,
    "reg_smooth": 1e-3,
    "warm_start_db4": False,
    "align": "pad_to_first",
    # Training
    "dropout_rate": 0.0,
    "l2_reg": 0.0,
    "learning_rate": 1e-3,
    "use_warmup": False,
}


def _dummy_input():
    return np.random.randn(BATCH, SEQ_LEN, N_FEAT).astype(np.float32)


def test_build_model_shapes():
    """Every (backbone, mode, task) must build and produce the right output shape."""
    errors = []
    x = _dummy_input()

    for backbone in BACKBONES:
        for mode in MODES_WAVELET:
            for task in TASKS:
                n_classes = 3 if task == "multiclass" else None
                label = f"{backbone}/{mode}/{task}"
                try:
                    model = build_model(
                        backbone, mode, INPUT_SHAPE,
                        task=task, n_classes=n_classes, cfg=dict(BASE_CFG),
                    )
                    out = model(x, training=False)
                    if task == "regression":
                        assert out.shape == (BATCH, 1), f"{label}: shape {out.shape}"
                    elif task == "binary":
                        assert out.shape == (BATCH, 1), f"{label}: shape {out.shape}"
                    elif task == "multiclass":
                        assert out.shape == (BATCH, 3), f"{label}: shape {out.shape}"
                    print(f"  OK  {label}  → {out.shape}")
                except Exception as e:
                    errors.append(f"FAIL {label}: {e}")
                    print(f"  FAIL {label}: {e}")

    return errors


def test_get_callbacks(tmp_path="/tmp/test_callbacks"):
    """get_callbacks must return a non-empty list and create the parent dir."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "sub" / "model.keras"
        cbs = get_callbacks(path, early_patience=5, lr_patience=3)
        assert len(cbs) == 3, f"Expected 3 callbacks, got {len(cbs)}"
        cbs_no_lr = get_callbacks(path, use_reduce_lr=False)
        assert len(cbs_no_lr) == 2, f"Expected 2 callbacks, got {len(cbs_no_lr)}"
        assert path.parent.exists(), "Parent dir not created"
    print("  OK  get_callbacks")


def test_experiment_wrappers():
    """Each experiment's models.py must import cleanly and build a model."""
    errors = []
    x = _dummy_input()

    # --- synthetic (regression) ---
    try:
        sys.path.insert(0, os.path.join(_ROOT, "tests", "synthetic", "src"))
        import importlib
        syn = importlib.import_module("models")
        m = syn.create_cnn_model(INPUT_SHAPE, params=dict(BASE_CFG))
        out = m(x, training=False)
        assert out.shape == (BATCH, 1)
        print("  OK  synthetic.create_cnn_model")
    except Exception as e:
        errors.append(f"FAIL synthetic.create_cnn_model: {e}")
        print(f"  FAIL synthetic.create_cnn_model: {e}")

    # --- ford-a (binary) ---
    try:
        # Remove cached synthetic models module before importing ford-a
        if "models" in sys.modules:
            del sys.modules["models"]
        sys.path.insert(0, os.path.join(_ROOT, "tests", "ford-a", "src"))
        forda = importlib.import_module("models")
        m = forda.create_cnn_model(INPUT_SHAPE, params=dict(BASE_CFG))
        out = m(x, training=False)
        assert out.shape == (BATCH, 1)
        print("  OK  ford-a.create_cnn_model")
    except Exception as e:
        errors.append(f"FAIL ford-a.create_cnn_model: {e}")
        print(f"  FAIL ford-a.create_cnn_model: {e}")

    # --- financial (multiclass) ---
    try:
        if "models" in sys.modules:
            del sys.modules["models"]
        sys.path.insert(0, os.path.join(_ROOT, "tests", "financial", "src"))
        fin = importlib.import_module("models")
        cfg = dict(BASE_CFG)
        cfg["n_classes"] = 3
        m = fin.build_model("CNN", "raw", INPUT_SHAPE, n_classes=3, cfg=cfg)
        out = m(x, training=False)
        assert out.shape == (BATCH, 3)
        print("  OK  financial.build_model CNN/raw/multiclass")
    except Exception as e:
        errors.append(f"FAIL financial.build_model: {e}")
        print(f"  FAIL financial.build_model: {e}")

    return errors


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== test_build_model_shapes ===")
    e1 = test_build_model_shapes()

    print("\n=== test_get_callbacks ===")
    test_get_callbacks()

    print("\n=== test_experiment_wrappers ===")
    e2 = test_experiment_wrappers()

    all_errors = e1 + e2
    if all_errors:
        print(f"\n{len(all_errors)} FAILURE(S):")
        for e in all_errors:
            print(" ", e)
        sys.exit(1)
    else:
        print(f"\nAll tests passed.")
