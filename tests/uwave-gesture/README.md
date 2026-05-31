# UWaveGestureLibrary — Multivariate Multiclass Classification

**Task**: classify 3-channel accelerometer gestures into one of **8 classes**.
**Goal**: validate the LWT layer (`LearnedWaveletDWT1D_QMF`) on a low-channel **multivariate**
UEA benchmark — the scenario where the per-channel learned filters should shine versus a
fixed db4 DWT (which applies the same filter to every channel).

---

## Dataset

**UWaveGestureLibrary** — UEA Time Series Classification Archive (multivariate variant).
Accelerometer (x/y/z) gesture recordings; each sample is a 315-point, 3-channel series.
Loaded via `aeon.datasets.load_classification("UWaveGestureLibrary")`, with a direct-ZIP fallback.

| Split | Samples |
|---|---|
| Train | 102 |
| Validation | 18 |
| Test | 320 |

Classes: **8 gestures** (UEA labels `1..8` remapped to integers `0..7`).

> Note: this is the **multivariate** UWaveGestureLibrary (120 train / 320 test). The much
> larger `UWaveGestureLibraryAll/X/Y/Z` (~896/3582) are the *univariate* UCR variants — a
> different dataset. The small training set means DL configs should lean on regularization.

Downloaded and preprocessed by `00_download_and_eda.ipynb`.

---

## Structure

```
tests/uwave-gesture/
├── config/
│   └── experiment_config.py       # Centralised hyperparameters (multiclass)
├── src/
│   ├── data_loader.py             # UWaveDataLoader — keeps 3-D (n, 315, 3)
│   ├── feature_extraction.py      # WaveletFeatureExtractor (multivariate)
│   ├── models.py                  # ML + DL factories (task="multiclass", n_classes=8)
│   ├── evaluation.py              # ClassificationEvaluator (macro + AUC-OvR)
│   ├── pipeline_queue.py          # UWaveExperimentPipeline
│   └── visualization.py           # Plots (multivariate samples, 8×8 confusion, ROC-OvR)
├── gpu_queue/                     # Multi-GPU job queue (job/worker/runner)
├── run_dl_queue.py                # Entry point for the DL experiments
├── data/                          # .npy splits + metadata.json + wavelet features
├── results/                       # results/<run_id>/<MODEL>_<mode>/cfgNNN/metrics.json
├── saved_models/  logs/
└── README.md
```

Dashboard: `dashboards/uwave_gesture_dashboard.py`.

---

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 00 | `00_download_and_eda.ipynb` | Download, EDA (3 channels/class, spectra), wavelet features |
| 01 | `01_ml_experiments.ipynb` | ML on multivariate wavelet features (RF/XGB/LGBM, `f1_macro`) |
| 02 | `02_dl_raw_experiments.ipynb` | DL on raw 3-channel signal |
| 03 | `03_dl_wavelet_experiments.ipynb` | DL on fixed db4 DWT |
| 04 | `04_learned_wavelet_experiments.ipynb` | DL with `LearnedWaveletDWT1D_QMF` |
| 05 | `05_comparison_analysis.ipynb` | Aggregate results, ranking, raw vs fixed vs learned |
| 06 | `06_learned_filter_analysis.ipynb` | Inspect learned filters (per channel/level) |

---

## Running the DL experiments

```bash
cd tests/uwave-gesture

# full grid, all models × modes, multi-GPU
python run_dl_queue.py --fresh

# subset / smoke test
EPOCHS_OVERRIDE=1 MAX_GRID_CONFIGS=1 N_GPUS=1 DL_MODELS=CNN MODES=raw,learned_wavelet \
  python run_dl_queue.py --fresh

# monitor
streamlit run ../../dashboards/uwave_gesture_dashboard.py
```

Models: CNN · LSTM · CNN-LSTM · Transformer · MLP
Modes: `raw` · `db4` · `learned_wavelet_no_warmup` · `learned_wavelet`

---

## Metrics (multiclass)

| Metric | Description |
|---|---|
| accuracy | Overall accuracy |
| f1_macro / f1_weighted | F1 averaged over classes (macro / weighted) |
| precision_macro / recall_macro | Macro-averaged precision and recall |
| auc_ovr | ROC-AUC one-vs-rest, macro-averaged |

---

## Key Hyperparameters

```python
UWAVE_CONFIG = {"n_classes": 8, "sequence_length": 315, "n_features": 3}
WAVELET_CONFIG = {"wavelet_type": "db2", "decomposition_level": 3}
LEARNED_WAVELET_CONFIG = {
    "levels": 3,
    "kernel_size": 8,          # adequado a seq_len=315 (157→78→39)
    "reg_energy": 1e-2, "reg_high_dc": 1e-2, "reg_smooth": 1e-3,
    "warm_start_db4": True,
}
DL_TRAINING_CONFIG = {"epochs": 100, "batch_size": 256, "early_stopping_patience": 15}
```
