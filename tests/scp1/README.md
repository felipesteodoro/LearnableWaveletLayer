# SelfRegulationSCP1 — Multivariate Binary Classification (EEG)

**Task**: classify 6-channel EEG (slow cortical potentials) as **negativity** vs **positivity**.
**Goal**: validate the LWT layer (`LearnedWaveletDWT1D_QMF`) on **EEG** — the canonical domain
for wavelets (non-stationary, multi-scale signals) — and on a **binary multivariate** task,
complementing the UWaveGestureLibrary (multiclass) and Ford-A (univariate) experiments.

---

## Dataset

**SelfRegulationSCP1** — UEA Time Series Classification Archive (multivariate).
EEG recordings; each sample is an 896-point, 6-channel series.
Loaded via `aeon.datasets.load_classification("SelfRegulationSCP1")`, with a direct-ZIP fallback.

| Split | Samples |
|---|---|
| Train | 227 |
| Validation | 41 |
| Test | 293 |

Classes: **negativity** (0) / **positivity** (1) — balanced. Labels mapped from strings to 0/1.

Downloaded and preprocessed by `00_download_and_eda.ipynb`.

---

## Structure

```
tests/scp1/
├── config/experiment_config.py    # Centralised hyperparameters (binary)
├── src/
│   ├── data_loader.py             # SCP1DataLoader — keeps 3-D (n, 896, 6)
│   ├── feature_extraction.py      # WaveletFeatureExtractor (multivariate)
│   ├── models.py                  # ML + DL factories (task="binary")
│   ├── evaluation.py              # ClassificationEvaluator (binary: f1, auc_roc, specificity)
│   ├── pipeline_queue.py          # SCP1ExperimentPipeline
│   └── visualization.py           # Plots (6-channel samples, 2×2 confusion, ROC)
├── gpu_queue/                     # Multi-GPU job queue
├── run_dl_queue.py                # Entry point for the DL experiments
├── data/  results/  saved_models/  logs/
└── README.md
```

Dashboard: `dashboards/scp1_dashboard.py`.

---

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 00 | `00_download_and_eda.ipynb` | Download, EDA (6 EEG channels, spectra), wavelet features |
| 01 | `01_ml_experiments.ipynb` | ML on multivariate wavelet features (RF/XGB/LGBM, `accuracy`) |
| 02 | `02_dl_raw_experiments.ipynb` | DL on raw 6-channel EEG |
| 03 | `03_dl_wavelet_experiments.ipynb` | DL on fixed db4 DWT |
| 04 | `04_learned_wavelet_experiments.ipynb` | DL with `LearnedWaveletDWT1D_QMF` |
| 05 | `05_comparison_analysis.ipynb` | Aggregate results, raw vs fixed vs learned |
| 06 | `06_learned_filter_analysis.ipynb` | Inspect learned filters (per channel/level) |

---

## Running the DL experiments

```bash
cd tests/scp1

# full grid, all models × modes, multi-GPU
python run_dl_queue.py --fresh

# subset / smoke test
EPOCHS_OVERRIDE=1 MAX_GRID_CONFIGS=1 N_GPUS=1 DL_MODELS=CNN MODES=raw,learned_wavelet \
  python run_dl_queue.py --fresh

# monitor
streamlit run ../../dashboards/scp1_dashboard.py
```

Models: CNN · LSTM · CNN-LSTM · Transformer · MLP
Modes: `raw` · `db4` · `learned_wavelet_no_warmup` · `learned_wavelet`

---

## Metrics (binary)

| Metric | Description |
|---|---|
| accuracy | Overall accuracy |
| f1 | Binary F1 (positive class) |
| f1_macro / precision_macro / recall_macro | Macro-averaged |
| auc_roc | Area under the ROC curve |
| specificity | TN / (TN + FP) |

---

## Key Hyperparameters

```python
SCP1_CONFIG = {"n_classes": 2, "sequence_length": 896, "n_features": 6}
WAVELET_CONFIG = {"wavelet_type": "db4", "decomposition_level": 3}
LEARNED_WAVELET_CONFIG = {
    "levels": 3,
    "kernel_size": 16,         # adequado a seq_len=896 (448→224→112)
    "reg_energy": 1e-2, "reg_high_dc": 1e-2, "reg_smooth": 1e-3,
    "warm_start_db4": True,
}
DL_TRAINING_CONFIG = {"epochs": 100, "batch_size": 64, "early_stopping_patience": 15}
```

---

## Benchmark (referência — split padrão UEA 268/293)

Acurácias publicadas (Ruiz et al. 2021, bake-off multivariado), aproximadas:

| Método | accuracy |
|---|---|
| ROCKET / HIVE-COTE | ~0.88–0.92 |
| InceptionTime | ~0.84 |
| 1-NN DTW (dependente) | ~0.775 |
| MUSE | ~0.85 |

> Valores aproximados; confirmar no paper/`aeon` antes de citar. SCP1 é EEG real e ruidoso —
> o teto fica em torno de 0.90. O objetivo aqui é a comparação **interna** LWT vs db4-fixa vs raw.
