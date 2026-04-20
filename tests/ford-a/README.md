# Track 2 — FordA Binary Classification

**Task**: classify univariate time series as Normal (−1) or Anomaly (+1).  
**Goal**: validate the LWT layer on a standard UCR benchmark before applying it to financial data.

---

## Dataset

**FordA** — UCR Time Series Classification Archive  
Automotive engine sensor signals; each sample is a 500-point univariate time series.

| Split | Samples |
|---|---|
| Train | 3,060 |
| Validation | 541 |
| Test | 1,320 |

Classes: **Normal** and **Anomaly** (binary).

Downloaded and preprocessed by `00_download_and_feature_extraction.ipynb`.

---

## Structure

```
tests/ford-a/
├── config/
│   └── experiment_config.py       # Centralised hyperparameters
├── src/
│   ├── data_loader.py             # Load FordA splits
│   ├── feature_extraction.py      # WaveletFeatureExtractor
│   ├── models.py                  # ML and DL model factories
│   ├── evaluation.py              # ClassificationEvaluator + ResultsManager
│   └── pipeline.py                # ExperimentPipeline
├── data/                          # Raw and wavelet-extracted features
├── results/                       # Metrics JSON + predictions .npz
├── saved_models/                  # Trained model weights
└── catboost_info/                 # CatBoost training logs
```

---

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 00 | `00_download_and_feature_extraction.ipynb` | Download FordA, extract wavelet features, save splits |
| 01 | `01_ml_experiments.ipynb` | ML classification with wavelet features |
| 02 | `02_dl_raw_experiments.ipynb` | DL on raw signals (CNN, LSTM, CNN-LSTM, Transformer) |
| 03 | `03_dl_wavelet_experiments.ipynb` | DL on fixed db2 wavelet coefficients |
| 04 | `04_learned_wavelet_experiments.ipynb` | DL with `LearnedWaveletDWT1D_QMF` |
| 05 | `05_comparison_analysis.ipynb` | Aggregate results, ranking tables |
| 06 | `06_learned_filter_analysis.ipynb` | Frequency response of learned filters |

---

## Models

**ML** (wavelet features → scalar vector → classifier):
- LinearSVC, SGDClassifier, LogisticRegression
- RandomForest, XGBoost, LightGBM, CatBoost, Stacking

Cross-validation: `RandomizedSearchCV` with 5-fold CV (15–20 iterations per model).

**DL** (three input modes):

| Mode | Input |
|---|---|
| `raw` | Raw 500-sample sequence |
| `db2` | Fixed db2 wavelet coefficients |
| `learned_wavelet` | Coefficients from `LearnedWaveletDWT1D_QMF` |

Architectures: CNN · LSTM · CNN-LSTM · Transformer

---

## Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy |
| F1 | Binary F1 score |
| Precision / Recall | Per-class precision and recall |
| AUC-ROC | Area under the ROC curve |

---

## Key Hyperparameters

```python
WAVELET_CONFIG = {
    "wavelet_type": "db2",
    "decomposition_level": 2,
}
LEARNED_WAVELET_CONFIG = {
    "levels": 2,
    "kernel_size": 32,
    "reg_energy": 1e-2,
    "reg_high_dc": 1e-2,
    "reg_smooth": 1e-3,
}
DL_TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 64,
    "early_stopping_patience": 15,
}
```
