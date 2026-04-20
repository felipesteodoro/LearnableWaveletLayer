# LearnableWaveletLayer

Research codebase for the paper:

> **Decomposing Financial Signals with Learnable Wavelet Transforms for Trend Classification in Brazilian Equities**

---

## Overview

This project proposes a **Learnable Wavelet Transform (LWT)** layer that can be integrated into deep neural networks. Instead of using a fixed wavelet basis (e.g., Daubechies-4), the layer learns its own low-pass filter coefficients during training and automatically derives the complementary high-pass filter via the **Quadrature Mirror Filter (QMF)** relationship, performing a multi-level pyramidal decomposition analogous to the classical DWT.

The LWT layer is evaluated across three progressive experiment tracks:

| Track | Problem | Dataset |
|---|---|---|
| Synthetic | Regression — signal denoising | Controlled synthetic time series |
| FordA | Classification — anomaly detection | UCR FordA benchmark (binary) |
| Financial | Classification — buy/sell/hold | 25 B3 Ibovespa equities (2005–2025) |

---

## Project Structure

```
LearnableWaveletLayer/
├── models/
│   └── LWT/
│       ├── fixed_db4_dwt.py               # Fixed Daubechies-4 DWT layer
│       ├── learned_wavelet_dwt_qmf.py     # Multi-level learnable DWT (QMF)
│       └── learned_wavelet_pair_qmf.py    # Single-level learnable wavelet pair
├── utils/
│   ├── data_utils.py                      # Download / load financial data
│   ├── features.py                        # Technical indicators + wavelet features
│   ├── labeling.py                        # Triple Barrier Method labeling
│   ├── validation.py                      # PurgedKFold cross-validation
│   ├── evaluation.py                      # Sharpe, Sortino, MDD, Alpha, Beta
│   └── model.py                           # CNN-LSTM factory
├── tests/
│   ├── synthetic/                         # Track 1 — regression on synthetic signals
│   ├── ford-a/                            # Track 2 — binary classification (FordA)
│   └── financial/                         # Track 3 — financial trend classification (B3)
├── data/                                  # Raw OHLCV CSVs for 25 B3 tickers
├── requirements.txt
└── 0-download-tickers-br.ipynb            # Download script for B3 data
```

---

## Learnable Wavelet Layer

### Core Architecture

The `LearnedWaveletPair1D_QMF` learns a single low-pass filter `h` and derives the high-pass filter `g` via the QMF conjugate relationship:

```
g[n] = (-1)^n * h[L-1-n]
```

This guarantees complementary coverage of the full frequency spectrum. Multiple pairs are stacked to form `LearnedWaveletDWT1D_QMF`, which performs a multi-level pyramidal decomposition (analogous to the classical DWT).

### Regularizers

Three regularization losses encourage well-behaved filters during training:

| Regularizer | Loss | Purpose |
|---|---|---|
| Energy | `(‖h‖₂ − 1)²` | Normalised filter energy |
| High-pass DC | `(Σg)²` | Zero mean for high-pass filter |
| Smoothness | `Σ(Δ²h)²` | Penalises oscillatory filters |

### Output Modes

- `"coeffs"` — returns `(A_L, [D₁, D₂, ..., D_L])` separately
- `"concat"` — concatenates all coefficients, aligned via upsampling/padding

---

## Experiment Tracks

### Track 1 — Synthetic Signal Regression

Controlled synthetic signals with known components (trend, harmonics, chirp, noise, spikes, regime changes). Goal: evaluate whether the learned wavelet filter can adapt to multi-scale signal structure better than a fixed db2/db4 baseline.

→ See [tests/synthetic/README.md](tests/synthetic/README.md)

### Track 2 — FordA Binary Classification

UCR FordA dataset: univariate 500-sample signals from automotive sensors, classified as Normal or Anomaly. Goal: validate the LWT layer on a standard time-series classification benchmark.

→ See [tests/ford-a/README.md](tests/ford-a/README.md)

### Track 3 — B3 Financial Classification (buy/sell/hold)

25 highly liquid Ibovespa equities, daily OHLCV, 2005–2025. Labels assigned via the Triple Barrier Method calibrated per asset. Cross-validation via PurgedKFold with 10-day embargo to prevent temporal leakage. GPU queue enables parallel training across 7 GPUs.

→ See [tests/financial/README.md](tests/financial/README.md)

---

## Models Compared

All three tracks compare four model families:

| Family | Variants |
|---|---|
| ML (classical) | Random Forest, XGBoost, LightGBM, CatBoost, Stacking |
| DL — Raw | CNN, LSTM, CNN-LSTM, Transformer |
| DL — Fixed Wavelet | same architectures, prepended with `FixedDb4DWT1D` |
| DL — Learnable Wavelet | same architectures, prepended with `LearnedWaveletDWT1D_QMF` |

---

## Installation

```bash
pip install -r requirements.txt
# Requires Python 3.11+, NVIDIA GPU + driver >= 570 (CUDA 12)
```

---

## Citation

> Felipe Teodoro. *Learnable Wavelet Transform for Financial Time Series Classification*. PhD research, 2025.
