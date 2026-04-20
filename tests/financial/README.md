# Track 3 — B3 Financial Classification (buy / sell / hold)

**Task**: classify daily price trends of Brazilian equities as buy (2), hold (1), or sell (0).  
**Goal**: evaluate whether domain-adaptive wavelet filters improve trend prediction stability and financial performance across 25 Ibovespa stocks.

---

## Dataset

- **Assets**: 25 highly liquid B3 equities from the Ibovespa index
- **Frequency**: daily OHLCV
- **Period**: 2005 – 2025
- **Labeling**: Triple Barrier Method (take-profit, stop-loss, time-horizon barriers calibrated per asset volatility)
- **Cross-validation**: PurgedKFold (k = 5, 10-day embargo) — prevents temporal leakage

Tickers: ABEV3, B3SA3, BBAS3, BBDC4, BRKM5, COGN3, CSNA3, CYRE3, EZTC3, GGBR4, HYPE3, ITUB4, LREN3, MGLU3, MRVE3, MULT3, PETR4, RADL3, RENT3, SUZB3, UGPA3, USIM5, VALE3, VIVT3, WEGE3

---

## Structure

```
tests/financial/
├── config/
│   └── experiment_config.py       # Centralised hyperparameters
├── src/
│   ├── data_loader.py             # Load raw CSV / processed parquets
│   ├── feature_engineering.py     # ~25 stationary features + distribution tests
│   ├── labeling.py                # Triple Barrier wrapper (labels: 0/1/2)
│   ├── models.py                  # DL model factory (classification, softmax)
│   ├── evaluation.py              # ClassificationEvaluator + FinancialMetrics + ResultsManager
│   ├── backtest.py                # Long/short/flat strategy simulation
│   └── pipeline.py                # FinancialExperimentPipeline (used by queue + notebooks)
├── queue/
│   ├── job.py                     # ExperimentJob dataclass
│   ├── job_queue.py               # GPUJobQueueManager (resume after crash)
│   ├── worker.py                  # GPUWorker (1 thread per GPU)
│   ├── dashboard.py               # Rich terminal monitor
│   └── experiment_runner.py       # Subprocess entry point called by each GPU worker
├── data/
│   ├── processed/{ticker}.parquet # Features (created by notebook 00)
│   └── labels/{ticker}.parquet    # Triple Barrier labels
├── results/
│   └── {ticker}/{model}_{mode}/
│       ├── metrics.json           # ML + financial metrics (all folds aggregated)
│       └── predictions_fold*.npz  # y_true / y_pred per fold
├── saved_models/
│   └── {ticker}/{model}_{mode}/fold_*.keras
├── run_dl_queue.py                # Main DL queue launcher
└── README.md
```

---

## Features (~25 stationary features)

All features are verified for stationarity (ADF test) and distribution quality (kurtosis, NaN ratio, pairwise correlation) before use. Failing features are removed automatically by `src/feature_engineering.py`.

| Group | Features |
|---|---|
| Returns | `log_return_1`, `log_return_5` |
| Momentum | `rsi_14`, `stoch_k_14`, `williams_r_14`, `roc_10` |
| Trend | `ema_ratio_20`, `sma_ratio_50`, `macd_norm`, `macd_signal_norm`, `macd_hist_norm`, `adx_14` |
| Volatility | `atr_norm`, `bb_width_norm`, `bb_position`, `hv_21`, `garman_klass_vol` |
| Volume | `volume_ratio`, `obv_roc_10`, `force_index_norm` |
| Statistical | `zscore_20`, `zscore_returns_20`, `autocorr_returns_20` |

---

## Metrics

**ML metrics** (per fold, then averaged):

| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy |
| F1-macro | Macro-averaged F1 across 3 classes |
| F1 per class | F1 for sell, hold, and buy separately |
| MCC | Matthews Correlation Coefficient |
| AUC-ROC (OvR) | One-vs-rest ROC area |

**Financial metrics** (strategy simulation with 0.1% transaction cost):

| Metric | Description |
|---|---|
| Sharpe Ratio | Annualised risk-adjusted return |
| Sortino Ratio | Penalises only downside volatility |
| Calmar Ratio | CAGR / Max Drawdown |
| Max Drawdown | Largest peak-to-valley decline |
| CAGR | Compound annual growth rate |
| Win Rate | Fraction of profitable trades |
| Profit Factor | Gross profit / gross loss |
| VaR / CVaR (95%) | Value at Risk / Expected Shortfall |
| Alpha / Beta | Excess return / correlation vs Buy-and-Hold |

---

## Execution Order

### Step 1 — Data preparation (run once, all assets)

```bash
cd tests/financial
jupyter notebook 00_data_preparation.ipynb
```

Loads raw CSVs → computes features → tests distributions → removes failing features → applies Triple Barrier → saves `data/processed/` and `data/labels/`.

---

### Step 2 — Feature analysis (optional, all assets)

```bash
jupyter notebook 01_feature_analysis.ipynb
```

Histograms, ADF/KPSS stationarity tests, correlation heatmap, label distribution per asset.

---

### Step 3 — ML experiments (all assets, CPU parallel)

```bash
jupyter notebook 02_ml_experiments.ipynb
```

Runs Random Forest, XGBoost, LightGBM, CatBoost, and Stacking for all 25 assets in parallel via `joblib.Parallel`. Results saved to `results/{ticker}/ml_{model}/metrics.json`.

---

### Step 4 — DL experiments (GPU queue, resumable)

Open two terminals:

```bash
# Terminal 1 — start (or resume) the job queue
cd tests/financial
python run_dl_queue.py

# Terminal 2 — real-time dashboard
python queue/dashboard.py
```

- **300 jobs**: 25 assets × 4 models × 3 modes
- **7 GPUs** in parallel; each GPU runs one job at a time
- **Auto-resume**: if the machine crashes, just run `python run_dl_queue.py` again — completed jobs are skipped automatically
- **Force fresh start**: `python run_dl_queue.py --fresh`

**Run a subset** (via environment variables):

```bash
TICKERS="PETR4.SA,VALE3.SA" DL_MODELS="CNN,LSTM" MODES="raw,learned_wavelet" python run_dl_queue.py
GPU_IDS="0,1,2" python run_dl_queue.py        # use only GPUs 0, 1, 2
```

**Interactive single-asset DL** (notebook 03):

```bash
jupyter notebook 03_dl_experiments.ipynb
# Set TICKER, MODEL_NAME, MODE in the first cell
```

---

### Step 5 — Comparison analysis (run after queue finishes)

```bash
jupyter notebook 04_comparison_analysis.ipynb
```

Aggregates all results, ranks models by F1-macro, tests statistical significance of learned wavelets vs baselines, produces heatmaps and box plots.

---

### Step 6 — Financial analysis

```bash
jupyter notebook 05_financial_analysis.ipynb
```

Sharpe ratio distribution, equity curve comparison (strategy vs Buy-and-Hold), risk metrics (MDD, VaR, CVaR), best configurations per ticker.

---

## Resume After Crash

The queue implements two independent layers of protection:

1. **`queue_status.json`** — persisted after every state change. On restart, `GPUJobQueueManager.resume_or_create()` reloads the file, skips `done` / permanently-`failed` jobs, and re-queues anything that was `running` or `pending` at crash time.

2. **`results/{ticker}/{model}_{mode}/metrics.json`** — `experiment_runner.py` checks for this file before starting any experiment. Even if the checkpoint is corrupted, a job with existing results exits immediately with success.

---

## Key Hyperparameters

```python
FEATURE_CONFIG   = {"sequence_length": 30}      # sliding window (days)
LABELING_CONFIG  = {"pt_sl": [1.5, 1.0],         # Triple Barrier multipliers
                    "time_horizon": 10}           # max holding period (days)
VALIDATION_CONFIG = {"n_folds": 5,
                     "embargo_days": 10}
DL_TRAINING_CONFIG = {"epochs": 100,
                      "batch_size": 64,
                      "early_stopping_patience": 15}
LEARNED_WAVELET_CONFIG = {"levels": 2,
                           "kernel_size": 32,
                           "reg_energy": 1e-2}
BACKTEST_CONFIG  = {"transaction_cost": 0.001,   # 0.1% per trade
                    "allow_short": True}
```
