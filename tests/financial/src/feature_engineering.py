"""
Compute financial features and enforce distribution quality.

All features are designed to be stationary:
  - ratios instead of raw levels
  - log-returns instead of prices
  - normalised by ATR or rolling std where needed
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError("Install pandas-ta: pip install pandas-ta")

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    raise ImportError("Install statsmodels: pip install statsmodels")


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute ~25 stationary features from OHLCV data.

    Parameters
    ----------
    df  : DataFrame with columns Open, High, Low, Close, Volume (+ Adj Close optional)
    cfg : feature config dict (from experiment_config.FEATURE_CONFIG)

    Returns
    -------
    DataFrame aligned to df.index, columns = feature names
    """
    if cfg is None:
        from config.experiment_config import FEATURE_CONFIG
        cfg = FEATURE_CONFIG

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    out = pd.DataFrame(index=df.index)

    # ── Returns ─────────────────────────────────────────────────────────────
    out["log_return_1"] = np.log(close / close.shift(1))
    out["log_return_5"] = np.log(close / close.shift(5))

    # ── Momentum ─────────────────────────────────────────────────────────────
    rsi_p = cfg.get("rsi_period", 14)
    out["rsi_14"] = ta.rsi(close, length=rsi_p)

    stoch = ta.stoch(high, low, close, k=cfg.get("stoch_period", 14))
    if stoch is not None and not stoch.empty:
        out["stoch_k_14"] = stoch.iloc[:, 0]

    willr = ta.willr(high, low, close, length=cfg.get("williams_period", 14))
    if willr is not None:
        out["williams_r_14"] = willr

    roc_p = cfg.get("roc_period", 10)
    out["roc_10"] = close.pct_change(roc_p)

    # ── Trend (normalised by price) ──────────────────────────────────────────
    ema_p = cfg.get("ema_period", 20)
    sma_p = cfg.get("sma_period", 50)
    ema   = ta.ema(close, length=ema_p)
    sma   = ta.sma(close, length=sma_p)
    out["ema_ratio_20"] = (close / ema) - 1
    out["sma_ratio_50"] = (close / sma) - 1

    macd_res = ta.macd(
        close,
        fast=cfg.get("macd_fast", 12),
        slow=cfg.get("macd_slow", 26),
        signal=cfg.get("macd_signal", 9),
    )
    atr_p   = cfg.get("atr_period", 14)
    atr_raw = ta.atr(high, low, close, length=atr_p)
    atr_safe = atr_raw.replace(0, np.nan)
    if macd_res is not None and not macd_res.empty:
        macd_col   = [c for c in macd_res.columns if c.startswith("MACD_")][0]
        signal_col = [c for c in macd_res.columns if "MACDs" in c][0]
        hist_col   = [c for c in macd_res.columns if "MACDh" in c][0]
        out["macd_norm"]        = macd_res[macd_col]   / atr_safe
        out["macd_signal_norm"] = macd_res[signal_col] / atr_safe
        out["macd_hist_norm"]   = macd_res[hist_col]   / atr_safe

    adx_res = ta.adx(high, low, close, length=cfg.get("adx_period", 14))
    if adx_res is not None and not adx_res.empty:
        adx_col = [c for c in adx_res.columns if c.startswith("ADX_")][0]
        out["adx_14"] = adx_res[adx_col]

    # ── Volatility ───────────────────────────────────────────────────────────
    out["atr_norm"] = atr_raw / close

    bb_res = ta.bbands(close, length=cfg.get("bb_period", 20))
    if bb_res is not None and not bb_res.empty:
        upper_col = [c for c in bb_res.columns if "BBU" in c][0]
        lower_col = [c for c in bb_res.columns if "BBL" in c][0]
        mid_col   = [c for c in bb_res.columns if "BBM" in c][0]
        band_width = (bb_res[upper_col] - bb_res[lower_col]).replace(0, np.nan)
        out["bb_width_norm"] = band_width / bb_res[mid_col]
        out["bb_position"]   = (close - bb_res[lower_col]) / band_width

    hv_p = cfg.get("hv_period", 21)
    out["hv_21"] = out["log_return_1"].rolling(hv_p).std() * np.sqrt(252)

    # Garman-Klass volatility (uses OHLC, lower noise than close-to-close)
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / df["Open"]) ** 2
    out["garman_klass_vol"] = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_sma_p = cfg.get("vol_sma_period", 20)
    vol_sma   = volume.rolling(vol_sma_p).mean().replace(0, np.nan)
    out["volume_ratio"] = volume / vol_sma

    # OBV rate of change (stationary; raw OBV drifts)
    obv = ta.obv(close, volume)
    if obv is not None:
        out["obv_roc_10"] = obv.pct_change(roc_p)

    # Force Index normalised by ATR
    force = (close - close.shift(1)) * volume
    out["force_index_norm"] = force / (atr_safe * vol_sma)

    # ── Statistical ─────────────────────────────────────────────────────────
    zp = cfg.get("zscore_period", 20)
    roll_mean = close.rolling(zp).mean()
    roll_std  = close.rolling(zp).std().replace(0, np.nan)
    out["zscore_20"] = (close - roll_mean) / roll_std

    ret_mean = out["log_return_1"].rolling(zp).mean()
    ret_std  = out["log_return_1"].rolling(zp).std().replace(0, np.nan)
    out["zscore_returns_20"] = (out["log_return_1"] - ret_mean) / ret_std

    lag  = cfg.get("autocorr_lag", 1)
    out["autocorr_returns_20"] = (
        out["log_return_1"]
        .rolling(zp)
        .apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=False)
    )

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Distribution quality tests
# ---------------------------------------------------------------------------

def test_distribution(series: pd.Series) -> dict:
    """Per-feature distribution diagnostics."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 30:
        return {"adf_pvalue": 1.0, "kurtosis": np.nan, "nan_ratio": 1.0, "ok": False}

    nan_ratio = series.isna().mean()

    # Winsorise at 1%-99% before kurtosis
    p1, p99 = clean.quantile([0.01, 0.99])
    winsorised = clean.clip(p1, p99)
    kurt = float(stats.kurtosis(winsorised, fisher=True))

    try:
        adf_stat, adf_pvalue, *_ = adfuller(clean, autolag="AIC")
    except Exception:
        adf_pvalue = 1.0

    return {
        "adf_pvalue": float(adf_pvalue),
        "kurtosis": kurt,
        "nan_ratio": float(nan_ratio),
        "ok": adf_pvalue < 0.05 and abs(kurt) < 20 and nan_ratio < 0.05,
    }


def select_features(
    dfs: dict[str, pd.DataFrame],
    cfg: Optional[dict] = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Test each feature across all assets and return (selected_columns, report).

    A feature is removed if it fails thresholds in more than `adf_max_failing_pct`
    of assets, or if it is highly correlated with another feature.

    Parameters
    ----------
    dfs : {ticker: feature_df} — one DataFrame per asset (from compute_features)
    cfg : FEATURE_CONFIG

    Returns
    -------
    selected_cols : list of column names that passed
    report        : DataFrame with per-feature diagnostic summary
    """
    if cfg is None:
        from config.experiment_config import FEATURE_CONFIG
        cfg = FEATURE_CONFIG

    max_fail_pct   = cfg.get("adf_max_failing_pct", 0.30)
    max_kurt       = cfg.get("max_kurtosis", 20.0)
    max_nan        = cfg.get("max_nan_pct", 0.05)
    max_corr       = cfg.get("max_correlation", 0.95)

    tickers = list(dfs.keys())
    feature_cols = list(next(iter(dfs.values())).columns)
    records = []

    for col in feature_cols:
        col_results = [test_distribution(df[col]) for df in dfs.values() if col in df.columns]
        fail_pct    = sum(1 for r in col_results if not r["ok"]) / len(col_results)
        avg_kurt    = float(np.nanmean([r["kurtosis"] for r in col_results]))
        avg_nan     = float(np.nanmean([r["nan_ratio"] for r in col_results]))
        avg_adf     = float(np.nanmean([r["adf_pvalue"] for r in col_results]))

        passed = (
            fail_pct <= max_fail_pct
            and abs(avg_kurt) <= max_kurt
            and avg_nan <= max_nan
        )
        records.append({
            "feature": col,
            "fail_pct": round(fail_pct, 3),
            "avg_adf_pvalue": round(avg_adf, 4),
            "avg_kurtosis": round(avg_kurt, 2),
            "avg_nan_ratio": round(avg_nan, 4),
            "passed_distribution": passed,
            "removed_reason": "" if passed else (
                "non-stationary" if avg_adf >= 0.05
                else "high kurtosis" if abs(avg_kurt) > max_kurt
                else "too many NaN"
            ),
        })

    report = pd.DataFrame(records).set_index("feature")
    selected = report[report["passed_distribution"]].index.tolist()

    # Correlation filter: among selected, drop one from each correlated pair
    if len(selected) > 1:
        # Use mean values across all assets to build a single correlation matrix
        combined = pd.concat(
            [dfs[t][selected].add_suffix(f"__{t}") for t in tickers], axis=1
        )
        # Average pairwise correlation across assets
        corr_per_asset = [dfs[t][selected].corr().abs() for t in tickers]
        avg_corr = pd.concat(
            [c.stack() for c in corr_per_asset], axis=1
        ).mean(axis=1).unstack()

        to_drop: set[str] = set()
        for i, f1 in enumerate(selected):
            for f2 in selected[i + 1:]:
                if f2 not in to_drop and avg_corr.loc[f1, f2] > max_corr:
                    to_drop.add(f2)
                    report.loc[f2, "removed_reason"] = f"corr>{max_corr:.2f} with {f1}"
                    report.loc[f2, "passed_distribution"] = False

        selected = [f for f in selected if f not in to_drop]

    return selected, report
