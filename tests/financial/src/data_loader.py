"""Load and persist per-asset OHLCV data."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR      = Path(__file__).parent.parent.parent.parent / "data"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
LABELS_DIR    = Path(__file__).parent.parent / "data" / "labels"


def load_raw(ticker: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Read raw OHLCV CSV for one ticker."""
    path = (data_dir or DATA_DIR) / f"{ticker}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.sort_index()
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")
    return df


def load_processed(ticker: str) -> pd.DataFrame:
    """Read feature-engineered parquet for one ticker."""
    path = PROCESSED_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found for {ticker}. Run 00_data_preparation.ipynb first."
        )
    return pd.read_parquet(path)


def load_labels(ticker: str) -> pd.Series:
    """Read Triple Barrier labels (0/1/2) for one ticker."""
    path = LABELS_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Labels not found for {ticker}. Run 00_data_preparation.ipynb first."
        )
    return pd.read_parquet(path)["label"]


def save_processed(
    ticker: str,
    features: pd.DataFrame,
    labels: pd.Series,
    t1: pd.Series | None = None,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PROCESSED_DIR / f"{ticker}.parquet")
    label_df = labels.to_frame("label")
    if t1 is not None:
        label_df["t1"] = t1
    label_df.to_parquet(LABELS_DIR / f"{ticker}.parquet")


def load_t1(ticker: str) -> pd.Series | None:
    """Load the actual event-end dates (t1) saved alongside labels, or None."""
    path = LABELS_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "t1" not in df.columns:
        return None
    return df["t1"]


def load_all_raw(tickers: list[str], data_dir: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    return {t: load_raw(t, data_dir) for t in tickers}


def available_tickers(data_dir: Optional[Path] = None) -> list[str]:
    return sorted(
        p.stem for p in (data_dir or DATA_DIR).glob("*.csv")
    )
