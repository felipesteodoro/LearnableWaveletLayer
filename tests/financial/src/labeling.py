"""
Triple Barrier labeling adapted for 3-class classification.

Wraps utils/labeling.py and remaps {-1, 0, 1} → {0, 1, 2} (sell/hold/buy).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Allow importing from project utils/
_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from utils.labeling import apply_labeling  # noqa: E402


_LABEL_MAP = {-1: 0, 0: 1, 1: 2}   # sell=0, hold=1, buy=2
CLASS_NAMES = {0: "sell", 1: "hold", 2: "buy"}


def label_asset(
    df: pd.DataFrame,
    cfg: Optional[dict] = None,
) -> pd.Series:
    """
    Apply Triple Barrier labeling to a single asset.

    Parameters
    ----------
    df  : OHLCV DataFrame (must have 'Close' column)
    cfg : LABELING_CONFIG dict

    Returns
    -------
    Series with integer labels 0 (sell), 1 (hold), 2 (buy), indexed like df.
    Missing labels (warmup rows) are dropped.
    """
    if cfg is None:
        from config.experiment_config import LABELING_CONFIG
        cfg = LABELING_CONFIG

    labeled = apply_labeling(
        df,
        pt_sl=cfg.get("pt_sl", [1.5, 1.0]),
        min_ret=cfg.get("min_ret", 0.001),
        num_days=cfg.get("time_horizon", 10),
    )

    labels = labeled["label"].map(_LABEL_MAP).dropna().astype(int)
    return labels


def label_distribution(labels: pd.Series) -> pd.Series:
    """Return percentage distribution of each class."""
    counts = labels.value_counts().sort_index()
    counts.index = counts.index.map(CLASS_NAMES)
    return (counts / counts.sum() * 100).round(2)
