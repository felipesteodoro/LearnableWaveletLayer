"""
Simple long/short/flat strategy simulation from model predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_strategy(
    predictions: np.ndarray,
    prices: pd.Series,
    transaction_cost: float = 0.001,
    allow_short: bool = True,
) -> pd.Series:
    """
    Convert label predictions to a daily return series.

    Label convention:
      0 (sell) → short  (-1 × price return)  or flat if allow_short=False
      1 (hold) → flat   (0 return)
      2 (buy)  → long   (+1 × price return)

    Transaction cost is applied at every position change.

    Parameters
    ----------
    predictions      : array of predicted labels (0 / 1 / 2)
    prices           : close price Series aligned with predictions
    transaction_cost : one-way cost fraction (default 0.1%)
    allow_short      : if False, sell signals produce 0 instead of -1

    Returns
    -------
    pd.Series of daily strategy returns (same index as prices)
    """
    prices = pd.Series(prices).reset_index(drop=True)
    preds  = pd.Series(predictions).reset_index(drop=True)

    # Map labels to position: buy=1, hold=0, sell=-1 (or 0 if no short)
    pos_map = {2: 1, 1: 0, 0: -1 if allow_short else 0}
    positions = preds.map(pos_map).fillna(0).astype(float)

    # Price returns (next-day close-to-close)
    price_returns = prices.pct_change().fillna(0)

    # Strategy return = position(t-1) × price_return(t)
    strategy_returns = positions.shift(1).fillna(0) * price_returns

    # Transaction costs at position changes
    position_changes = (positions.diff().abs() > 0).astype(float)
    strategy_returns -= position_changes * transaction_cost

    strategy_returns.index = prices.index
    return strategy_returns


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def buy_and_hold_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0)
