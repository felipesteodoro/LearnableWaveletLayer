"""
Simple long/short/flat strategy simulation from model predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_strategy(
    predictions: np.ndarray,
    prices: pd.Series | None = None,
    returns: np.ndarray | pd.Series | None = None,
    transaction_cost: float = 0.001,
    allow_short: bool = True,
    position_lag: int = 1,
) -> pd.Series:
    """
    Convert label predictions to a daily return series.

    Label convention:
      0 (sell) → short  (-1 × asset return)  or flat if allow_short=False
      1 (hold) → flat   (0 return)
      2 (buy)  → long   (+1 × asset return)

    Transaction cost is applied at every position change.

    Parameters
    ----------
    predictions      : array of predicted labels (0 / 1 / 2)
    prices           : close price Series — if provided, returns are computed
                       via pct_change(). Use when you have raw price data.
    returns          : pre-computed daily returns (log or pct) aligned with
                       predictions — preferred when prices are unavailable.
                       At least one of prices / returns must be provided.
    transaction_cost : one-way cost fraction (default 0.1%)
    allow_short      : if False, sell signals produce 0 instead of -1
    position_lag     : number of observations between signal and return realization.
                       Use 1 for next-bar daily returns and 0 for event returns
                       already aligned to the entry signal.

    Returns
    -------
    pd.Series of daily strategy returns
    """
    if position_lag < 0:
        raise ValueError("position_lag must be non-negative.")

    if returns is None and prices is not None:
        asset_returns = pd.Series(prices).reset_index(drop=True).pct_change().fillna(0)
    elif returns is not None:
        asset_returns = pd.Series(returns).reset_index(drop=True).fillna(0)
    else:
        raise ValueError("Either prices or returns must be provided.")

    preds = pd.Series(predictions).reset_index(drop=True)

    # Map labels to position: detecta 2-class vs 3-class automaticamente
    all_preds = set(preds.unique())
    if all_preds.issubset({0, 1}):  # meta-labeling binário: 0=down, 1=up
        pos_map = {0: -1 if allow_short else 0, 1: 1}
    else:  # 3-class: sell=0, hold=1, buy=2
        pos_map = {2: 1, 1: 0, 0: -1 if allow_short else 0}
    positions = preds.map(pos_map).fillna(0).astype(float)

    # Strategy return uses the position that is actually live for the return interval.
    if position_lag == 0:
        live_positions = positions
    else:
        live_positions = positions.shift(position_lag).fillna(0)
    strategy_returns = live_positions * asset_returns

    # One-way transaction cost is charged on every unit of turnover.
    # First entry costs |position|, and long<->short flips cost 2x.
    position_changes = positions.diff().abs().fillna(positions.abs())
    strategy_returns -= position_changes * transaction_cost
    strategy_returns = strategy_returns.clip(lower=-0.99)  # previne overflow no cumprod

    return strategy_returns


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def buy_and_hold_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0)
