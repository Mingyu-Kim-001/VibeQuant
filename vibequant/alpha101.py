"""
WorldQuant 101 Formulaic Alphas Implementation

Based on: Kakushadze, Z. (2016). 101 Formulaic Alphas. arXiv:1601.00991
https://arxiv.org/abs/1601.00991

These are price-volume based alphas with short holding periods (0.6-6.4 days average).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Callable
from dataclasses import dataclass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank (percentile rank across assets at each time)."""
    return x.rank(axis=1, pct=True)


def ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Time-series rank over window (how current value ranks vs past values)."""
    return x.rolling(window).apply(
        lambda arr: stats.rankdata(arr)[-1] / len(arr),
        raw=True
    )


def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Position of max value in rolling window (0 = oldest, window-1 = newest)."""
    return x.rolling(window).apply(np.argmax, raw=True)


def ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Position of min value in rolling window."""
    return x.rolling(window).apply(np.argmin, raw=True)


def ts_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling max."""
    return x.rolling(window).max()


def ts_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling min."""
    return x.rolling(window).min()


def ts_sum(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling sum."""
    return x.rolling(window).sum()


def ts_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling standard deviation."""
    return x.rolling(window).std()


def ts_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling mean (SMA)."""
    return x.rolling(window).mean()


def delta(x: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Difference from period days ago."""
    return x.diff(period)


def delay(x: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Value from period days ago."""
    return x.shift(period)


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling correlation between x and y."""
    return x.rolling(window).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling covariance between x and y."""
    return x.rolling(window).cov(y)


def scale(x: pd.DataFrame, target: float = 1.0) -> pd.DataFrame:
    """Scale to sum to target (cross-sectionally)."""
    return x.div(x.abs().sum(axis=1), axis=0) * target


def decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Linearly weighted moving average (recent values weighted higher)."""
    weights = np.arange(1, window + 1).astype(float)
    weights = weights / weights.sum()
    return x.rolling(window).apply(lambda arr: (arr * weights).sum(), raw=True)


def sign(x: pd.DataFrame) -> pd.DataFrame:
    """Sign of values (-1, 0, or 1)."""
    return np.sign(x)


def log(x: pd.DataFrame) -> pd.DataFrame:
    """Natural logarithm."""
    return np.log(x.replace(0, np.nan))


def power(x: pd.DataFrame, exp: float) -> pd.DataFrame:
    """Element-wise power."""
    return np.power(x, exp)


# =============================================================================
# ALPHA IMPLEMENTATIONS
# =============================================================================

@dataclass
class AlphaResult:
    """Result from alpha calculation."""
    name: str
    description: str
    alpha: pd.DataFrame  # Raw alpha values
    signals: pd.DataFrame  # Tradeable signals (normalized weights)


def alpha001(close: pd.DataFrame, returns: pd.DataFrame) -> AlphaResult:
    """
    Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)

    Simplified: Rank of time-series argmax of squared values.
    For negative returns, use volatility; for positive, use close.
    """
    stddev = ts_std(returns, 20)
    inner = close.where(returns >= 0, stddev)
    inner_sq = inner ** 2
    alpha = rank(ts_argmax(inner_sq, 5)) - 0.5

    # Convert to signals: long positive alphas, short negative
    signals = alpha.copy()
    # Normalize per row
    signals = signals.sub(signals.mean(axis=1), axis=0)
    row_sum = signals.abs().sum(axis=1).replace(0, 1)
    signals = signals.div(row_sum, axis=0)

    return AlphaResult(
        name="Alpha001",
        description="Volatility regime rank",
        alpha=alpha,
        signals=signals.fillna(0),
    )


def alpha002(open_: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame) -> AlphaResult:
    """
    Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))

    Correlation between volume change rank and intraday return rank.
    """
    vol_delta = delta(log(volume), 2)
    intraday_ret = (close - open_) / open_

    alpha = -1 * correlation(rank(vol_delta), rank(intraday_ret), 6)

    signals = alpha.copy()
    signals = signals.sub(signals.mean(axis=1), axis=0)
    row_sum = signals.abs().sum(axis=1).replace(0, 1)
    signals = signals.div(row_sum, axis=0)

    return AlphaResult(
        name="Alpha002",
        description="Volume-price correlation",
        alpha=alpha,
        signals=signals.fillna(0),
    )


def alpha006(open_: pd.DataFrame, volume: pd.DataFrame) -> AlphaResult:
    """
    Alpha#6: (-1 * correlation(open, volume, 10))

    Negative correlation between open price and volume.
    """
    alpha = -1 * correlation(open_, volume, 10)

    signals = alpha.copy()
    signals = signals.sub(signals.mean(axis=1), axis=0)
    row_sum = signals.abs().sum(axis=1).replace(0, 1)
    signals = signals.div(row_sum, axis=0)

    return AlphaResult(
        name="Alpha006",
        description="Open-volume correlation",
        alpha=alpha,
        signals=signals.fillna(0),
    )


def alpha012(close: pd.DataFrame, volume: pd.DataFrame) -> AlphaResult:
    """
    Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))

    Volume direction times negative price direction.
    Buy when volume up + price down, sell when volume down + price up.
    """
    alpha = sign(delta(volume, 1)) * (-1 * delta(close, 1))

    signals = alpha.copy()
    signals = signals.sub(signals.mean(axis=1), axis=0)
    row_sum = signals.abs().sum(axis=1).replace(0, 1)
    signals = signals.div(row_sum, axis=0)

    return AlphaResult(
        name="Alpha012",
        description="Volume-price divergence",
        alpha=alpha,
        signals=signals.fillna(0),
    )


def alpha033(open_: pd.DataFrame, close: pd.DataFrame) -> AlphaResult:
    """
    Alpha#33: rank((-1 + (open / close)))

    Rank of negative gap percentage.
    Stocks that gapped down (open < close) rank higher.
    """
    alpha = rank(-1 + (open_ / close))

    signals = alpha.copy()
    signals = signals.sub(signals.mean(axis=1), axis=0)
    row_sum = signals.abs().sum(axis=1).replace(0, 1)
    signals = signals.div(row_sum, axis=0)

    return AlphaResult(
        name="Alpha033",
        description="Gap rank (mean reversion)",
        alpha=alpha,
        signals=signals.fillna(0),
    )


def alpha041(high: pd.DataFrame, low: pd.DataFrame, vwap: pd.DataFrame) -> AlphaResult:
    """
    Alpha#41: (((high * low)^0.5) - vwap)

    Geometric mean of high/low minus VWAP.
    """
    alpha = power(high * low, 0.5) - vwap

    signals = alpha.copy()
    signals = signals.sub(signals.mean(axis=1), axis=0)
    row_sum = signals.abs().sum(axis=1).replace(0, 1)
    signals = signals.div(row_sum, axis=0)

    return AlphaResult(
        name="Alpha041",
        description="Price range vs VWAP",
        alpha=alpha,
        signals=signals.fillna(0),
    )


def alpha042(close: pd.DataFrame, vwap: pd.DataFrame) -> AlphaResult:
    """
    Alpha#42: (rank((vwap - close)) / rank((vwap + close)))

    Ratio of ranks: VWAP vs close relationship.
    """
    alpha = rank(vwap - close) / rank(vwap + close).replace(0, np.nan)

    signals = alpha.copy()
    signals = signals.sub(signals.mean(axis=1), axis=0)
    row_sum = signals.abs().sum(axis=1).replace(0, 1)
    signals = signals.div(row_sum, axis=0)

    return AlphaResult(
        name="Alpha042",
        description="VWAP-close ratio",
        alpha=alpha,
        signals=signals.fillna(0),
    )


def alpha053(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> AlphaResult:
    """
    Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))

    Change in Williams %R-like indicator over 9 days.
    """
    denom = (close - low).replace(0, np.nan)
    inner = ((close - low) - (high - close)) / denom
    alpha = -1 * delta(inner, 9)

    signals = alpha.copy()
    signals = signals.sub(signals.mean(axis=1), axis=0)
    row_sum = signals.abs().sum(axis=1).replace(0, 1)
    signals = signals.div(row_sum, axis=0)

    return AlphaResult(
        name="Alpha053",
        description="Williams %R momentum",
        alpha=alpha,
        signals=signals.fillna(0),
    )


def alpha054(open_: pd.DataFrame, close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> AlphaResult:
    """
    Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

    Complex price ratio alpha.
    """
    num = -1 * (low - close) * power(open_, 5)
    denom = (low - high).replace(0, np.nan) * power(close, 5).replace(0, np.nan)
    alpha = num / denom

    # Clip extreme values
    alpha = alpha.clip(-10, 10)

    signals = alpha.copy()
    signals = signals.sub(signals.mean(axis=1), axis=0)
    row_sum = signals.abs().sum(axis=1).replace(0, 1)
    signals = signals.div(row_sum, axis=0)

    return AlphaResult(
        name="Alpha054",
        description="Price ratio alpha",
        alpha=alpha,
        signals=signals.fillna(0),
    )


def alpha101(open_: pd.DataFrame, close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> AlphaResult:
    """
    Alpha#101: ((close - open) / ((high - low) + .001))

    Intraday momentum: close-open relative to day's range.
    """
    alpha = (close - open_) / ((high - low) + 0.001)

    signals = alpha.copy()
    signals = signals.sub(signals.mean(axis=1), axis=0)
    row_sum = signals.abs().sum(axis=1).replace(0, 1)
    signals = signals.div(row_sum, axis=0)

    return AlphaResult(
        name="Alpha101",
        description="Intraday momentum",
        alpha=alpha,
        signals=signals.fillna(0),
    )


# =============================================================================
# LONG-ONLY STRATEGY GENERATORS
# =============================================================================

def alpha_to_long_only_signals(
    alpha: pd.DataFrame,
    top_n: int = 10,
    hold_period: int = 5,
) -> pd.DataFrame:
    """
    Convert alpha values to long-only signals.

    Args:
        alpha: Alpha values (higher = better)
        top_n: Number of top stocks to hold
        hold_period: Days between rebalances

    Returns:
        Long-only position weights
    """
    signals = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)

    for i in range(0, len(alpha), hold_period):
        end_idx = min(i + hold_period, len(alpha))
        alpha_row = alpha.iloc[i].dropna()

        if len(alpha_row) >= top_n:
            top_stocks = alpha_row.nlargest(top_n).index.tolist()
            weight = 1.0 / len(top_stocks)
            for j in range(i, end_idx):
                for sym in top_stocks:
                    if sym in signals.columns:
                        signals.loc[alpha.index[j], sym] = weight

    return signals.fillna(0)


def create_alpha_strategy(
    alpha_func: Callable,
    alpha_name: str,
    top_n: int = 10,
    hold_period: int = 5,
) -> str:
    """
    Create a strategy code string from an alpha function.

    Returns Python code that can be used with the backtest system.
    """
    code = f'''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Strategy based on {alpha_name}."""
    from vibequant.alpha101 import {alpha_func.__name__}, alpha_to_long_only_signals

    # We need OHLCV data, but if only close is available, approximate
    close = prices
    open_ = prices.shift(1).fillna(prices)  # Approximate: yesterday's close
    high = prices.rolling(1).max()  # For daily data, high ≈ close
    low = prices.rolling(1).min()   # For daily data, low ≈ close
    volume = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)  # Placeholder
    vwap = prices  # Approximate: close ≈ vwap for daily

    # Calculate alpha
    returns = prices.pct_change()
    result = {alpha_func.__name__}(
        open_=open_, close=close, high=high, low=low,
        volume=volume, vwap=vwap, returns=returns
    )

    # Convert to long-only signals
    signals = alpha_to_long_only_signals(
        result.alpha, top_n={top_n}, hold_period={hold_period}
    )
    return signals
'''
    return code


# =============================================================================
# ALL ALPHAS DICT (for easy iteration)
# =============================================================================

ALPHA_FUNCTIONS = {
    "alpha001": {
        "func": alpha001,
        "inputs": ["close", "returns"],
        "description": "Volatility regime rank",
    },
    "alpha002": {
        "func": alpha002,
        "inputs": ["open", "close", "volume"],
        "description": "Volume-price correlation",
    },
    "alpha006": {
        "func": alpha006,
        "inputs": ["open", "volume"],
        "description": "Open-volume correlation",
    },
    "alpha012": {
        "func": alpha012,
        "inputs": ["close", "volume"],
        "description": "Volume-price divergence",
    },
    "alpha033": {
        "func": alpha033,
        "inputs": ["open", "close"],
        "description": "Gap rank (mean reversion)",
    },
    "alpha041": {
        "func": alpha041,
        "inputs": ["high", "low", "vwap"],
        "description": "Price range vs VWAP",
    },
    "alpha042": {
        "func": alpha042,
        "inputs": ["close", "vwap"],
        "description": "VWAP-close ratio",
    },
    "alpha053": {
        "func": alpha053,
        "inputs": ["close", "high", "low"],
        "description": "Williams %R momentum",
    },
    "alpha054": {
        "func": alpha054,
        "inputs": ["open", "close", "high", "low"],
        "description": "Price ratio alpha",
    },
    "alpha101": {
        "func": alpha101,
        "inputs": ["open", "close", "high", "low"],
        "description": "Intraday momentum",
    },
}


if __name__ == "__main__":
    # Quick test with random data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="B")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Generate random price data
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, (len(dates), len(symbols))), axis=0)),
        index=dates,
        columns=symbols,
    )
    open_ = close.shift(1).fillna(close)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, close.shape)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, close.shape)))
    volume = pd.DataFrame(
        np.random.uniform(1e6, 1e7, (len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )
    vwap = (high + low + close) / 3
    returns = close.pct_change()

    print("Testing Alpha101 implementations...")
    print("=" * 60)

    for name, info in ALPHA_FUNCTIONS.items():
        try:
            func = info["func"]
            inputs = info["inputs"]

            # Build kwargs
            kwargs = {}
            for inp in inputs:
                if inp == "close":
                    kwargs["close"] = close
                elif inp == "open":
                    kwargs["open_"] = open_
                elif inp == "high":
                    kwargs["high"] = high
                elif inp == "low":
                    kwargs["low"] = low
                elif inp == "volume":
                    kwargs["volume"] = volume
                elif inp == "vwap":
                    kwargs["vwap"] = vwap
                elif inp == "returns":
                    kwargs["returns"] = returns

            result = func(**kwargs)
            print(f"{name}: {info['description']}")
            print(f"  Alpha mean: {result.alpha.mean().mean():.4f}")
            print(f"  Signals sum (should be ~0): {result.signals.sum(axis=1).mean():.4f}")
            print()

        except Exception as e:
            print(f"{name}: ERROR - {e}")
            print()
