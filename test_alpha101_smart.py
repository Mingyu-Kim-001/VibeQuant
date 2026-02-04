"""
Test best Alpha101 with Smart Variations on Survivorship-Free Universe

Tests Alpha033 and Alpha042 (the top performers) with:
1. Survivorship-bias-free S&P 500 universe
2. Smart variations based on strategy characteristics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from vibequant.data import AlpacaDataLoader, SP500SurvivorshipFreeLoader
from vibequant.backtest import BacktestEngine, BacktestConfig
from vibequant import test_smart_variations
from vibequant.alpha101 import (
    alpha033, alpha042, alpha053, alpha054, alpha101,
    alpha_to_long_only_signals,
    rank, delta, log, ts_rank, ts_std, sign,
    correlation, ts_argmax, power,
)


def create_alpha033_strategy(top_n: int = 10, hold_period: int = 5) -> str:
    """
    Create Alpha033 strategy code.

    Alpha#33: rank((-1 + (open / close)))
    Stocks that gapped down (open < close yesterday) rank higher.
    This is a mean reversion alpha.
    """
    code = f'''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Alpha033: Gap rank mean reversion strategy."""
    close = prices
    # For daily data, open today ≈ close yesterday (approximation)
    open_ = prices.shift(1).fillna(prices)

    # Alpha: rank of negative gap
    gap = -1 + (open_ / close)
    alpha = gap.rank(axis=1, pct=True)

    # Convert to long-only: buy top N gapped-down stocks
    top_n = {top_n}
    hold_period = {hold_period}

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(1, len(alpha), hold_period):
        end_idx = min(i + hold_period, len(alpha))
        alpha_row = alpha.iloc[i].dropna()

        if len(alpha_row) >= top_n:
            # Higher alpha = gapped down more = buy
            top_stocks = alpha_row.nlargest(top_n).index.tolist()
            weight = 1.0 / len(top_stocks)
            for j in range(i, end_idx):
                for sym in top_stocks:
                    if sym in signals.columns:
                        signals.loc[prices.index[j], sym] = weight

    return signals.fillna(0)
'''
    return code


def create_alpha042_strategy(top_n: int = 10, hold_period: int = 5) -> str:
    """
    Create Alpha042 strategy code.

    Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
    Ratio of VWAP deviation ranks.
    """
    code = f'''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Alpha042: VWAP-close ratio strategy."""
    import numpy as np

    close = prices
    # Approximate VWAP as rolling average (for daily close-only data)
    vwap = prices.rolling(5).mean().fillna(prices)

    # Alpha: ratio of ranks
    rank_diff = (vwap - close).rank(axis=1, pct=True)
    rank_sum = (vwap + close).rank(axis=1, pct=True).replace(0, np.nan)
    alpha = rank_diff / rank_sum

    # Convert to long-only
    top_n = {top_n}
    hold_period = {hold_period}

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(5, len(alpha), hold_period):
        end_idx = min(i + hold_period, len(alpha))
        alpha_row = alpha.iloc[i].dropna()

        if len(alpha_row) >= top_n:
            top_stocks = alpha_row.nlargest(top_n).index.tolist()
            weight = 1.0 / len(top_stocks)
            for j in range(i, end_idx):
                for sym in top_stocks:
                    if sym in signals.columns:
                        signals.loc[prices.index[j], sym] = weight

    return signals.fillna(0)
'''
    return code


def create_alpha053_strategy(top_n: int = 10, hold_period: int = 5) -> str:
    """
    Create Alpha053 strategy code.

    Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    Change in Williams %R-like indicator.
    """
    code = f'''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Alpha053: Williams %R momentum strategy."""
    import numpy as np

    close = prices
    # Approximate high/low from close (for daily close-only data)
    high = prices.rolling(1).max()
    low = prices.rolling(1).min()

    # Calculate Williams %R like indicator
    denom = (close - low).replace(0, np.nan)
    inner = ((close - low) - (high - close)) / denom

    # 9-day delta
    alpha = -1 * inner.diff(9)

    # Convert to long-only
    top_n = {top_n}
    hold_period = {hold_period}

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(10, len(alpha), hold_period):
        end_idx = min(i + hold_period, len(alpha))
        alpha_row = alpha.iloc[i].dropna()
        # Remove extreme values
        alpha_row = alpha_row[alpha_row.abs() < 10]

        if len(alpha_row) >= top_n:
            top_stocks = alpha_row.nlargest(top_n).index.tolist()
            weight = 1.0 / len(top_stocks)
            for j in range(i, end_idx):
                for sym in top_stocks:
                    if sym in signals.columns:
                        signals.loc[prices.index[j], sym] = weight

    return signals.fillna(0)
'''
    return code


def main():
    print("=" * 70)
    print("ALPHA101 SMART VARIATION TESTING")
    print("=" * 70)

    # Test configurations
    alphas_to_test = [
        {
            "name": "Alpha033 Gap Reversal",
            "code_func": create_alpha033_strategy,
            "category": "mean_reversion",
            "top_n": 10,
            "hold_period": 5,
        },
        {
            "name": "Alpha042 VWAP Ratio",
            "code_func": create_alpha042_strategy,
            "category": "factor",
            "top_n": 10,
            "hold_period": 5,
        },
        {
            "name": "Alpha053 Williams R",
            "code_func": create_alpha053_strategy,
            "category": "momentum",
            "top_n": 10,
            "hold_period": 5,
        },
    ]

    all_results = []

    for alpha_config in alphas_to_test:
        print(f"\n{'='*70}")
        print(f"Testing: {alpha_config['name']}")
        print(f"Category: {alpha_config['category']}")
        print("=" * 70)

        code = alpha_config["code_func"](
            top_n=alpha_config["top_n"],
            hold_period=alpha_config["hold_period"],
        )

        try:
            results, chars, variations = test_smart_variations(
                strategy_code=code,
                strategy_name=alpha_config["name"],
                category=alpha_config["category"],
                universes=["sp500_sf_200", "liquid_500"],
                max_variations=6,
            )

            if not results.empty:
                results["alpha"] = alpha_config["name"]
                all_results.append(results)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        print("\n" + "=" * 70)
        print("FINAL SUMMARY - ALL ALPHAS")
        print("=" * 70)

        # Best by Sharpe (survivorship-free only)
        sf_results = combined[combined["survivorship_free"] == True]
        if not sf_results.empty:
            print("\nTOP 5 SURVIVORSHIP-FREE CONFIGURATIONS:")
            top5 = sf_results.nlargest(5, "sharpe_ratio")
            for _, row in top5.iterrows():
                print(f"  {row['alpha']} + {row['variation']}: "
                      f"Sharpe={row['sharpe_ratio']:.2f}, Return={row['annual_return']:.1%}")

        # By alpha (survivorship-free)
        print("\nAVERAGE SHARPE BY ALPHA (Survivorship-Free):")
        if not sf_results.empty:
            by_alpha = sf_results.groupby("alpha")["sharpe_ratio"].mean().sort_values(ascending=False)
            for alpha, sharpe in by_alpha.items():
                status = "PASS" if sharpe >= 0.5 else "FAIL"
                print(f"  {alpha}: {sharpe:.2f} [{status}]")

        # Save
        combined.to_csv("memory/alpha101_smart_results.csv", index=False)
        print(f"\nResults saved to memory/alpha101_smart_results.csv")


if __name__ == "__main__":
    main()
