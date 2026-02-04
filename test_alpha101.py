"""
Test WorldQuant Alpha101 Implementations

Tests selected alphas from the 101 Formulaic Alphas paper
using the VibeQuant backtesting framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from vibequant.data import AlpacaDataLoader
from vibequant.backtest import BacktestEngine, BacktestConfig
from vibequant.alpha101 import (
    ALPHA_FUNCTIONS,
    alpha_to_long_only_signals,
    alpha001, alpha002, alpha006, alpha012, alpha033,
    alpha041, alpha042, alpha053, alpha054, alpha101,
)


def load_ohlcv_data(symbols: list, start_date: datetime = datetime(2018, 1, 1)):
    """Load OHLCV data for testing."""
    loader = AlpacaDataLoader()

    print(f"Loading OHLCV data for {len(symbols)} symbols...")
    data = loader.get_bars(
        symbols=symbols,
        timeframe="1Day",
        start=start_date,
    )

    # Pivot each field
    close = data["close"].unstack(level=0)
    open_ = data["open"].unstack(level=0)
    high = data["high"].unstack(level=0)
    low = data["low"].unstack(level=0)
    volume = data["volume"].unstack(level=0)
    vwap = data["vwap"].unstack(level=0) if "vwap" in data.columns else (high + low + close) / 3

    print(f"Loaded {close.shape[0]} days, {close.shape[1]} symbols")
    print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "vwap": vwap,
        "returns": close.pct_change(),
    }


def test_alpha(
    alpha_name: str,
    alpha_func,
    data: dict,
    top_n: int = 10,
    hold_period: int = 5,
) -> dict:
    """Test a single alpha and return results."""
    # Get inputs for this alpha
    info = ALPHA_FUNCTIONS.get(alpha_name, {})
    inputs = info.get("inputs", [])

    # Build kwargs
    kwargs = {}
    for inp in inputs:
        if inp == "close":
            kwargs["close"] = data["close"]
        elif inp == "open":
            kwargs["open_"] = data["open"]
        elif inp == "high":
            kwargs["high"] = data["high"]
        elif inp == "low":
            kwargs["low"] = data["low"]
        elif inp == "volume":
            kwargs["volume"] = data["volume"]
        elif inp == "vwap":
            kwargs["vwap"] = data["vwap"]
        elif inp == "returns":
            kwargs["returns"] = data["returns"]

    # Calculate alpha
    result = alpha_func(**kwargs)

    # Convert to long-only signals
    signals = alpha_to_long_only_signals(
        result.alpha, top_n=top_n, hold_period=hold_period
    )

    # Run backtest
    engine = BacktestEngine(BacktestConfig())
    benchmark = data["close"].get("SPY")
    backtest = engine.run(data["close"], signals, benchmark)

    return {
        "name": alpha_name,
        "description": result.description,
        "sharpe": backtest.sharpe_ratio,
        "annual_return": backtest.annual_return,
        "max_drawdown": backtest.max_drawdown,
        "win_rate": backtest.win_rate,
        "num_trades": backtest.num_trades,
        "profit_factor": backtest.profit_factor,
        "correlation": backtest.correlation,
    }


def run_alpha_tests():
    """Run tests on all implemented alphas."""
    print("=" * 70)
    print("WORLDQUANT ALPHA101 BACKTEST")
    print("=" * 70)

    # Get liquid universe
    loader = AlpacaDataLoader()
    symbols = loader.get_liquid_universe(
        min_volume=1_000_000,
        max_symbols=200,
    )

    # Load OHLCV data
    data = load_ohlcv_data(symbols, start_date=datetime(2018, 1, 1))

    # Test configurations
    configs = [
        {"top_n": 10, "hold_period": 5},   # Short-term
        {"top_n": 20, "hold_period": 10},  # Medium-term
    ]

    all_results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: top_n={config['top_n']}, hold_period={config['hold_period']}")
        print("=" * 70)

        for alpha_name, info in ALPHA_FUNCTIONS.items():
            print(f"\n[Testing] {alpha_name}: {info['description']}")

            try:
                result = test_alpha(
                    alpha_name=alpha_name,
                    alpha_func=info["func"],
                    data=data,
                    **config,
                )
                result["config"] = f"top{config['top_n']}_hold{config['hold_period']}"
                all_results.append(result)

                status = "PASS" if result["sharpe"] >= 0.5 else "FAIL"
                print(f"    Sharpe: {result['sharpe']:.2f} | "
                      f"Return: {result['annual_return']:.1%} | "
                      f"DD: {result['max_drawdown']:.1%} | "
                      f"Trades: {result['num_trades']} | {status}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(all_results)
    if not df.empty:
        # Best by Sharpe
        print("\nTOP 5 BY SHARPE:")
        top5 = df.nlargest(5, "sharpe")
        for _, row in top5.iterrows():
            print(f"  {row['name']} ({row['config']}): "
                  f"Sharpe={row['sharpe']:.2f}, Return={row['annual_return']:.1%}")

        # Average by alpha
        print("\nAVERAGE SHARPE BY ALPHA:")
        by_alpha = df.groupby("name")["sharpe"].mean().sort_values(ascending=False)
        for alpha, sharpe in by_alpha.items():
            desc = ALPHA_FUNCTIONS[alpha]["description"]
            status = "PASS" if sharpe >= 0.5 else "FAIL"
            print(f"  {alpha}: {sharpe:.2f} ({desc}) [{status}]")

        # Save results
        df.to_csv("memory/alpha101_results.csv", index=False)
        print(f"\nResults saved to memory/alpha101_results.csv")

    return df


if __name__ == "__main__":
    results = run_alpha_tests()
