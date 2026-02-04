"""
Batch Alpha Discovery Script
Discovers multiple alphas in sequence.
"""

import json
from datetime import datetime
from vibequant.memory.strategy_memory import StrategyMemory, BacktestResult
from vibequant.data.alpaca_loader import AlpacaDataLoader
from vibequant.backtest.engine import BacktestEngine, BacktestConfig, execute_strategy_code
from vibequant.agents.base import check_passing_criteria

# Pre-defined hypotheses and strategies to test
ALPHA_CANDIDATES = [
    # 1. Mean Reversion - RSI
    {
        "hypothesis": {
            "hypothesis": "ETFs with RSI below 35 tend to revert upward over the next 5 days",
            "rationale": "Oversold conditions create buying opportunities as prices mean-revert",
            "category": "mean_reversion",
            "priority_score": 0.7
        },
        "strategy": {
            "strategy_name": "RSI Oversold Mean Reversion",
            "description": "Buy ETFs when RSI < 35, sell when RSI > 50 or after 5 days",
            "code": '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    rsi_period = 14
    rsi_oversold = 35
    rsi_exit = 50
    max_hold = 5
    max_positions = 5

    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    holding = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    for i in range(1, len(prices)):
        for col in prices.columns:
            if signals.iloc[i-1][col] > 0:
                if rsi.iloc[i][col] > rsi_exit or holding.iloc[i-1][col] >= max_hold:
                    signals.loc[prices.index[i], col] = 0
                    holding.loc[prices.index[i], col] = 0
                else:
                    signals.loc[prices.index[i], col] = 1.0
                    holding.loc[prices.index[i], col] = holding.iloc[i-1][col] + 1
            else:
                if rsi.iloc[i][col] < rsi_oversold:
                    signals.loc[prices.index[i], col] = 1.0
                    holding.loc[prices.index[i], col] = 1

    # Limit positions
    for i in range(len(signals)):
        active = signals.iloc[i] > 0
        if active.sum() > max_positions:
            rsi_vals = rsi.iloc[i][active].nsmallest(max_positions).index
            for col in signals.columns:
                if col not in rsi_vals:
                    signals.loc[prices.index[i], col] = 0

    row_sums = signals.sum(axis=1).replace(0, 1)
    signals = signals.div(row_sums, axis=0)
    return signals.fillna(0)
'''
        }
    },

    # 2. Low Volatility Anomaly
    {
        "hypothesis": {
            "hypothesis": "Low volatility ETFs outperform high volatility ETFs on risk-adjusted basis",
            "rationale": "The low-vol anomaly: investors overpay for lottery-like high-vol assets",
            "category": "volatility",
            "priority_score": 0.8
        },
        "strategy": {
            "strategy_name": "Low Volatility Factor",
            "description": "Hold the 5 lowest volatility ETFs, rebalance monthly",
            "code": '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    vol_lookback = 60
    top_n = 5
    rebal_period = 21  # Monthly

    returns = prices.pct_change()
    volatility = returns.rolling(window=vol_lookback, min_periods=vol_lookback).std()

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(vol_lookback, len(prices), rebal_period):
        end_idx = min(i + rebal_period, len(prices))
        today_vol = volatility.iloc[i]
        valid_vol = today_vol.dropna()

        if len(valid_vol) >= top_n:
            lowest_vol = valid_vol.nsmallest(top_n).index.tolist()
            weight = 1.0 / len(lowest_vol)
            for j in range(i, end_idx):
                for sym in lowest_vol:
                    signals.loc[prices.index[j], sym] = weight

    return signals.fillna(0)
'''
        }
    },

    # 3. Trend Following - MA Crossover
    {
        "hypothesis": {
            "hypothesis": "ETFs above their 50-day MA with 50-day MA above 200-day MA tend to continue trending",
            "rationale": "Trend following captures momentum in established trends",
            "category": "momentum",
            "priority_score": 0.7
        },
        "strategy": {
            "strategy_name": "Golden Cross Trend Following",
            "description": "Long ETFs with price > 50MA > 200MA",
            "code": '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    ma_fast = 50
    ma_slow = 200
    max_positions = 5

    ma50 = prices.rolling(window=ma_fast, min_periods=ma_fast).mean()
    ma200 = prices.rolling(window=ma_slow, min_periods=ma_slow).mean()

    # Trend conditions
    price_above_ma50 = prices > ma50
    ma50_above_ma200 = ma50 > ma200
    trend_up = price_above_ma50 & ma50_above_ma200

    signals = trend_up.astype(float)

    # Limit to max positions, prefer strongest trends
    trend_strength = (prices - ma200) / ma200  # Distance from 200MA

    for i in range(len(signals)):
        active = signals.iloc[i] > 0
        if active.sum() > max_positions:
            strength = trend_strength.iloc[i][active].nlargest(max_positions).index
            for col in signals.columns:
                if col not in strength:
                    signals.loc[prices.index[i], col] = 0

    row_sums = signals.sum(axis=1).replace(0, 1)
    signals = signals.div(row_sums, axis=0)
    return signals.fillna(0)
'''
        }
    },

    # 4. Contrarian - Buy Losers
    {
        "hypothesis": {
            "hypothesis": "ETFs with worst 1-month performance tend to rebound over next month",
            "rationale": "Short-term reversals due to overreaction and mean reversion",
            "category": "mean_reversion",
            "priority_score": 0.6
        },
        "strategy": {
            "strategy_name": "Monthly Loser Reversal",
            "description": "Buy bottom 3 performers from past month, hold for 1 month",
            "code": '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    lookback = 21  # 1 month
    hold_period = 21
    bottom_n = 3

    monthly_return = prices.pct_change(lookback)

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(lookback, len(prices), hold_period):
        end_idx = min(i + hold_period, len(prices))
        returns = monthly_return.iloc[i]
        valid = returns.dropna()

        if len(valid) >= bottom_n:
            losers = valid.nsmallest(bottom_n).index.tolist()
            weight = 1.0 / len(losers)
            for j in range(i, end_idx):
                for sym in losers:
                    signals.loc[prices.index[j], sym] = weight

    return signals.fillna(0)
'''
        }
    },

    # 5. Breakout - 52-Week High
    {
        "hypothesis": {
            "hypothesis": "ETFs making new 52-week highs tend to continue outperforming",
            "rationale": "New highs indicate strong momentum and attract trend followers",
            "category": "momentum",
            "priority_score": 0.75
        },
        "strategy": {
            "strategy_name": "52-Week High Breakout",
            "description": "Buy ETFs within 2% of 52-week high",
            "code": '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    high_lookback = 252  # 52 weeks
    threshold = 0.98  # Within 2% of high
    max_positions = 5

    rolling_high = prices.rolling(window=high_lookback, min_periods=high_lookback).max()
    near_high = prices >= (rolling_high * threshold)

    signals = near_high.astype(float)

    # Prefer those closest to new high
    distance_from_high = prices / rolling_high

    for i in range(len(signals)):
        active = signals.iloc[i] > 0
        if active.sum() > max_positions:
            closest = distance_from_high.iloc[i][active].nlargest(max_positions).index
            for col in signals.columns:
                if col not in closest:
                    signals.loc[prices.index[i], col] = 0

    row_sums = signals.sum(axis=1).replace(0, 1)
    signals = signals.div(row_sums, axis=0)
    return signals.fillna(0)
'''
        }
    },

    # 6. Relative Strength vs Market
    {
        "hypothesis": {
            "hypothesis": "Stocks with highest 3-month relative strength vs market median continue outperforming",
            "rationale": "Relative momentum persists as capital flows into winning stocks",
            "category": "momentum",
            "priority_score": 0.7
        },
        "strategy": {
            "strategy_name": "Relative Strength Leaders",
            "description": "Hold top 10 stocks by 63-day relative strength vs market median",
            "code": '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    lookback = 63  # 3 months
    top_n = 10
    rebal_period = 21  # Monthly rebalance

    returns = prices.pct_change(lookback)
    # Use median return as benchmark
    median_return = returns.median(axis=1)

    # Relative strength = asset return - median return
    relative_strength = returns.sub(median_return, axis=0)

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(lookback, len(prices), rebal_period):
        end_idx = min(i + rebal_period, len(prices))
        rs = relative_strength.iloc[i]
        valid = rs.dropna()

        if len(valid) >= top_n:
            top_stocks = valid.nlargest(top_n).index.tolist()
            weight = 1.0 / len(top_stocks)
            for j in range(i, end_idx):
                for sym in top_stocks:
                    signals.loc[prices.index[j], sym] = weight

    return signals.fillna(0)
'''
        }
    },

    # 7. Volatility Breakout
    {
        "hypothesis": {
            "hypothesis": "ETFs breaking out of low volatility regimes with positive momentum outperform",
            "rationale": "Volatility contraction followed by expansion signals directional moves",
            "category": "volatility",
            "priority_score": 0.65
        },
        "strategy": {
            "strategy_name": "Volatility Breakout",
            "description": "Buy when current vol < 50% of 60-day vol average AND momentum positive",
            "code": '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    vol_lookback = 20
    vol_avg_lookback = 60
    momentum_lookback = 20
    max_positions = 5

    returns = prices.pct_change()
    current_vol = returns.rolling(window=vol_lookback).std()
    avg_vol = current_vol.rolling(window=vol_avg_lookback).mean()

    vol_contraction = current_vol < (avg_vol * 0.5)
    momentum = prices.pct_change(momentum_lookback)
    positive_momentum = momentum > 0

    entry_signal = vol_contraction & positive_momentum
    signals = entry_signal.astype(float)

    # Limit positions
    for i in range(len(signals)):
        active = signals.iloc[i] > 0
        if active.sum() > max_positions:
            mom_vals = momentum.iloc[i][active].nlargest(max_positions).index
            for col in signals.columns:
                if col not in mom_vals:
                    signals.loc[prices.index[i], col] = 0

    row_sums = signals.sum(axis=1).replace(0, 1)
    signals = signals.div(row_sums, axis=0)
    return signals.fillna(0)
'''
        }
    },

    # 8. End of Month Effect
    {
        "hypothesis": {
            "hypothesis": "Stocks tend to have positive returns in the last 3 days and first 3 days of each month",
            "rationale": "Window dressing by institutions and pension fund flows create turn-of-month effect",
            "category": "seasonal",
            "priority_score": 0.6
        },
        "strategy": {
            "strategy_name": "Turn of Month Effect",
            "description": "Long top momentum stocks during last 3 and first 3 trading days of month",
            "code": '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    top_n = 10
    momentum_period = 20

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    momentum = prices.pct_change(momentum_period)

    dates = prices.index.to_series()
    day_of_month = dates.dt.day

    for i in range(momentum_period + 1, len(prices)):
        dom = day_of_month.iloc[i]
        is_turn_of_month = (dom <= 3) or (dom >= 26)

        if is_turn_of_month:
            mom = momentum.iloc[i].dropna()
            if len(mom) >= top_n:
                top_stocks = mom.nlargest(top_n).index.tolist()
                weight = 1.0 / len(top_stocks)
                for sym in top_stocks:
                    signals.loc[prices.index[i], sym] = weight

    return signals.fillna(0)
'''
        }
    },

    # 9. Quality - Stable Performers
    {
        "hypothesis": {
            "hypothesis": "ETFs with highest Sharpe ratio over past 6 months continue to outperform",
            "rationale": "Consistent risk-adjusted performers indicate quality and stability",
            "category": "factor",
            "priority_score": 0.7
        },
        "strategy": {
            "strategy_name": "Rolling Sharpe Quality",
            "description": "Hold top 3 ETFs by 126-day Sharpe ratio, rebalance monthly",
            "code": '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    sharpe_lookback = 126  # 6 months
    top_n = 3
    rebal_period = 21

    returns = prices.pct_change()
    rolling_mean = returns.rolling(window=sharpe_lookback).mean()
    rolling_std = returns.rolling(window=sharpe_lookback).std()
    rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(sharpe_lookback, len(prices), rebal_period):
        end_idx = min(i + rebal_period, len(prices))
        sharpe = rolling_sharpe.iloc[i]
        valid = sharpe.dropna()

        if len(valid) >= top_n:
            top_quality = valid.nlargest(top_n).index.tolist()
            weight = 1.0 / len(top_quality)
            for j in range(i, end_idx):
                for sym in top_quality:
                    signals.loc[prices.index[j], sym] = weight

    return signals.fillna(0)
'''
        }
    },

    # 10. Pairs Mean Reversion - Bond/Equity
    {
        "hypothesis": {
            "hypothesis": "When TLT/SPY ratio deviates significantly from mean, it reverts over 20 days",
            "rationale": "Bond-equity relationship mean reverts as risk sentiment normalizes",
            "category": "pairs",
            "priority_score": 0.65
        },
        "strategy": {
            "strategy_name": "Bond-Equity Mean Reversion",
            "description": "Trade TLT vs SPY based on z-score of their ratio",
            "code": '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    if "TLT" not in prices.columns or "SPY" not in prices.columns:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    lookback = 60
    zscore_entry = 2.0
    zscore_exit = 0.5

    ratio = prices["TLT"] / prices["SPY"]
    ratio_mean = ratio.rolling(window=lookback).mean()
    ratio_std = ratio.rolling(window=lookback).std()
    zscore = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    position = 0  # 1 = long TLT/short SPY, -1 = short TLT/long SPY

    for i in range(lookback, len(prices)):
        z = zscore.iloc[i]

        if pd.isna(z):
            continue

        if position == 0:
            if z > zscore_entry:  # TLT expensive, go long SPY
                position = -1
            elif z < -zscore_entry:  # TLT cheap, go long TLT
                position = 1
        elif position == 1:
            if z > -zscore_exit:
                position = 0
        elif position == -1:
            if z < zscore_exit:
                position = 0

        if position == 1:
            signals.loc[prices.index[i], "TLT"] = 1.0
        elif position == -1:
            signals.loc[prices.index[i], "SPY"] = 1.0

    return signals.fillna(0)
'''
        }
    },
]


def run_discovery():
    """Run batch alpha discovery."""
    print("=" * 70)
    print("VIBEQUANT BATCH ALPHA DISCOVERY")
    print("=" * 70)
    print(f"Candidates to test: {len(ALPHA_CANDIDATES)}")
    print(f"Started: {datetime.now()}\n")

    # Initialize components
    memory = StrategyMemory("./memory")
    data_loader = AlpacaDataLoader()
    backtest_engine = BacktestEngine(BacktestConfig())

    # Load data once - use liquid universe (stocks + ETFs)
    print("Building liquid universe...")
    symbols = data_loader.get_liquid_universe(
        min_price=5.0,
        max_price=10000.0,
        min_volume=500_000,
        max_symbols=200,
    )
    print(f"Universe: {len(symbols)} liquid tickers")

    print("\nLoading market data...")
    data = data_loader.get_bars(
        symbols=symbols,
        timeframe="1Day",
        start=datetime(2016, 1, 1),
    )
    prices = data["close"].unstack(level=0)
    print(f"Data shape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}\n")

    # Benchmark
    benchmark_prices = prices.get("SPY")

    results_summary = []

    for idx, candidate in enumerate(ALPHA_CANDIDATES, 1):
        print(f"\n{'='*70}")
        print(f"ALPHA {idx}/{len(ALPHA_CANDIDATES)}: {candidate['strategy']['strategy_name']}")
        print(f"{'='*70}")
        print(f"Hypothesis: {candidate['hypothesis']['hypothesis'][:80]}...")
        print(f"Category: {candidate['hypothesis']['category']}")

        # Save hypothesis
        hyp_record = memory.add_hypothesis(
            hypothesis=candidate['hypothesis']['hypothesis'],
            rationale=candidate['hypothesis']['rationale'],
            category=candidate['hypothesis']['category'],
            priority_score=candidate['hypothesis']['priority_score'],
        )

        # Save strategy
        code = candidate['strategy']['code'].strip()
        strategy_record = memory.add_strategy(
            name=candidate['strategy']['strategy_name'],
            hypothesis_id=hyp_record.id,
            description=candidate['strategy']['description'],
            category=candidate['hypothesis']['category'],
            code=code,
        )

        # Run backtest
        try:
            print("Running backtest...")
            signals = execute_strategy_code(code, prices)
            results = backtest_engine.run(prices, signals, benchmark_prices)

            metrics = {
                "total_return": results.total_return,
                "annual_return": results.annual_return,
                "sharpe_ratio": results.sharpe_ratio,
                "sortino_ratio": results.sortino_ratio,
                "max_drawdown": results.max_drawdown,
                "num_trades": results.num_trades,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor,
                "alpha": results.alpha,
                "beta": results.beta,
                "correlation": results.correlation,
            }

            passed, failures = check_passing_criteria(metrics)

            print(f"\nResults:")
            print(f"  Total Return:  {metrics['total_return']:>8.2%}")
            print(f"  Annual Return: {metrics['annual_return']:>8.2%}")
            print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:>8.2f}")
            print(f"  Max Drawdown:  {metrics['max_drawdown']:>8.2%}")
            print(f"  Trades:        {metrics['num_trades']:>8d}")
            print(f"  Win Rate:      {metrics['win_rate']:>8.2%}")
            print(f"  Profit Factor: {metrics['profit_factor']:>8.2f}")
            print(f"  Alpha:         {metrics['alpha']:>8.4f}")
            print(f"  Correlation:   {metrics['correlation']:>8.2f}")
            print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
            if failures:
                for f in failures:
                    print(f"    - {f}")

            # Update memory
            bt_result = BacktestResult(
                total_return=metrics["total_return"],
                annual_return=metrics["annual_return"],
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                win_rate=metrics["win_rate"],
                num_trades=metrics["num_trades"],
                start_date=str(results.start_date),
                end_date=str(results.end_date),
                alpha=metrics["alpha"],
                beta=metrics["beta"],
                correlation_to_spy=metrics["correlation"],
                sortino_ratio=metrics["sortino_ratio"],
                profit_factor=metrics["profit_factor"],
            )

            memory.update_strategy_backtest(
                strategy_id=strategy_record.id,
                backtest_results=bt_result,
                passed=passed,
                failure_reasons=failures if not passed else [],
            )

            results_summary.append({
                "name": candidate['strategy']['strategy_name'],
                "category": candidate['hypothesis']['category'],
                "passed": passed,
                "sharpe": metrics['sharpe_ratio'],
                "return": metrics['annual_return'],
                "drawdown": metrics['max_drawdown'],
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results_summary.append({
                "name": candidate['strategy']['strategy_name'],
                "category": candidate['hypothesis']['category'],
                "passed": False,
                "sharpe": 0,
                "return": 0,
                "drawdown": 0,
                "error": str(e),
            })

    # Final Summary
    print("\n" + "=" * 70)
    print("DISCOVERY COMPLETE - SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for r in results_summary if r['passed'])
    print(f"\nPassed: {passed_count}/{len(results_summary)}")

    print(f"\n{'Strategy':<35} {'Category':<15} {'Sharpe':>8} {'Return':>8} {'DD':>8} {'Pass':<6}")
    print("-" * 85)
    for r in sorted(results_summary, key=lambda x: x['sharpe'], reverse=True):
        print(f"{r['name']:<35} {r['category']:<15} {r['sharpe']:>8.2f} {r['return']:>7.1%} {r['drawdown']:>7.1%} {'YES' if r['passed'] else 'NO':<6}")

    print("\n" + memory.get_summary_report())

    return results_summary


if __name__ == "__main__":
    run_discovery()
