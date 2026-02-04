"""
VibeQuant Fully Automated Workflow
No LLM or user input required - uses rule-based generation and evaluation.
"""

import json
import random
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import itertools

from .memory.strategy_memory import StrategyMemory, BacktestResult
from .data.alpaca_loader import AlpacaDataLoader
from .data.sp500_loader import SP500SurvivorshipFreeLoader
from .backtest.engine import BacktestEngine, BacktestConfig, execute_strategy_code
from .agents.base import check_passing_criteria, PASSING_CRITERIA


@dataclass
class HypothesisTemplate:
    """Template for generating hypotheses."""
    name: str
    category: str
    hypothesis: str
    code_template: str
    parameters: Dict[str, List]  # Parameter name -> list of values to try


# Strategy templates with parameter variations
STRATEGY_TEMPLATES = [
    # =========================================================================
    # MOMENTUM STRATEGIES
    # =========================================================================
    HypothesisTemplate(
        name="Price Momentum",
        category="momentum",
        hypothesis="Stocks with highest {lookback}-day momentum continue outperforming over the next {hold_period} days",
        code_template='''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    lookback = {lookback}
    hold_period = {hold_period}
    top_n = {top_n}

    momentum = prices.pct_change(lookback)
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(lookback, len(prices), hold_period):
        end_idx = min(i + hold_period, len(prices))
        mom = momentum.iloc[i].dropna()
        if len(mom) >= top_n:
            top = mom.nlargest(top_n).index.tolist()
            weight = 1.0 / len(top)
            for j in range(i, end_idx):
                for sym in top:
                    signals.loc[prices.index[j], sym] = weight
    return signals.fillna(0)
''',
        parameters={
            "lookback": [5, 10, 20, 60, 120, 252],
            "hold_period": [5, 10, 21, 63],
            "top_n": [5, 10, 20],
        }
    ),

    HypothesisTemplate(
        name="Momentum with Trend Filter",
        category="momentum",
        hypothesis="Stocks with {lookback}-day momentum above MA{ma_period} outperform",
        code_template='''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    lookback = {lookback}
    ma_period = {ma_period}
    top_n = {top_n}
    hold_period = {hold_period}

    momentum = prices.pct_change(lookback)
    ma = prices.rolling(window=ma_period, min_periods=ma_period).mean()
    above_ma = prices > ma

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(max(lookback, ma_period), len(prices), hold_period):
        end_idx = min(i + hold_period, len(prices))
        mom = momentum.iloc[i]
        trend = above_ma.iloc[i]

        # Only consider stocks above MA with positive momentum
        valid = mom[(trend == True) & (mom > 0)].dropna()

        if len(valid) >= top_n:
            top = valid.nlargest(top_n).index.tolist()
            weight = 1.0 / len(top)
            for j in range(i, end_idx):
                for sym in top:
                    signals.loc[prices.index[j], sym] = weight
    return signals.fillna(0)
''',
        parameters={
            "lookback": [20, 60, 120],
            "ma_period": [50, 100, 200],
            "hold_period": [21, 63],
            "top_n": [10, 20],
        }
    ),

    # =========================================================================
    # MEAN REVERSION STRATEGIES
    # =========================================================================
    HypothesisTemplate(
        name="RSI Mean Reversion",
        category="mean_reversion",
        hypothesis="Stocks with RSI below {rsi_entry} revert upward within {max_hold} days",
        code_template='''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    rsi_period = {rsi_period}
    rsi_entry = {rsi_entry}
    rsi_exit = {rsi_exit}
    max_hold = {max_hold}
    max_positions = {max_positions}

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
            prev_sig = signals.iloc[i-1][col] if i > 0 else 0
            prev_hold = holding.iloc[i-1][col] if i > 0 else 0

            if prev_sig > 0:
                if rsi.iloc[i][col] > rsi_exit or prev_hold >= max_hold:
                    signals.loc[prices.index[i], col] = 0
                    holding.loc[prices.index[i], col] = 0
                else:
                    signals.loc[prices.index[i], col] = 1.0
                    holding.loc[prices.index[i], col] = prev_hold + 1
            else:
                if rsi.iloc[i][col] < rsi_entry:
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
''',
        parameters={
            "rsi_period": [7, 14, 21],
            "rsi_entry": [20, 25, 30, 35],
            "rsi_exit": [50, 60, 70],
            "max_hold": [5, 10, 20],
            "max_positions": [5, 10, 20],
        }
    ),

    HypothesisTemplate(
        name="Short-term Reversal",
        category="mean_reversion",
        hypothesis="Stocks with worst {lookback}-day returns rebound over next {hold_period} days",
        code_template='''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    lookback = {lookback}
    hold_period = {hold_period}
    bottom_n = {bottom_n}

    returns = prices.pct_change(lookback)
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(lookback, len(prices), hold_period):
        end_idx = min(i + hold_period, len(prices))
        ret = returns.iloc[i].dropna()

        if len(ret) >= bottom_n:
            losers = ret.nsmallest(bottom_n).index.tolist()
            weight = 1.0 / len(losers)
            for j in range(i, end_idx):
                for sym in losers:
                    signals.loc[prices.index[j], sym] = weight
    return signals.fillna(0)
''',
        parameters={
            "lookback": [5, 10, 21],
            "hold_period": [5, 10, 21],
            "bottom_n": [5, 10, 20],
        }
    ),

    # =========================================================================
    # VOLATILITY STRATEGIES
    # =========================================================================
    HypothesisTemplate(
        name="Low Volatility",
        category="volatility",
        hypothesis="Stocks with lowest {vol_lookback}-day volatility outperform on risk-adjusted basis",
        code_template='''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    vol_lookback = {vol_lookback}
    top_n = {top_n}
    hold_period = {hold_period}

    returns = prices.pct_change()
    volatility = returns.rolling(window=vol_lookback, min_periods=vol_lookback).std()

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(vol_lookback, len(prices), hold_period):
        end_idx = min(i + hold_period, len(prices))
        vol = volatility.iloc[i].dropna()

        if len(vol) >= top_n:
            low_vol = vol.nsmallest(top_n).index.tolist()
            weight = 1.0 / len(low_vol)
            for j in range(i, end_idx):
                for sym in low_vol:
                    signals.loc[prices.index[j], sym] = weight
    return signals.fillna(0)
''',
        parameters={
            "vol_lookback": [20, 60, 120],
            "hold_period": [21, 63],
            "top_n": [10, 20, 30],
        }
    ),

    # =========================================================================
    # QUALITY/FACTOR STRATEGIES
    # =========================================================================
    HypothesisTemplate(
        name="Sharpe Quality",
        category="factor",
        hypothesis="Stocks with highest {sharpe_lookback}-day Sharpe ratio continue outperforming",
        code_template='''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    sharpe_lookback = {sharpe_lookback}
    top_n = {top_n}
    hold_period = {hold_period}

    returns = prices.pct_change()
    rolling_mean = returns.rolling(window=sharpe_lookback).mean()
    rolling_std = returns.rolling(window=sharpe_lookback).std()
    rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(sharpe_lookback, len(prices), hold_period):
        end_idx = min(i + hold_period, len(prices))
        sharpe = rolling_sharpe.iloc[i].dropna()

        if len(sharpe) >= top_n:
            top_quality = sharpe.nlargest(top_n).index.tolist()
            weight = 1.0 / len(top_quality)
            for j in range(i, end_idx):
                for sym in top_quality:
                    signals.loc[prices.index[j], sym] = weight
    return signals.fillna(0)
''',
        parameters={
            "sharpe_lookback": [63, 126, 252],
            "hold_period": [21, 63],
            "top_n": [10, 20],
        }
    ),

    # =========================================================================
    # BREAKOUT STRATEGIES
    # =========================================================================
    HypothesisTemplate(
        name="New High Breakout",
        category="momentum",
        hypothesis="Stocks within {threshold}% of {lookback}-day high continue trending",
        code_template='''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    lookback = {lookback}
    threshold = {threshold}
    max_positions = {max_positions}

    rolling_high = prices.rolling(window=lookback, min_periods=lookback).max()
    near_high = prices >= (rolling_high * threshold)
    distance = prices / rolling_high

    signals = near_high.astype(float)

    for i in range(len(signals)):
        active = signals.iloc[i] > 0
        if active.sum() > max_positions:
            closest = distance.iloc[i][active].nlargest(max_positions).index
            for col in signals.columns:
                if col not in closest:
                    signals.loc[prices.index[i], col] = 0

    row_sums = signals.sum(axis=1).replace(0, 1)
    signals = signals.div(row_sums, axis=0)
    return signals.fillna(0)
''',
        parameters={
            "lookback": [60, 126, 252],
            "threshold": [0.95, 0.98, 1.0],
            "max_positions": [10, 20],
        }
    ),

    # =========================================================================
    # COMBINED STRATEGIES
    # =========================================================================
    HypothesisTemplate(
        name="Momentum + Low Vol",
        category="factor",
        hypothesis="Stocks with top momentum AND low volatility provide best risk-adjusted returns",
        code_template='''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    mom_lookback = {mom_lookback}
    vol_lookback = {vol_lookback}
    hold_period = {hold_period}
    top_n = {top_n}

    momentum = prices.pct_change(mom_lookback)
    returns = prices.pct_change()
    volatility = returns.rolling(window=vol_lookback).std()

    # Rank both factors (higher is better for momentum, lower is better for vol)
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(max(mom_lookback, vol_lookback), len(prices), hold_period):
        end_idx = min(i + hold_period, len(prices))

        mom = momentum.iloc[i].dropna()
        vol = volatility.iloc[i].dropna()

        # Get common symbols
        common = mom.index.intersection(vol.index)
        if len(common) < top_n:
            continue

        # Rank momentum (higher = better rank)
        mom_rank = mom[common].rank(ascending=True)
        # Rank volatility (lower = better rank)
        vol_rank = vol[common].rank(ascending=False)
        # Combined score
        combined = mom_rank + vol_rank

        top = combined.nlargest(top_n).index.tolist()
        weight = 1.0 / len(top)
        for j in range(i, end_idx):
            for sym in top:
                signals.loc[prices.index[j], sym] = weight

    return signals.fillna(0)
''',
        parameters={
            "mom_lookback": [60, 120],
            "vol_lookback": [60, 120],
            "hold_period": [21, 63],
            "top_n": [10, 20],
        }
    ),
]


class AutomatedWorkflow:
    """
    Fully automated alpha discovery workflow.
    No LLM or user input required.
    """

    def __init__(
        self,
        memory_dir: str = "./memory",
        max_iterations: int = 50,
        min_successful: int = 3,
        universe_size: int = 200,
        survivorship_bias_free: bool = False,
        include_etfs: bool = True,
        min_avg_volume: int = 0,
    ):
        self.memory_dir = memory_dir
        self.max_iterations = max_iterations
        self.min_successful = min_successful
        self.universe_size = universe_size
        self.survivorship_bias_free = survivorship_bias_free
        self.include_etfs = include_etfs
        self.min_avg_volume = min_avg_volume

        # Initialize components
        self.memory = StrategyMemory(memory_dir)
        self.data_loader = AlpacaDataLoader()
        self.sp500_loader = None
        self.backtest_engine = BacktestEngine(BacktestConfig())

        # Track tested combinations
        self.tested_combinations = set()

        # Results
        self.successful_strategies = []
        self.failed_strategies = []

        # Data cache
        self.prices = None
        self.universe_mask = None  # For survivorship-bias-free
        self.benchmark = None

    def _load_data(self) -> bool:
        """Load market data."""
        if self.survivorship_bias_free:
            print("="*50)
            print("SURVIVORSHIP-BIAS-FREE MODE")
            print("="*50)
            print("Loading S&P 500 constituents with historical membership...")

            self.sp500_loader = SP500SurvivorshipFreeLoader()
            self.prices, self.universe_mask = self.sp500_loader.load_survivorship_free_data(
                start_date=datetime(2016, 1, 1),
                include_etfs=self.include_etfs,
                max_symbols=self.universe_size,
                min_avg_volume=self.min_avg_volume,
            )

            if self.prices.empty:
                print("ERROR: No data loaded")
                return False

            self.benchmark = self.prices.get("SPY")

            print(f"Data shape: {self.prices.shape}")
            print(f"Date range: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")

            # Store learning about survivorship bias free
            self.memory.add_learning(
                "technical_notes",
                "Using survivorship-bias-free S&P 500 universe with historical constituent membership dates"
            )
            return True

        else:
            print("Building liquid universe...")
            symbols = self.data_loader.get_liquid_universe(
                min_price=5.0,
                max_price=10000.0,
                min_volume=500_000,
                max_symbols=self.universe_size,
            )
            print(f"Universe: {len(symbols)} liquid tickers")

            print("Loading market data from 2016...")
            data = self.data_loader.get_bars(
                symbols=symbols,
                timeframe="1Day",
                start=datetime(2016, 1, 1),
            )

            if data.empty:
                print("ERROR: No data loaded")
                return False

            self.prices = data["close"].unstack(level=0)
            self.benchmark = self.prices.get("SPY")

            print(f"Data shape: {self.prices.shape}")
            print(f"Date range: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")
            return True

    def _generate_strategy(self) -> Optional[Tuple[str, str, str, Dict]]:
        """Generate a strategy from templates with random parameters."""
        # Shuffle templates to try different categories
        templates = list(STRATEGY_TEMPLATES)
        random.shuffle(templates)

        for template in templates:
            # Generate all parameter combinations
            param_names = list(template.parameters.keys())
            param_values = list(template.parameters.values())

            for combo in itertools.product(*param_values):
                params = dict(zip(param_names, combo))
                combo_key = f"{template.name}_{params}"

                if combo_key in self.tested_combinations:
                    continue

                self.tested_combinations.add(combo_key)

                # Generate hypothesis
                try:
                    hypothesis = template.hypothesis.format(**params)
                except:
                    hypothesis = template.hypothesis

                # Generate code
                code = template.code_template.format(**params)

                # Generate name
                name = f"{template.name} ({', '.join(f'{k}={v}' for k, v in params.items())})"

                return name, template.category, hypothesis, code, params

        return None

    def _run_backtest(self, code: str) -> Optional[Dict]:
        """Run backtest and return metrics."""
        try:
            signals = execute_strategy_code(code, self.prices)

            # Apply survivorship-bias-free universe mask if enabled
            if self.survivorship_bias_free and self.universe_mask is not None:
                signals = self.sp500_loader.apply_universe_mask(signals, self.universe_mask)

            results = self.backtest_engine.run(self.prices, signals, self.benchmark)

            return {
                "total_return": results.total_return,
                "annual_return": results.annual_return,
                "sharpe_ratio": results.sharpe_ratio,
                "sortino_ratio": results.sortino_ratio,
                "calmar_ratio": results.calmar_ratio,
                "max_drawdown": results.max_drawdown,
                "num_trades": results.num_trades,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor,
                "alpha": results.alpha,
                "beta": results.beta,
                "correlation": results.correlation,
                "start_date": results.start_date,
                "end_date": results.end_date,
            }
        except Exception as e:
            print(f"    Backtest error: {e}")
            return None

    def _evaluate(self, metrics: Dict) -> Tuple[bool, List[str], List[str]]:
        """Evaluate metrics and return (passed, failures, learnings)."""
        passed, failures = check_passing_criteria(metrics)

        learnings = []

        # Extract learnings based on results
        if metrics["sharpe_ratio"] > 0.3:
            learnings.append(f"Positive risk-adjusted returns with Sharpe {metrics['sharpe_ratio']:.2f}")

        if metrics["max_drawdown"] < -0.5:
            learnings.append(f"High drawdown risk: {metrics['max_drawdown']:.1%}")

        if metrics["win_rate"] > 0.55:
            learnings.append(f"Good win rate: {metrics['win_rate']:.1%}")

        if metrics["profit_factor"] > 1.5:
            learnings.append(f"Strong profit factor: {metrics['profit_factor']:.2f}")

        if abs(metrics["correlation"]) < 0.3:
            learnings.append(f"Low market correlation: {metrics['correlation']:.2f}")

        return passed, failures, learnings

    def run(self) -> Dict[str, Any]:
        """Run the automated workflow."""
        print("\n" + "=" * 70)
        print("VIBEQUANT FULLY AUTOMATED ALPHA DISCOVERY")
        print("=" * 70)
        print(f"Max iterations: {self.max_iterations}")
        print(f"Target successful strategies: {self.min_successful}")
        print(f"Started: {datetime.now()}")
        print("=" * 70 + "\n")

        # Load data
        if not self._load_data():
            return {"error": "Failed to load data"}

        iteration = 0

        while iteration < self.max_iterations:
            # Check termination
            if len(self.successful_strategies) >= self.min_successful:
                print(f"\n[SUCCESS] Found {len(self.successful_strategies)} successful strategies!")
                break

            # Generate strategy
            result = self._generate_strategy()
            if result is None:
                print("\n[EXHAUSTED] All parameter combinations tested")
                break

            name, category, hypothesis, code, params = result
            iteration += 1

            print(f"\n[{iteration}/{self.max_iterations}] {name}")
            print(f"    Category: {category}")

            # Save to memory
            hyp_record = self.memory.add_hypothesis(
                hypothesis=hypothesis,
                rationale=f"Auto-generated from {name} template",
                category=category,
            )

            strategy_record = self.memory.add_strategy(
                name=name,
                hypothesis_id=hyp_record.id,
                description=hypothesis,
                category=category,
                code=code,
                parameters=params,
            )

            # Run backtest
            metrics = self._run_backtest(code)

            if metrics is None:
                self.failed_strategies.append({
                    "name": name,
                    "reason": "Backtest execution error",
                })
                continue

            # Evaluate
            passed, failures, learnings = self._evaluate(metrics)

            # Print results
            print(f"    Return: {metrics['annual_return']:>7.1%}  |  Sharpe: {metrics['sharpe_ratio']:>6.2f}  |  DD: {metrics['max_drawdown']:>7.1%}  |  {'PASS' if passed else 'FAIL'}")

            # Update memory
            bt_result = BacktestResult(
                total_return=metrics["total_return"],
                annual_return=metrics["annual_return"],
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                win_rate=metrics["win_rate"],
                num_trades=metrics["num_trades"],
                start_date=metrics["start_date"],
                end_date=metrics["end_date"],
                alpha=metrics["alpha"],
                beta=metrics["beta"],
                correlation_to_spy=metrics["correlation"],
                sortino_ratio=metrics["sortino_ratio"],
                profit_factor=metrics["profit_factor"],
            )

            self.memory.update_strategy_backtest(
                strategy_id=strategy_record.id,
                backtest_results=bt_result,
                passed=passed,
                failure_reasons=failures,
            )

            # Store learnings
            for learning in learnings:
                if passed:
                    self.memory.add_learning("successful_patterns", f"{name}: {learning}")
                else:
                    self.memory.add_learning("failed_patterns", f"{name}: {learning}")

            # Track results
            if passed:
                self.successful_strategies.append({
                    "name": name,
                    "category": category,
                    "sharpe": metrics["sharpe_ratio"],
                    "return": metrics["annual_return"],
                    "drawdown": metrics["max_drawdown"],
                    "params": params,
                })
                print(f"    *** PASSED! ***")
            else:
                self.failed_strategies.append({
                    "name": name,
                    "reason": "; ".join(failures[:2]),
                })

        # Final summary
        print("\n" + "=" * 70)
        print("DISCOVERY COMPLETE")
        print("=" * 70)
        print(f"Iterations: {iteration}")
        print(f"Successful: {len(self.successful_strategies)}")
        print(f"Failed: {len(self.failed_strategies)}")

        if self.successful_strategies:
            print(f"\n{'='*70}")
            print("SUCCESSFUL STRATEGIES")
            print(f"{'='*70}")
            for s in sorted(self.successful_strategies, key=lambda x: x["sharpe"], reverse=True):
                print(f"\n{s['name']}")
                print(f"  Category: {s['category']}")
                print(f"  Sharpe: {s['sharpe']:.2f}")
                print(f"  Annual Return: {s['return']:.1%}")
                print(f"  Max Drawdown: {s['drawdown']:.1%}")
                print(f"  Parameters: {s['params']}")

        print("\n" + self.memory.get_summary_report())

        return {
            "iterations": iteration,
            "successful": self.successful_strategies,
            "failed_count": len(self.failed_strategies),
        }


def run_auto(
    max_iterations: int = 50,
    min_successful: int = 3,
    universe_size: int = 200,
    survivorship_bias_free: bool = False,
    include_etfs: bool = True,
) -> Dict[str, Any]:
    """Run the automated workflow."""
    workflow = AutomatedWorkflow(
        max_iterations=max_iterations,
        min_successful=min_successful,
        universe_size=universe_size,
        survivorship_bias_free=survivorship_bias_free,
        include_etfs=include_etfs,
    )
    return workflow.run()


if __name__ == "__main__":
    run_auto()
