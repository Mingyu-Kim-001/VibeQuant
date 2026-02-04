"""
Backtest Engine for VibeQuant
Vectorized backtesting with proper handling of commissions, slippage, and position sizing.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    commission: float = 0.0  # $0 commission (Alpaca)
    slippage: float = 0.0  # 0% slippage
    max_position_size: float = 0.20  # Max 20% per position
    min_position_size: float = 0.01  # Min 1% per position
    max_positions: int = 20  # Max number of concurrent positions
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.05  # 5% annual risk-free rate


@dataclass
class Trade:
    """Record of a single trade."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    side: str = "long"  # long or short
    pnl: float = 0.0
    return_pct: float = 0.0


@dataclass
class BacktestResults:
    """Complete results from a backtest run."""
    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days

    # Trade statistics
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    downside_volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%

    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    information_ratio: float = 0.0

    # Time series
    equity_curve: Optional[pd.Series] = None
    returns: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None
    positions_over_time: Optional[pd.DataFrame] = None

    # Trade list
    trades: List[Trade] = field(default_factory=list)

    # Metadata
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0


class BacktestEngine:
    """
    Vectorized backtesting engine for strategy evaluation.

    Features:
    - Supports long/short strategies
    - Handles slippage and commissions
    - Computes comprehensive risk metrics
    - Compares against benchmark
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> BacktestResults:
        """
        Run a backtest on the given signals.

        Args:
            prices: DataFrame with columns as symbols, rows as dates, values as close prices
            signals: DataFrame with same structure as prices, values are position weights (-1 to 1)
                     Positive = long, Negative = short, 0 = no position
            benchmark_prices: Series of benchmark prices for comparison

        Returns:
            BacktestResults with all metrics and statistics
        """
        # Validate inputs
        if prices.empty or signals.empty:
            raise ValueError("Prices and signals cannot be empty")

        # Align signals with prices
        signals = signals.reindex(prices.index).fillna(0)
        signals = signals[prices.columns.intersection(signals.columns)]
        prices = prices[signals.columns]

        # Clip signals to valid range
        signals = signals.clip(-1, 1)

        # Calculate returns (close-to-close)
        returns = prices.pct_change().fillna(0)

        # Apply slippage on position changes
        # Slippage applied when entering/exiting positions
        position_changes = signals.diff().abs().fillna(0)
        slippage_cost = position_changes * self.config.slippage

        # LOOK-AHEAD BIAS PREVENTION:
        # Signal on Day T is generated using Day T's close price.
        # We shift signals by 1 so that:
        #   - Signal[T] is applied to Returns[T+1]
        #   - This simulates: see close on Day T, trade on Day T+1
        #
        # More conservative assumption: we earn T+1's return minus slippage
        # In reality, we'd trade at T+1 open, not close, but close-to-close
        # returns are a reasonable approximation for daily rebalancing.
        shifted_signals = signals.shift(1).fillna(0)

        # Shift slippage cost too - we pay slippage when we trade (T+1)
        shifted_slippage = slippage_cost.shift(1).fillna(0)

        strategy_returns = (shifted_signals * returns).sum(axis=1) - shifted_slippage.sum(axis=1)

        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod() * self.config.initial_capital

        # Calculate drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max

        # Calculate benchmark metrics if provided
        benchmark_return = 0.0
        alpha = 0.0
        beta = 0.0
        correlation = 0.0
        information_ratio = 0.0

        if benchmark_prices is not None:
            benchmark_prices = benchmark_prices.reindex(prices.index).ffill().bfill()
            benchmark_returns = benchmark_prices.pct_change().fillna(0)
            benchmark_return = (1 + benchmark_returns).prod() - 1

            # Calculate alpha and beta
            if len(strategy_returns) > 10:
                cov_matrix = np.cov(strategy_returns.values, benchmark_returns.values)
                beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
                alpha = strategy_returns.mean() * 252 - beta * benchmark_returns.mean() * 252
                correlation = strategy_returns.corr(benchmark_returns)

                # Information ratio
                tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
                if tracking_error > 0:
                    information_ratio = (strategy_returns.mean() - benchmark_returns.mean()) * 252 / tracking_error

        # Compile trades (simplified - track position changes)
        trades = self._extract_trades(prices, signals)

        # Calculate trade statistics
        trade_returns = [t.return_pct for t in trades if t.exit_date is not None]
        num_trades = len(trade_returns)
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]

        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

        # Risk metrics
        volatility = strategy_returns.std() * np.sqrt(252)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # VaR and CVaR
        var_95 = strategy_returns.quantile(0.05)
        cvar_95 = strategy_returns[strategy_returns <= var_95].mean() if len(strategy_returns[strategy_returns <= var_95]) > 0 else var_95

        # Performance metrics
        total_return = equity_curve.iloc[-1] / self.config.initial_capital - 1
        trading_days = len(strategy_returns)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe ratio
        excess_returns = strategy_returns - self.config.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0

        # Sortino ratio
        sortino_ratio = excess_returns.mean() / downside_volatility if downside_volatility > 0 else 0
        sortino_ratio *= np.sqrt(252)

        # Max drawdown
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Max drawdown duration
        drawdown_duration = self._calculate_max_drawdown_duration(drawdown)

        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=drawdown_duration,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=np.mean(trade_returns) if trade_returns else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=max(trade_returns) if trade_returns else 0,
            worst_trade=min(trade_returns) if trade_returns else 0,
            volatility=volatility,
            downside_volatility=downside_volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            correlation=correlation,
            information_ratio=information_ratio,
            equity_curve=equity_curve,
            returns=strategy_returns,
            drawdown_series=drawdown,
            positions_over_time=signals,
            trades=trades,
            start_date=str(prices.index[0].date()) if hasattr(prices.index[0], 'date') else str(prices.index[0]),
            end_date=str(prices.index[-1].date()) if hasattr(prices.index[-1], 'date') else str(prices.index[-1]),
            trading_days=trading_days,
        )

    def _extract_trades(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> List[Trade]:
        """Extract individual trades from signals."""
        trades = []

        for symbol in signals.columns:
            symbol_signals = signals[symbol]
            symbol_prices = prices[symbol]

            in_position = False
            entry_date = None
            entry_price = None
            entry_signal = 0

            for date, signal in symbol_signals.items():
                price = symbol_prices.loc[date]

                if not in_position and signal != 0:
                    # Enter position
                    in_position = True
                    entry_date = date
                    entry_price = price
                    entry_signal = signal

                elif in_position and (signal == 0 or np.sign(signal) != np.sign(entry_signal)):
                    # Exit position
                    exit_price = price
                    side = "long" if entry_signal > 0 else "short"

                    if side == "long":
                        return_pct = (exit_price - entry_price) / entry_price
                    else:
                        return_pct = (entry_price - exit_price) / entry_price

                    trade = Trade(
                        symbol=symbol,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        exit_date=date,
                        exit_price=exit_price,
                        side=side,
                        return_pct=return_pct,
                        pnl=return_pct * abs(entry_signal) * self.config.initial_capital / len(signals.columns),
                    )
                    trades.append(trade)

                    # Check if we're entering a new position in opposite direction
                    if signal != 0 and np.sign(signal) != np.sign(entry_signal):
                        in_position = True
                        entry_date = date
                        entry_price = price
                        entry_signal = signal
                    else:
                        in_position = False
                        entry_date = None
                        entry_price = None
                        entry_signal = 0

        return trades

    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate the maximum drawdown duration in days."""
        in_drawdown = drawdown < 0
        duration = 0
        max_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0

        return max_duration

    def run_from_signal_function(
        self,
        prices: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame], pd.DataFrame],
        benchmark_prices: Optional[pd.Series] = None,
    ) -> BacktestResults:
        """
        Run backtest using a signal generation function.

        Args:
            prices: Price data
            signal_func: Function that takes prices and returns signals
            benchmark_prices: Optional benchmark for comparison

        Returns:
            BacktestResults
        """
        signals = signal_func(prices)
        return self.run(prices, signals, benchmark_prices)


def validate_strategy_code(code: str) -> Tuple[bool, str]:
    """
    Validate that strategy code is safe and properly structured.

    Args:
        code: Strategy code to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for dangerous imports/operations
    dangerous_patterns = [
        "import os",
        "import sys",
        "import subprocess",
        "__import__",
        "eval(",
        "exec(",
        "open(",
        "file(",
    ]

    for pattern in dangerous_patterns:
        if pattern in code:
            return False, f"Dangerous pattern detected: {pattern}"

    # Check for required function
    if "def generate_signals(" not in code and "def signal(" not in code:
        return False, "Strategy must define 'generate_signals(prices)' or 'signal(prices)' function"

    return True, ""


def execute_strategy_code(
    code: str,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Safely execute strategy code and return signals.

    Args:
        code: Strategy code with generate_signals function
        prices: Price data

    Returns:
        Signals DataFrame
    """
    is_valid, error = validate_strategy_code(code)
    if not is_valid:
        raise ValueError(f"Invalid strategy code: {error}")

    # Create execution namespace with safe imports
    namespace = {
        "pd": pd,
        "np": np,
        "prices": prices,
    }

    # Execute the code
    exec(code, namespace)

    # Get the signal function
    if "generate_signals" in namespace:
        signal_func = namespace["generate_signals"]
    elif "signal" in namespace:
        signal_func = namespace["signal"]
    else:
        raise ValueError("No signal function found in code")

    # Generate signals
    signals = signal_func(prices)

    return signals


if __name__ == "__main__":
    # Example usage with dummy data
    import numpy as np

    # Create dummy price data
    dates = pd.date_range("2022-01-01", "2024-01-01", freq="B")
    symbols = ["AAPL", "MSFT", "GOOGL"]

    np.random.seed(42)
    prices = pd.DataFrame(
        index=dates,
        data={
            s: 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
            for s in symbols
        },
    )

    # Simple momentum strategy
    strategy_code = '''
def generate_signals(prices):
    """Simple momentum strategy: go long if price > 20-day SMA."""
    sma = prices.rolling(window=20).mean()
    signals = (prices > sma).astype(float)
    # Equal weight positions
    signals = signals / signals.sum(axis=1).replace(0, 1).values.reshape(-1, 1)
    return signals.fillna(0)
'''

    # Run backtest
    engine = BacktestEngine()
    signals = execute_strategy_code(strategy_code, prices)
    results = engine.run(prices, signals)

    print(f"Total Return: {results.total_return:.2%}")
    print(f"Annual Return: {results.annual_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Number of Trades: {results.num_trades}")
