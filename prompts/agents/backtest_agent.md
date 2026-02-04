# Backtest Agent

Execute backtests using `vibequant.backtest.BacktestEngine`.

## Input
- `strategy_name`: Name of the strategy
- `strategy_code`: Python code with `generate_signals(prices)` function
- `universe`: Universe to test (default: `sp500_sf`)

## Execution

```python
from vibequant.data import load_universe_with_data
from vibequant.backtest import BacktestEngine, execute_strategy_code

# Load data
prices, mask, info = load_universe_with_data(universe, max_symbols=200)

# Execute strategy and apply mask
signals = execute_strategy_code(strategy_code, prices)
masked_signals = signals.where(mask, 0)

# Normalize
row_sums = masked_signals.abs().sum(axis=1).replace(0, 1)
masked_signals = masked_signals.div(row_sums, axis=0)

# Run backtest
engine = BacktestEngine()
results = engine.run(prices, masked_signals)
```

## Output

Return JSON with:
- `success`: bool
- `universe`: str  
- `survivorship_free`: bool (from `info.survivorship_free`)
- `metrics`: dict from `results` (sharpe_ratio, annual_return, max_drawdown, etc.)
- `error_message`: str if failed

## Important
- Always report which universe was used
- Flag if `info.survivorship_free` is False
