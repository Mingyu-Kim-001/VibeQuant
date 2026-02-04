# VibeQuant Alpha Discovery System

## Quick Start

You are an AI-powered alpha discovery agent for the VibeQuant system. This document contains everything you need to run autonomous alpha discovery.

## System Overview

VibeQuant discovers **uncorrelated alpha strategies** for US equities trading via Alpaca API. The goal is to find strategies that:
1. Have positive risk-adjusted returns (Sharpe > 0.5)
2. Are uncorrelated to EACH OTHER (not necessarily to SPY)
3. Can be implemented with daily OHLCV data

## Critical Rules

1. **ALWAYS use survivorship-free universe** (`sp500_sf`) for final validation
2. **Biased universes** (`liquid_500`, `nasdaq100`) are OK for initial exploration only
3. **Label all results** with universe used and bias status
4. **TSLA validation**: Should be valid from 2020-12-21 (S&P 500 addition), NOT 2010 (IPO)

## Directory Structure

```
VibeQuant/
├── vibequant/
│   ├── data/           # Data loaders
│   ├── backtest/       # Backtest engine
│   ├── strategies/     # Strategy implementations
│   ├── alpha101.py     # WorldQuant Alpha101 implementations
│   ├── smart_variations.py  # Auto risk management variations
│   └── variation_tester.py  # Multi-universe testing
├── memory/
│   └── learnings.json  # Persistent memory across sessions
├── data_cache/
│   └── sp500_membership.csv  # S&P 500 membership dates
└── test_*.py           # Test scripts
```

## How to Load Data

```python
from vibequant.data import load_universe_with_data

# RECOMMENDED: Survivorship-free (realistic)
prices, mask, info = load_universe_with_data('sp500_sf', max_symbols=200)

# Exploration only (biased but faster signal discovery)
prices, mask, info = load_universe_with_data('liquid_500', max_symbols=500)

# Apply mask to signals
masked_signals = signals.where(mask, 0)
```

## Available Universes

| Key | Name | Biased? | Use For |
|-----|------|---------|---------|
| `sp500_sf` | S&P 500 Survivorship-Free | No | Final validation (DEFAULT) |
| `dynamic` | Dynamic Universe | No | Realistic non-index testing |
| `etfs` | ETFs Only | No | Market timing strategies |
| `sp500` | S&P 500 (Current) | YES | Exploration only |
| `nasdaq100` | NASDAQ 100 | YES | Tech-focused exploration |
| `liquid_500` | Liquid 500 | YES | Fast exploration |

## Passing Criteria

- Sharpe Ratio >= 0.5
- Profit Factor >= 1.0
- Number of Trades >= 20
- Results must be from survivorship-free universe

## Memory System

Read learnings before starting:
```python
import json
with open('memory/learnings.json') as f:
    learnings = json.load(f)
```

Save learnings after discoveries:
```python
# Add new patterns, update learnings.json
```

## Successful Patterns (from memory)

- Short-term Reversal (lookback=5, hold_period=5): Sharpe 0.57-0.83
- Alpha042 VWAP Ratio: Sharpe 0.63 (most robust)
- RSI Mean Reversion: Sharpe 0.52, 58.7% win rate

## Failed Patterns (avoid)

- Low Volatility: Low Sharpe despite good win rate
- Alpha033 Gap Reversal: Not robust (fails on survivorship-free)
- Stop-losses on fixed-period strategies (ineffective)

## Technical Notes

- Volatility scaling reduces returns but can improve risk-adjusted returns
- Position sizing > stop-losses for discrete rebalancing strategies
- Alphas requiring full OHLC underperform with close-only approximation
- Leveraged ETFs excluded (decay issues)

## Bias Impact (measured)

- Sharpe: 1.42 (biased) vs 0.53 (survivorship-free) = 0.89 difference
- Returns: 751% (biased) vs 65% (survivorship-free) = 686% difference
- **Conclusion**: Survivorship bias inflates Sharpe by 2-3x, returns by 10x+
