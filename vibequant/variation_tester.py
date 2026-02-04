"""
Strategy Variation Tester

Tests strategies across multiple configurations:
- Different universes
- Stop-loss variants
- Position sizing
- With/without survivorship mask
- Smart variations based on strategy characteristics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple, Any, Union
from dataclasses import dataclass
import json
import os

from .data import AlpacaDataLoader, SP500SurvivorshipFreeLoader, UniverseLoader
from .backtest import BacktestEngine, BacktestConfig
from .smart_variations import (
    SmartVariationGenerator,
    SmartVariation,
    StrategyAnalyzer,
    StrategyCharacteristics,
    analyze_and_suggest_variations,
    print_analysis_report,
)


@dataclass
class UniverseConfig:
    """Configuration for a universe."""
    name: str
    description: str
    survivorship_free: bool = False
    max_symbols: int = 200
    min_avg_volume: int = 500_000
    include_etfs: bool = True
    etfs_only: bool = False


# Predefined universe configurations
UNIVERSE_CONFIGS = {
    # S&P 500 variants
    "sp500_sf_200": UniverseConfig(
        name="S&P 500 Survivorship-Free (200)",
        description="Top 200 most liquid S&P 500 stocks with survivorship-free mask",
        survivorship_free=True,
        max_symbols=200,
        include_etfs=True,
    ),
    "sp500_sf_all": UniverseConfig(
        name="S&P 500 Survivorship-Free (All)",
        description="All S&P 500 stocks with survivorship-free mask",
        survivorship_free=True,
        max_symbols=0,  # No limit
        include_etfs=True,
    ),
    "sp500_no_mask_200": UniverseConfig(
        name="S&P 500 No Mask (200)",
        description="Top 200 S&P 500 stocks WITHOUT survivorship mask (for comparison)",
        survivorship_free=False,  # Load S&P 500 data but don't apply mask
        max_symbols=200,
        include_etfs=True,
    ),

    # Broader liquid universes
    "liquid_200": UniverseConfig(
        name="Liquid 200",
        description="Top 200 most liquid stocks (any stock, contains survivorship bias)",
        survivorship_free=False,
        max_symbols=200,
        include_etfs=True,
    ),
    "liquid_500": UniverseConfig(
        name="Liquid 500",
        description="Top 500 most liquid US stocks",
        survivorship_free=False,
        max_symbols=500,
        include_etfs=True,
    ),
    "liquid_1000": UniverseConfig(
        name="Liquid 1000",
        description="Top 1000 most liquid US stocks",
        survivorship_free=False,
        max_symbols=1000,
        include_etfs=True,
    ),

    # Index-based
    "nasdaq100": UniverseConfig(
        name="NASDAQ 100",
        description="NASDAQ 100 index constituents",
        survivorship_free=False,
        max_symbols=100,
        include_etfs=False,
    ),
    "russell1000": UniverseConfig(
        name="Russell 1000 (Sample)",
        description="Russell 1000 large-cap stocks (representative sample)",
        survivorship_free=False,
        max_symbols=200,
        include_etfs=False,
    ),

    # ETFs
    "etfs_only": UniverseConfig(
        name="ETFs Only",
        description="ETFs only - no single stock risk",
        survivorship_free=False,
        max_symbols=0,
        include_etfs=True,
        etfs_only=True,
    ),
}


@dataclass
class VariationConfig:
    """Configuration for a strategy variation."""
    name: str
    stop_loss: Optional[float] = None  # e.g., 0.05 for 5% stop
    take_profit: Optional[float] = None  # e.g., 0.10 for 10% take profit
    max_position_pct: float = 1.0  # Max position size as % of portfolio
    trailing_stop: Optional[float] = None  # e.g., 0.03 for 3% trailing stop
    volatility_scaling: bool = False  # Scale positions by inverse volatility


# Predefined variation configurations
VARIATION_CONFIGS = {
    "base": VariationConfig(
        name="Base (No Risk Management)",
    ),
    "stop_5pct": VariationConfig(
        name="5% Stop Loss",
        stop_loss=0.05,
    ),
    "stop_10pct": VariationConfig(
        name="10% Stop Loss",
        stop_loss=0.10,
    ),
    "stop_trail_5pct": VariationConfig(
        name="5% Trailing Stop",
        trailing_stop=0.05,
    ),
    "stop_3_tp_10": VariationConfig(
        name="3% Stop / 10% Take Profit",
        stop_loss=0.03,
        take_profit=0.10,
    ),
    "vol_scaled": VariationConfig(
        name="Volatility-Scaled Positions",
        volatility_scaling=True,
    ),
    "stop_5_vol_scaled": VariationConfig(
        name="5% Stop + Vol Scaling",
        stop_loss=0.05,
        volatility_scaling=True,
    ),
}


class StrategyVariationTester:
    """
    Tests a strategy across multiple universes and configurations.
    """

    def __init__(self, memory_dir: str = "./memory"):
        self.memory_dir = memory_dir
        self.data_loader = AlpacaDataLoader()
        self.backtest_engine = BacktestEngine(BacktestConfig())

        # Cache for loaded data
        self._data_cache: Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]] = {}

    def _load_universe_data(
        self, config: UniverseConfig
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[SP500SurvivorshipFreeLoader]]:
        """Load data for a universe configuration."""
        cache_key = f"{config.name}_{config.max_symbols}_{config.min_avg_volume}"

        if cache_key in self._data_cache:
            prices, universe_mask = self._data_cache[cache_key]
            return prices, universe_mask, None

        print(f"\n[Loading] {config.name}...")

        universe_loader = UniverseLoader()

        if config.etfs_only:
            # ETFs only
            etfs = self.data_loader.get_popular_etfs()
            data = self.data_loader.get_bars(
                symbols=etfs,
                timeframe="1Day",
                start=datetime(2016, 1, 1),
            )
            prices = data["close"].unstack(level=0)
            universe_mask = None
            sp500_loader = None

        elif config.survivorship_free or "sp500" in config.name.lower():
            # S&P 500 based
            sp500_loader = SP500SurvivorshipFreeLoader()
            prices, universe_mask = sp500_loader.load_survivorship_free_data(
                start_date=datetime(2016, 1, 1),
                include_etfs=config.include_etfs,
                max_symbols=config.max_symbols,
                min_avg_volume=config.min_avg_volume,
            )
            # If not survivorship_free mode, we still load S&P 500 data but won't apply mask
            if not config.survivorship_free:
                universe_mask = None

        elif "nasdaq100" in config.name.lower():
            # NASDAQ 100
            info = universe_loader.load_universe("nasdaq100")
            symbols = info.symbols
            if config.include_etfs:
                etfs = self.data_loader.get_popular_etfs()
                symbols = list(set(symbols + etfs))
            data = self.data_loader.get_bars(
                symbols=symbols,
                timeframe="1Day",
                start=datetime(2016, 1, 1),
            )
            prices = data["close"].unstack(level=0)
            universe_mask = None
            sp500_loader = None

        elif "russell" in config.name.lower():
            # Russell 1000 sample
            info = universe_loader.load_universe("russell1000", max_symbols=config.max_symbols)
            symbols = info.symbols
            if config.include_etfs:
                etfs = self.data_loader.get_popular_etfs()
                symbols = list(set(symbols + etfs))
            data = self.data_loader.get_bars(
                symbols=symbols,
                timeframe="1Day",
                start=datetime(2016, 1, 1),
            )
            prices = data["close"].unstack(level=0)
            universe_mask = None
            sp500_loader = None

        else:
            # Generic liquid universe
            symbols = self.data_loader.get_liquid_universe(
                min_volume=config.min_avg_volume,
                max_symbols=config.max_symbols if config.max_symbols > 0 else 5000,
            )
            if config.include_etfs:
                etfs = self.data_loader.get_popular_etfs()
                symbols = list(set(symbols + etfs))

            # Limit symbols if max_symbols is set
            if config.max_symbols > 0:
                symbols = symbols[:config.max_symbols]

            data = self.data_loader.get_bars(
                symbols=symbols,
                timeframe="1Day",
                start=datetime(2016, 1, 1),
            )
            prices = data["close"].unstack(level=0)
            universe_mask = None
            sp500_loader = None

        self._data_cache[cache_key] = (prices, universe_mask)
        print(f"[Loaded] {prices.shape[1]} symbols, {prices.shape[0]} days")

        return prices, universe_mask, sp500_loader if config.survivorship_free else None

    def apply_risk_management(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        config: Union[VariationConfig, SmartVariation],
    ) -> pd.DataFrame:
        """Apply risk management rules to signals.

        Supports both VariationConfig (static) and SmartVariation (smart).
        """
        # Extract parameters (works for both config types)
        stop_loss = getattr(config, 'stop_loss', None)
        take_profit = getattr(config, 'take_profit', None)
        trailing_stop = getattr(config, 'trailing_stop', None)
        volatility_scaling = getattr(config, 'volatility_scaling', False)
        inverse_volatility = getattr(config, 'inverse_volatility', False)
        max_position_pct = getattr(config, 'max_position_pct', 1.0)
        time_stop = getattr(config, 'time_stop', None)

        if stop_loss is None and take_profit is None and \
           trailing_stop is None and not volatility_scaling and \
           not inverse_volatility and time_stop is None and max_position_pct >= 1.0:
            return signals

        modified_signals = signals.copy()

        # Volatility scaling (target volatility approach)
        if volatility_scaling:
            returns = prices.pct_change()
            volatility = returns.rolling(window=20).std()
            target_vol = 0.02  # 2% daily vol target
            vol_scale = target_vol / volatility.replace(0, np.nan)
            vol_scale = vol_scale.clip(0.2, 5.0).fillna(1.0)  # Cap scaling factor
            modified_signals = modified_signals * vol_scale

        # Inverse volatility weighting (equal risk contribution)
        if inverse_volatility:
            returns = prices.pct_change()
            volatility = returns.rolling(window=20).std()
            inv_vol = 1.0 / volatility.replace(0, np.nan)
            # Normalize so weights sum to original sum
            for i in range(20, len(modified_signals)):
                row_signals = modified_signals.iloc[i]
                active = row_signals > 0
                if active.sum() > 0:
                    inv_vol_row = inv_vol.iloc[i][active]
                    if not inv_vol_row.isna().all():
                        # Scale by inverse volatility, normalized
                        weights = inv_vol_row / inv_vol_row.sum()
                        original_sum = row_signals[active].sum()
                        for col in weights.index:
                            modified_signals.iloc[i, modified_signals.columns.get_loc(col)] = weights[col] * original_sum

        # Max position cap
        if max_position_pct < 1.0:
            modified_signals = modified_signals.clip(upper=max_position_pct)
            # Renormalize
            row_sums = modified_signals.sum(axis=1).replace(0, 1)
            modified_signals = modified_signals.div(row_sums, axis=0)

        # Stop loss / take profit / trailing stop / time stop
        if stop_loss or take_profit or trailing_stop or time_stop:
            returns = prices.pct_change()

            # Track entry prices, trailing highs, and days held
            entry_prices = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
            trailing_highs = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
            days_held = pd.DataFrame(0, index=prices.index, columns=prices.columns)

            for i in range(1, len(modified_signals)):
                for col in modified_signals.columns:
                    prev_sig = modified_signals.iloc[i-1][col]
                    curr_sig = modified_signals.iloc[i][col]
                    curr_price = prices.iloc[i][col]
                    prev_price = prices.iloc[i-1][col]

                    # New position entry
                    if prev_sig == 0 and curr_sig > 0:
                        entry_prices.iloc[i][col] = curr_price
                        trailing_highs.iloc[i][col] = curr_price
                        days_held.iloc[i][col] = 1

                    # Existing position
                    elif prev_sig > 0 and curr_sig > 0:
                        entry_price = entry_prices.iloc[i-1][col]
                        entry_prices.iloc[i][col] = entry_price
                        days_held.iloc[i][col] = days_held.iloc[i-1][col] + 1

                        # Update trailing high
                        prev_high = trailing_highs.iloc[i-1][col]
                        trailing_highs.iloc[i][col] = max(prev_high, curr_price) if not np.isnan(prev_high) else curr_price

                        if not np.isnan(entry_price):
                            pct_change = (curr_price - entry_price) / entry_price

                            # Time stop (exit after N days)
                            if time_stop and days_held.iloc[i][col] >= time_stop:
                                modified_signals.iloc[i][col] = 0
                                continue

                            # Stop loss
                            if stop_loss and pct_change <= -stop_loss:
                                modified_signals.iloc[i][col] = 0
                                continue

                            # Take profit
                            if take_profit and pct_change >= take_profit:
                                modified_signals.iloc[i][col] = 0
                                continue

                            # Trailing stop
                            if trailing_stop:
                                high = trailing_highs.iloc[i][col]
                                if not np.isnan(high):
                                    drawdown = (curr_price - high) / high
                                    if drawdown <= -trailing_stop:
                                        modified_signals.iloc[i][col] = 0
                                        continue

        # Renormalize weights
        row_sums = modified_signals.sum(axis=1).replace(0, 1)
        modified_signals = modified_signals.div(row_sums, axis=0)

        return modified_signals.fillna(0)

    def test_strategy(
        self,
        strategy_code: str,
        strategy_name: str,
        universes: List[str] = None,
        variations: List[str] = None,
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Test a strategy across multiple universes and variations.

        Args:
            strategy_code: Python code that defines generate_signals(prices)
            strategy_name: Name for this strategy
            universes: List of universe config names (default: all)
            variations: List of variation config names (default: all)
            save_results: Whether to save results to memory

        Returns:
            DataFrame with results for all combinations
        """
        universes = universes or list(UNIVERSE_CONFIGS.keys())
        variations = variations or list(VARIATION_CONFIGS.keys())

        results = []

        print("=" * 70)
        print(f"STRATEGY VARIATION TESTING: {strategy_name}")
        print("=" * 70)
        print(f"Universes: {len(universes)}")
        print(f"Variations: {len(variations)}")
        print(f"Total combinations: {len(universes) * len(variations)}")
        print("=" * 70)

        for universe_key in universes:
            universe_config = UNIVERSE_CONFIGS[universe_key]

            try:
                prices, universe_mask, sp500_loader = self._load_universe_data(universe_config)
            except Exception as e:
                print(f"[ERROR] Failed to load {universe_key}: {e}")
                continue

            # Generate base signals
            try:
                local_vars = {"pd": pd, "np": np}
                exec(strategy_code, local_vars)
                generate_signals = local_vars["generate_signals"]
                base_signals = generate_signals(prices)
            except Exception as e:
                print(f"[ERROR] Strategy code failed: {e}")
                continue

            # Apply survivorship mask if configured
            if universe_config.survivorship_free and universe_mask is not None and sp500_loader:
                base_signals = sp500_loader.apply_universe_mask(base_signals, universe_mask)

            for variation_key in variations:
                variation_config = VARIATION_CONFIGS[variation_key]

                print(f"\n[Testing] {universe_config.name} + {variation_config.name}")

                try:
                    # Apply risk management
                    signals = self.apply_risk_management(base_signals, prices, variation_config)

                    # Run backtest
                    benchmark = prices.get("SPY")
                    backtest_results = self.backtest_engine.run(prices, signals, benchmark)

                    result = {
                        "strategy": strategy_name,
                        "universe": universe_config.name,
                        "universe_key": universe_key,
                        "variation": variation_config.name,
                        "variation_key": variation_key,
                        "survivorship_free": universe_config.survivorship_free,
                        "total_return": backtest_results.total_return,
                        "annual_return": backtest_results.annual_return,
                        "sharpe_ratio": backtest_results.sharpe_ratio,
                        "sortino_ratio": backtest_results.sortino_ratio,
                        "max_drawdown": backtest_results.max_drawdown,
                        "win_rate": backtest_results.win_rate,
                        "num_trades": backtest_results.num_trades,
                        "profit_factor": backtest_results.profit_factor,
                        "correlation": backtest_results.correlation,
                    }
                    results.append(result)

                    status = "PASS" if backtest_results.sharpe_ratio >= 0.5 else "FAIL"
                    print(f"    Return: {backtest_results.annual_return:.1%} | "
                          f"Sharpe: {backtest_results.sharpe_ratio:.2f} | "
                          f"DD: {backtest_results.max_drawdown:.1%} | {status}")

                except Exception as e:
                    print(f"    [ERROR] {e}")
                    continue

        # Create results DataFrame
        df = pd.DataFrame(results)

        if save_results and not df.empty:
            self._save_results(df, strategy_name)

        return df

    def test_strategy_smart(
        self,
        strategy_code: str,
        strategy_name: str,
        category: str = None,
        universes: List[str] = None,
        max_variations: int = 10,
        save_results: bool = True,
    ) -> Tuple[pd.DataFrame, StrategyCharacteristics, List[SmartVariation]]:
        """
        Test a strategy with SMART variations tailored to its characteristics.

        This method:
        1. Analyzes the strategy code to detect type, timing, etc.
        2. Generates appropriate variations based on characteristics
        3. Tests across specified universes with smart variations
        4. Returns results plus analysis info

        Args:
            strategy_code: Python code that defines generate_signals(prices)
            strategy_name: Name for this strategy
            category: Optional category hint (momentum, mean_reversion, etc.)
            universes: List of universe config names (default: subset)
            max_variations: Maximum number of smart variations to test
            save_results: Whether to save results to memory

        Returns:
            Tuple of (results DataFrame, characteristics, variations tested)
        """
        # Default to a reasonable subset of universes
        universes = universes or ["sp500_sf_200", "liquid_500", "nasdaq100"]

        # Step 1: Analyze the strategy
        generator = SmartVariationGenerator()
        chars = generator.analyzer.analyze(strategy_code, category)

        print("=" * 70)
        print(f"SMART VARIATION TESTING: {strategy_name}")
        print("=" * 70)
        print(f"\nStrategy Analysis (confidence: {chars.confidence:.0%})")
        print(f"  Type: {chars.strategy_type.value}")
        print(f"  Rebalance: {chars.rebalance_type.value}")
        print(f"  Lookback: {chars.lookback_period or 'unknown'}")
        print(f"  Hold Period: {chars.hold_period or 'unknown'}")
        print(f"  Max Positions: {chars.max_positions or 'unknown'}")
        print(f"  Indicators: {', '.join(chars.indicators) or 'none'}")

        # Step 2: Run base backtest first to get metrics for adaptive variations
        print("\n[Phase 1] Running base backtest to gather metrics...")

        base_metrics = None
        for universe_key in universes[:1]:  # Just first universe for base metrics
            try:
                universe_config = UNIVERSE_CONFIGS[universe_key]
                prices, universe_mask, sp500_loader = self._load_universe_data(universe_config)

                local_vars = {"pd": pd, "np": np}
                exec(strategy_code, local_vars)
                generate_signals = local_vars["generate_signals"]
                base_signals = generate_signals(prices)

                if universe_config.survivorship_free and universe_mask is not None and sp500_loader:
                    base_signals = sp500_loader.apply_universe_mask(base_signals, universe_mask)

                benchmark = prices.get("SPY")
                base_results = self.backtest_engine.run(prices, base_signals, benchmark)

                base_metrics = {
                    "sharpe_ratio": base_results.sharpe_ratio,
                    "max_drawdown": base_results.max_drawdown,
                    "win_rate": base_results.win_rate,
                    "correlation": base_results.correlation,
                    "profit_factor": base_results.profit_factor,
                }
                print(f"  Base Sharpe: {base_metrics['sharpe_ratio']:.2f}")
                print(f"  Base Drawdown: {base_metrics['max_drawdown']:.1%}")
                print(f"  Base Win Rate: {base_metrics['win_rate']:.1%}")
                break
            except Exception as e:
                print(f"  [ERROR] Could not get base metrics: {e}")

        # Step 3: Generate smart variations
        smart_variations = generator.generate_variations(strategy_code, category, base_metrics)
        smart_variations = smart_variations[:max_variations]

        print(f"\n[Phase 2] Generated {len(smart_variations)} smart variations:")
        for i, var in enumerate(smart_variations, 1):
            print(f"  {i}. {var.name} (priority: {var.priority})")

        # Step 4: Test all combinations
        print(f"\n[Phase 3] Testing {len(universes)} universes x {len(smart_variations)} variations")
        print("=" * 70)

        results = []

        for universe_key in universes:
            universe_config = UNIVERSE_CONFIGS.get(universe_key)
            if not universe_config:
                print(f"[WARN] Unknown universe: {universe_key}")
                continue

            try:
                prices, universe_mask, sp500_loader = self._load_universe_data(universe_config)
            except Exception as e:
                print(f"[ERROR] Failed to load {universe_key}: {e}")
                continue

            # Generate base signals
            try:
                local_vars = {"pd": pd, "np": np}
                exec(strategy_code, local_vars)
                generate_signals = local_vars["generate_signals"]
                base_signals = generate_signals(prices)
            except Exception as e:
                print(f"[ERROR] Strategy code failed: {e}")
                continue

            # Apply survivorship mask if configured
            if universe_config.survivorship_free and universe_mask is not None and sp500_loader:
                base_signals = sp500_loader.apply_universe_mask(base_signals, universe_mask)

            for variation in smart_variations:
                print(f"\n[Testing] {universe_config.name} + {variation.name}")

                try:
                    # Apply smart variation's risk management
                    signals = self.apply_risk_management(base_signals, prices, variation)

                    # Run backtest
                    benchmark = prices.get("SPY")
                    backtest_results = self.backtest_engine.run(prices, signals, benchmark)

                    result = {
                        "strategy": strategy_name,
                        "universe": universe_config.name,
                        "universe_key": universe_key,
                        "variation": variation.name,
                        "variation_rationale": variation.rationale,
                        "survivorship_free": universe_config.survivorship_free,
                        "total_return": backtest_results.total_return,
                        "annual_return": backtest_results.annual_return,
                        "sharpe_ratio": backtest_results.sharpe_ratio,
                        "sortino_ratio": backtest_results.sortino_ratio,
                        "max_drawdown": backtest_results.max_drawdown,
                        "win_rate": backtest_results.win_rate,
                        "num_trades": backtest_results.num_trades,
                        "profit_factor": backtest_results.profit_factor,
                        "correlation": backtest_results.correlation,
                        # Smart variation details
                        "stop_loss": variation.stop_loss,
                        "trailing_stop": variation.trailing_stop,
                        "take_profit": variation.take_profit,
                        "time_stop": variation.time_stop,
                        "vol_scaling": variation.volatility_scaling,
                        "inv_vol": variation.inverse_volatility,
                    }
                    results.append(result)

                    status = "PASS" if backtest_results.sharpe_ratio >= 0.5 else "FAIL"
                    print(f"    Return: {backtest_results.annual_return:.1%} | "
                          f"Sharpe: {backtest_results.sharpe_ratio:.2f} | "
                          f"DD: {backtest_results.max_drawdown:.1%} | {status}")

                except Exception as e:
                    print(f"    [ERROR] {e}")
                    continue

        # Create results DataFrame
        df = pd.DataFrame(results)

        if save_results and not df.empty:
            self._save_smart_results(df, strategy_name, chars, smart_variations)

        return df, chars, smart_variations

    def _save_smart_results(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        chars: StrategyCharacteristics,
        variations: List[SmartVariation],
    ):
        """Save smart variation test results."""
        os.makedirs(self.memory_dir, exist_ok=True)

        # Save full results
        filename = f"smart_variation_{strategy_name.replace(' ', '_').lower()}.json"
        filepath = os.path.join(self.memory_dir, filename)

        # Include analysis metadata
        output = {
            "strategy": strategy_name,
            "tested_at": datetime.now().isoformat(),
            "analysis": {
                "strategy_type": chars.strategy_type.value,
                "rebalance_type": chars.rebalance_type.value,
                "lookback_period": chars.lookback_period,
                "hold_period": chars.hold_period,
                "max_positions": chars.max_positions,
                "indicators": chars.indicators,
                "confidence": chars.confidence,
            },
            "variations_tested": [
                {
                    "name": v.name,
                    "rationale": v.rationale,
                    "stop_loss": v.stop_loss,
                    "trailing_stop": v.trailing_stop,
                    "take_profit": v.take_profit,
                    "time_stop": v.time_stop,
                    "volatility_scaling": v.volatility_scaling,
                    "inverse_volatility": v.inverse_volatility,
                    "priority": v.priority,
                }
                for v in variations
            ],
            "results": df.to_dict(orient="records"),
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n[Saved] Smart variation results to {filepath}")

        # Update learnings
        learnings_path = os.path.join(self.memory_dir, "learnings.json")
        if os.path.exists(learnings_path):
            with open(learnings_path, "r") as f:
                learnings = json.load(f)
        else:
            learnings = {}

        if "smart_variation_tests" not in learnings:
            learnings["smart_variation_tests"] = []

        # Best improvement
        base_result = df[df["variation"] == "Base"]
        if not base_result.empty:
            base_sharpe = base_result["sharpe_ratio"].mean()
            best_sharpe = df["sharpe_ratio"].max()
            best_var = df.loc[df["sharpe_ratio"].idxmax(), "variation"]
            improvement = best_sharpe - base_sharpe

            learnings["smart_variation_tests"].append({
                "strategy": strategy_name,
                "strategy_type": chars.strategy_type.value,
                "tested_at": datetime.now().isoformat(),
                "base_sharpe": base_sharpe,
                "best_sharpe": best_sharpe,
                "best_variation": best_var,
                "improvement": improvement,
                "num_variations": len(variations),
            })

        with open(learnings_path, "w") as f:
            json.dump(learnings, f, indent=2, default=str)

    def print_smart_summary(
        self,
        df: pd.DataFrame,
        chars: StrategyCharacteristics,
        variations: List[SmartVariation],
    ):
        """Print a summary of smart variation test results."""
        if df.empty:
            print("No results to summarize.")
            return

        print("\n" + "=" * 70)
        print("SMART VARIATION TEST SUMMARY")
        print("=" * 70)

        # Strategy characteristics
        print(f"\nStrategy Type: {chars.strategy_type.value}")
        print(f"Detected Hold Period: {chars.hold_period or 'unknown'}")

        # Best overall
        best_idx = df["sharpe_ratio"].idxmax()
        best = df.loc[best_idx]
        print(f"\nBEST CONFIGURATION:")
        print(f"  Universe: {best['universe']}")
        print(f"  Variation: {best['variation']}")
        print(f"  Rationale: {best.get('variation_rationale', 'N/A')}")
        print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
        print(f"  Annual Return: {best['annual_return']:.1%}")
        print(f"  Max Drawdown: {best['max_drawdown']:.1%}")

        # Improvement from base
        base_results = df[df["variation"] == "Base"]
        if not base_results.empty:
            base_sharpe = base_results["sharpe_ratio"].mean()
            print(f"\nIMPROVEMENT FROM BASE:")
            print(f"  Base Sharpe: {base_sharpe:.2f}")
            print(f"  Best Sharpe: {best['sharpe_ratio']:.2f}")
            print(f"  Improvement: {best['sharpe_ratio'] - base_sharpe:+.2f}")

        # By variation (sorted by avg improvement)
        print("\nBY VARIATION (avg Sharpe):")
        by_variation = df.groupby("variation")["sharpe_ratio"].mean().sort_values(ascending=False)
        for variation, sharpe in by_variation.items():
            var_obj = next((v for v in variations if v.name == variation), None)
            priority = var_obj.priority if var_obj else "?"
            print(f"  {variation}: {sharpe:.2f} (priority: {priority})")

        # Insights
        print("\nINSIGHTS:")
        if chars.rebalance_type.value == "fixed_period":
            print("  - Fixed-period strategy: Stop-losses may not improve results")
        if chars.strategy_type.value == "mean_reversion":
            print("  - Mean reversion: Tight stops and time stops recommended")
        if chars.strategy_type.value == "momentum":
            print("  - Momentum: Trailing stops to ride winners recommended")

        print("=" * 70)

    def _save_results(self, df: pd.DataFrame, strategy_name: str):
        """Save variation test results."""
        os.makedirs(self.memory_dir, exist_ok=True)

        # Save full results
        filename = f"variation_test_{strategy_name.replace(' ', '_').lower()}.json"
        filepath = os.path.join(self.memory_dir, filename)
        df.to_json(filepath, orient="records", indent=2)
        print(f"\n[Saved] Results to {filepath}")

        # Update learnings
        learnings_path = os.path.join(self.memory_dir, "learnings.json")
        if os.path.exists(learnings_path):
            with open(learnings_path, "r") as f:
                learnings = json.load(f)
        else:
            learnings = {"variation_tests": []}

        if "variation_tests" not in learnings:
            learnings["variation_tests"] = []

        # Add summary
        summary = {
            "strategy": strategy_name,
            "tested_at": datetime.now().isoformat(),
            "num_combinations": len(df),
            "best_sharpe": df["sharpe_ratio"].max(),
            "best_config": df.loc[df["sharpe_ratio"].idxmax()].to_dict() if not df.empty else None,
            "worst_sharpe": df["sharpe_ratio"].min(),
            "avg_sharpe_by_universe": df.groupby("universe")["sharpe_ratio"].mean().to_dict(),
            "avg_sharpe_by_variation": df.groupby("variation")["sharpe_ratio"].mean().to_dict(),
        }
        learnings["variation_tests"].append(summary)

        with open(learnings_path, "w") as f:
            json.dump(learnings, f, indent=2, default=str)

    def print_summary(self, df: pd.DataFrame):
        """Print a summary of variation test results."""
        if df.empty:
            print("No results to summarize.")
            return

        print("\n" + "=" * 70)
        print("VARIATION TEST SUMMARY")
        print("=" * 70)

        # Best overall
        best_idx = df["sharpe_ratio"].idxmax()
        best = df.loc[best_idx]
        print(f"\nBEST CONFIGURATION:")
        print(f"  Universe: {best['universe']}")
        print(f"  Variation: {best['variation']}")
        print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
        print(f"  Annual Return: {best['annual_return']:.1%}")
        print(f"  Max Drawdown: {best['max_drawdown']:.1%}")

        # By universe
        print("\nBY UNIVERSE (avg Sharpe):")
        by_universe = df.groupby("universe")["sharpe_ratio"].mean().sort_values(ascending=False)
        for universe, sharpe in by_universe.items():
            print(f"  {universe}: {sharpe:.2f}")

        # By variation
        print("\nBY VARIATION (avg Sharpe):")
        by_variation = df.groupby("variation")["sharpe_ratio"].mean().sort_values(ascending=False)
        for variation, sharpe in by_variation.items():
            print(f"  {variation}: {sharpe:.2f}")

        # Survivorship bias impact
        if "survivorship_free" in df.columns:
            sf_true = df[df["survivorship_free"] == True]["sharpe_ratio"].mean()
            sf_false = df[df["survivorship_free"] == False]["sharpe_ratio"].mean()
            if not np.isnan(sf_true) and not np.isnan(sf_false):
                print(f"\nSURVIVORSHIP BIAS IMPACT:")
                print(f"  With survivorship-free mask: {sf_true:.2f} avg Sharpe")
                print(f"  Without mask: {sf_false:.2f} avg Sharpe")
                print(f"  Difference: {sf_false - sf_true:.2f}")

        print("=" * 70)


def test_strategy_variations(
    strategy_code: str,
    strategy_name: str,
    universes: List[str] = None,
    variations: List[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to test strategy variations (static variations).

    Args:
        strategy_code: Python code defining generate_signals(prices)
        strategy_name: Name for the strategy
        universes: List of universe keys (default: all)
        variations: List of variation keys (default: all)

    Returns:
        DataFrame with all test results
    """
    tester = StrategyVariationTester()
    results = tester.test_strategy(
        strategy_code=strategy_code,
        strategy_name=strategy_name,
        universes=universes,
        variations=variations,
    )
    tester.print_summary(results)
    return results


def test_smart_variations(
    strategy_code: str,
    strategy_name: str,
    category: str = None,
    universes: List[str] = None,
    max_variations: int = 10,
) -> Tuple[pd.DataFrame, StrategyCharacteristics, List[SmartVariation]]:
    """
    Convenience function to test with SMART variations.

    Smart variations are automatically generated based on:
    - Strategy type (momentum, mean-reversion, volatility, breakout)
    - Rebalance frequency (fixed-period vs signal-based)
    - Holding period
    - Base backtest metrics (win rate, drawdown, correlation)

    Args:
        strategy_code: Python code defining generate_signals(prices)
        strategy_name: Name for the strategy
        category: Optional hint (momentum, mean_reversion, volatility, breakout)
        universes: List of universe keys (default: sp500_sf_200, liquid_500, nasdaq100)
        max_variations: Maximum variations to test (default: 10)

    Returns:
        Tuple of (results DataFrame, strategy characteristics, variations tested)

    Example:
        >>> code = '''
        ... def generate_signals(prices):
        ...     lookback = 5
        ...     hold_period = 5
        ...     bottom_n = 5
        ...     returns = prices.pct_change(lookback)
        ...     signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        ...     for i in range(lookback, len(prices), hold_period):
        ...         ret = returns.iloc[i].dropna()
        ...         if len(ret) >= bottom_n:
        ...             losers = ret.nsmallest(bottom_n).index.tolist()
        ...             weight = 1.0 / len(losers)
        ...             for j in range(i, min(i + hold_period, len(prices))):
        ...                 for sym in losers:
        ...                     signals.loc[prices.index[j], sym] = weight
        ...     return signals.fillna(0)
        ... '''
        >>> results, chars, variations = test_smart_variations(
        ...     code, "Short-term Reversal", category="mean_reversion"
        ... )
    """
    tester = StrategyVariationTester()
    results, chars, variations = tester.test_strategy_smart(
        strategy_code=strategy_code,
        strategy_name=strategy_name,
        category=category,
        universes=universes,
        max_variations=max_variations,
    )
    tester.print_smart_summary(results, chars, variations)
    return results, chars, variations


if __name__ == "__main__":
    # Example: Test reversal strategy with SMART variations
    reversal_code = '''
def generate_signals(prices):
    lookback = 5
    hold_period = 5
    bottom_n = 5

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
                    signals.iloc[j, signals.columns.get_loc(sym)] = weight
    return signals.fillna(0)
'''

    print("Testing with SMART variations...")
    results, chars, variations = test_smart_variations(
        strategy_code=reversal_code,
        strategy_name="Short-term Reversal",
        category="mean_reversion",
        universes=["sp500_sf_200", "liquid_500"],
        max_variations=8,
    )
