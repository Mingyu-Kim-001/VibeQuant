"""
Smart Variation Generator

Analyzes strategy characteristics and generates appropriate variations:
- Detects strategy type (momentum, mean-reversion, volatility, etc.)
- Identifies holding period and rebalance frequency
- Generates variations suited to the strategy's behavior
- Learns from backtest results to suggest improvements
"""

import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class StrategyType(Enum):
    """Types of trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    BREAKOUT = "breakout"
    FACTOR = "factor"
    PAIRS = "pairs"
    SEASONAL = "seasonal"
    UNKNOWN = "unknown"


class RebalanceType(Enum):
    """How the strategy rebalances."""
    FIXED_PERIOD = "fixed_period"  # Rebalances every N days
    SIGNAL_BASED = "signal_based"  # Rebalances when signals change
    HYBRID = "hybrid"  # Both


@dataclass
class StrategyCharacteristics:
    """Detected characteristics of a strategy."""
    strategy_type: StrategyType = StrategyType.UNKNOWN
    rebalance_type: RebalanceType = RebalanceType.SIGNAL_BASED

    # Timing parameters (detected from code)
    lookback_period: Optional[int] = None
    hold_period: Optional[int] = None
    rebalance_frequency: Optional[int] = None

    # Position parameters
    max_positions: Optional[int] = None
    position_sizing: str = "equal_weight"  # equal_weight, rank_based, vol_scaled

    # Entry/exit logic
    has_explicit_exit: bool = False
    uses_stop_loss: bool = False
    uses_take_profit: bool = False

    # Indicators used
    indicators: List[str] = field(default_factory=list)

    # Confidence in detection
    confidence: float = 0.5


@dataclass
class SmartVariation:
    """A variation generated based on strategy characteristics."""
    name: str
    description: str
    rationale: str  # Why this variation might help

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    time_stop: Optional[int] = None  # Exit after N days regardless

    # Position sizing
    volatility_scaling: bool = False
    inverse_volatility: bool = False
    kelly_sizing: bool = False
    max_position_pct: float = 1.0

    # Entry/exit modifications
    scale_in: bool = False  # Scale into positions
    scale_out: bool = False  # Scale out of positions
    confirmation_period: int = 0  # Wait N days before entry

    # Expected impact
    expected_sharpe_impact: str = "neutral"  # positive, negative, neutral
    expected_drawdown_impact: str = "neutral"

    # Priority (higher = try first)
    priority: int = 5


class StrategyAnalyzer:
    """
    Analyzes strategy code to detect characteristics.
    """

    # Patterns for detecting strategy type
    MOMENTUM_PATTERNS = [
        r'\.pct_change\s*\(\s*(\d+)',  # Returns over N periods
        r'momentum\s*=',
        r'nlargest',  # Top N selection
        r'\.rank\(',
        r'winners|top_n|top\s*=',
    ]

    MEAN_REVERSION_PATTERNS = [
        r'rsi\s*[<>=]',
        r'oversold|overbought',
        r'nsmallest',  # Bottom N selection
        r'losers|bottom_n|bottom\s*=',
        r'z.?score',
        r'mean.*revert|revert.*mean',
    ]

    VOLATILITY_PATTERNS = [
        r'volatility|vol\s*=',
        r'\.std\s*\(',
        r'low.?vol|high.?vol',
        r'vol_lookback|vol_period',
    ]

    BREAKOUT_PATTERNS = [
        r'rolling.*max|rolling.*min',
        r'high_lookback|new.*high',
        r'breakout',
        r'52.?week',
    ]

    def analyze(self, code: str, category: str = None) -> StrategyCharacteristics:
        """
        Analyze strategy code to detect characteristics.

        Args:
            code: Strategy code string
            category: Optional category hint (from hypothesis)

        Returns:
            StrategyCharacteristics with detected parameters
        """
        chars = StrategyCharacteristics()
        code_lower = code.lower()

        # Detect strategy type
        chars.strategy_type = self._detect_strategy_type(code_lower, category)

        # Detect timing parameters
        chars.lookback_period = self._extract_param(code, ['lookback', 'lookback_period', 'period'])
        chars.hold_period = self._extract_param(code, ['hold_period', 'hold', 'holding_period'])
        chars.rebalance_frequency = self._extract_param(code, ['rebal_period', 'rebalance', 'rebal'])

        # Detect rebalance type
        chars.rebalance_type = self._detect_rebalance_type(code)

        # Detect position parameters
        chars.max_positions = self._extract_param(code, ['max_positions', 'top_n', 'bottom_n', 'n_positions'])

        # Detect existing risk management
        chars.uses_stop_loss = 'stop_loss' in code_lower or 'stop' in code_lower
        chars.uses_take_profit = 'take_profit' in code_lower or 'target' in code_lower
        chars.has_explicit_exit = self._has_explicit_exit(code)

        # Detect indicators
        chars.indicators = self._detect_indicators(code_lower)

        # Calculate confidence
        chars.confidence = self._calculate_confidence(chars, code)

        return chars

    def _detect_strategy_type(self, code: str, category: str = None) -> StrategyType:
        """Detect strategy type from code patterns."""
        # Use category hint if provided
        if category:
            category_lower = category.lower()
            if 'momentum' in category_lower:
                return StrategyType.MOMENTUM
            elif 'reversion' in category_lower or 'reversal' in category_lower:
                return StrategyType.MEAN_REVERSION
            elif 'volatility' in category_lower:
                return StrategyType.VOLATILITY
            elif 'breakout' in category_lower:
                return StrategyType.BREAKOUT
            elif 'factor' in category_lower:
                return StrategyType.FACTOR

        # Score each type based on pattern matches
        scores = {
            StrategyType.MOMENTUM: 0,
            StrategyType.MEAN_REVERSION: 0,
            StrategyType.VOLATILITY: 0,
            StrategyType.BREAKOUT: 0,
        }

        for pattern in self.MOMENTUM_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                scores[StrategyType.MOMENTUM] += 1

        for pattern in self.MEAN_REVERSION_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                scores[StrategyType.MEAN_REVERSION] += 1

        for pattern in self.VOLATILITY_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                scores[StrategyType.VOLATILITY] += 1

        for pattern in self.BREAKOUT_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                scores[StrategyType.BREAKOUT] += 1

        # Return highest scoring type
        max_score = max(scores.values())
        if max_score == 0:
            return StrategyType.UNKNOWN

        for stype, score in scores.items():
            if score == max_score:
                return stype

        return StrategyType.UNKNOWN

    def _extract_param(self, code: str, param_names: List[str]) -> Optional[int]:
        """Extract a numeric parameter from code."""
        for name in param_names:
            # Match patterns like "lookback = 5" or "lookback=5"
            pattern = rf'{name}\s*=\s*(\d+)'
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _detect_rebalance_type(self, code: str) -> RebalanceType:
        """Detect how the strategy rebalances."""
        # Fixed period: uses range(start, end, step) pattern
        if re.search(r'range\s*\([^)]+,\s*len\([^)]+\)\s*,', code):
            return RebalanceType.FIXED_PERIOD

        # Signal-based: directly assigns signals without period loop
        if 'for i in range(len(' in code and ', hold_period)' not in code and ', rebal' not in code:
            return RebalanceType.SIGNAL_BASED

        return RebalanceType.HYBRID

    def _has_explicit_exit(self, code: str) -> bool:
        """Check if strategy has explicit exit conditions."""
        exit_patterns = [
            r'exit',
            r'signals.*=\s*0',
            r'close.*position',
            r'rsi.*>.*exit',
        ]
        for pattern in exit_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False

    def _detect_indicators(self, code: str) -> List[str]:
        """Detect technical indicators used."""
        indicators = []

        indicator_patterns = {
            'RSI': r'\brsi\b',
            'SMA': r'\.rolling\(.*\)\.mean\(\)',
            'EMA': r'\.ewm\(',
            'Volatility': r'\.std\(',
            'Momentum': r'pct_change',
            'Bollinger': r'bollinger|bband',
            'MACD': r'macd',
            'ATR': r'\batr\b',
        }

        for indicator, pattern in indicator_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                indicators.append(indicator)

        return indicators

    def _calculate_confidence(self, chars: StrategyCharacteristics, code: str) -> float:
        """Calculate confidence in the analysis."""
        confidence = 0.3  # Base confidence

        # Higher confidence if we detected type
        if chars.strategy_type != StrategyType.UNKNOWN:
            confidence += 0.2

        # Higher confidence if we found timing parameters
        if chars.lookback_period:
            confidence += 0.1
        if chars.hold_period or chars.rebalance_frequency:
            confidence += 0.1

        # Higher confidence if we found indicators
        if chars.indicators:
            confidence += 0.1

        # Cap at 1.0
        return min(confidence, 1.0)


class SmartVariationGenerator:
    """
    Generates appropriate variations based on strategy characteristics.
    """

    def __init__(self):
        self.analyzer = StrategyAnalyzer()

    def generate_variations(
        self,
        code: str,
        category: str = None,
        base_metrics: Dict[str, float] = None,
    ) -> List[SmartVariation]:
        """
        Generate smart variations for a strategy.

        Args:
            code: Strategy code
            category: Strategy category hint
            base_metrics: Metrics from base backtest (optional, for adaptive variations)

        Returns:
            List of SmartVariation objects, sorted by priority
        """
        # Analyze the strategy
        chars = self.analyzer.analyze(code, category)

        variations = []

        # Always include base (no modification)
        variations.append(SmartVariation(
            name="Base",
            description="Original strategy without modifications",
            rationale="Baseline for comparison",
            priority=10,
        ))

        # Generate type-specific variations
        if chars.strategy_type == StrategyType.MOMENTUM:
            variations.extend(self._momentum_variations(chars, base_metrics))
        elif chars.strategy_type == StrategyType.MEAN_REVERSION:
            variations.extend(self._mean_reversion_variations(chars, base_metrics))
        elif chars.strategy_type == StrategyType.VOLATILITY:
            variations.extend(self._volatility_variations(chars, base_metrics))
        elif chars.strategy_type == StrategyType.BREAKOUT:
            variations.extend(self._breakout_variations(chars, base_metrics))
        else:
            variations.extend(self._generic_variations(chars, base_metrics))

        # Generate timing-based variations
        variations.extend(self._timing_variations(chars, base_metrics))

        # Generate metrics-based variations if we have base results
        if base_metrics:
            variations.extend(self._adaptive_variations(chars, base_metrics))

        # Sort by priority (higher first)
        variations.sort(key=lambda v: v.priority, reverse=True)

        return variations

    def _momentum_variations(
        self,
        chars: StrategyCharacteristics,
        metrics: Dict[str, float] = None
    ) -> List[SmartVariation]:
        """Generate variations for momentum strategies."""
        variations = []

        # Momentum strategies tend to have fat tails - trailing stops help
        variations.append(SmartVariation(
            name="Trailing Stop (Wide)",
            description="15% trailing stop to ride winners while limiting large losses",
            rationale="Momentum winners can run far; wide trailing stop captures more upside",
            trailing_stop=0.15,
            expected_sharpe_impact="positive",
            expected_drawdown_impact="positive",
            priority=8,
        ))

        variations.append(SmartVariation(
            name="Trailing Stop (Tight)",
            description="8% trailing stop for earlier profit protection",
            rationale="Tighter stop locks in gains faster but may exit winners early",
            trailing_stop=0.08,
            expected_sharpe_impact="neutral",
            expected_drawdown_impact="positive",
            priority=7,
        ))

        # Volatility scaling helps normalize risk across holdings
        variations.append(SmartVariation(
            name="Inverse Volatility Weighting",
            description="Size positions inversely to their volatility",
            rationale="Equalizes risk contribution; high-momentum stocks are often volatile",
            inverse_volatility=True,
            expected_sharpe_impact="positive",
            expected_drawdown_impact="positive",
            priority=9,
        ))

        # Scale into positions for momentum
        variations.append(SmartVariation(
            name="Scale-In Entry",
            description="Enter 50% initially, add 50% after 2-day confirmation",
            rationale="Confirms momentum before full position; reduces whipsaw",
            scale_in=True,
            confirmation_period=2,
            expected_sharpe_impact="positive",
            expected_drawdown_impact="positive",
            priority=6,
        ))

        return variations

    def _mean_reversion_variations(
        self,
        chars: StrategyCharacteristics,
        metrics: Dict[str, float] = None
    ) -> List[SmartVariation]:
        """Generate variations for mean reversion strategies."""
        variations = []

        # Mean reversion has high win rate but can have big losses
        # Tight stop loss is important
        variations.append(SmartVariation(
            name="Tight Stop Loss",
            description="5% stop loss to limit losses on failed reversions",
            rationale="Mean reversion can fail catastrophically; tight stops essential",
            stop_loss=0.05,
            expected_sharpe_impact="positive",
            expected_drawdown_impact="positive",
            priority=9,
        ))

        variations.append(SmartVariation(
            name="Stop + Take Profit",
            description="5% stop loss with 10% take profit",
            rationale="Captures typical reversion move while limiting downside",
            stop_loss=0.05,
            take_profit=0.10,
            expected_sharpe_impact="positive",
            expected_drawdown_impact="positive",
            priority=8,
        ))

        # Time-based exit for mean reversion
        if chars.hold_period:
            # Use detected hold period
            time_stop = chars.hold_period * 2
        else:
            time_stop = 10  # Default

        variations.append(SmartVariation(
            name=f"Time Stop ({time_stop}d)",
            description=f"Exit after {time_stop} days if reversion hasn't occurred",
            rationale="Failed reversions tie up capital; time stops free it",
            time_stop=time_stop,
            expected_sharpe_impact="neutral",
            expected_drawdown_impact="positive",
            priority=7,
        ))

        # Volatility scaling less important for mean reversion
        # but can help with sizing
        variations.append(SmartVariation(
            name="Vol-Adjusted Sizing",
            description="Reduce position size in highly volatile stocks",
            rationale="Volatile stocks have wider swings; smaller size = same risk",
            volatility_scaling=True,
            max_position_pct=0.8,
            expected_sharpe_impact="neutral",
            expected_drawdown_impact="positive",
            priority=6,
        ))

        return variations

    def _volatility_variations(
        self,
        chars: StrategyCharacteristics,
        metrics: Dict[str, float] = None
    ) -> List[SmartVariation]:
        """Generate variations for volatility-based strategies."""
        variations = []

        # Low vol strategies are defensive - less need for stops
        variations.append(SmartVariation(
            name="Wide Stop Only",
            description="20% stop loss as disaster protection only",
            rationale="Low-vol strategies rarely hit stops; this protects against regime changes",
            stop_loss=0.20,
            expected_sharpe_impact="neutral",
            expected_drawdown_impact="positive",
            priority=7,
        ))

        # Momentum overlay can help
        variations.append(SmartVariation(
            name="Momentum Filter",
            description="Only hold low-vol stocks with positive momentum",
            rationale="Avoids value traps in low-vol universe",
            confirmation_period=1,  # Placeholder - would need code modification
            expected_sharpe_impact="positive",
            expected_drawdown_impact="positive",
            priority=8,
        ))

        return variations

    def _breakout_variations(
        self,
        chars: StrategyCharacteristics,
        metrics: Dict[str, float] = None
    ) -> List[SmartVariation]:
        """Generate variations for breakout strategies."""
        variations = []

        # Breakouts can fail quickly - tight initial stop
        variations.append(SmartVariation(
            name="Breakout Stop",
            description="3% initial stop, then trailing 10%",
            rationale="Failed breakouts reverse fast; protect capital then let winners run",
            stop_loss=0.03,
            trailing_stop=0.10,
            expected_sharpe_impact="positive",
            expected_drawdown_impact="positive",
            priority=9,
        ))

        # Confirmation helps filter false breakouts
        variations.append(SmartVariation(
            name="Confirmed Breakout",
            description="Wait 1 day after breakout signal before entry",
            rationale="Filters false breakouts; real breakouts persist",
            confirmation_period=1,
            expected_sharpe_impact="positive",
            expected_drawdown_impact="positive",
            priority=8,
        ))

        # Scale out as move extends
        variations.append(SmartVariation(
            name="Scale-Out Exits",
            description="Exit 50% at +10%, remainder with trailing stop",
            rationale="Locks in gains while allowing remainder to run",
            scale_out=True,
            take_profit=0.10,
            trailing_stop=0.15,
            expected_sharpe_impact="positive",
            expected_drawdown_impact="positive",
            priority=7,
        ))

        return variations

    def _generic_variations(
        self,
        chars: StrategyCharacteristics,
        metrics: Dict[str, float] = None
    ) -> List[SmartVariation]:
        """Generate generic variations for unknown strategy types."""
        variations = []

        variations.append(SmartVariation(
            name="Conservative Stop",
            description="10% stop loss",
            rationale="Generic risk management",
            stop_loss=0.10,
            priority=6,
        ))

        variations.append(SmartVariation(
            name="Vol Scaling",
            description="Volatility-scaled position sizing",
            rationale="Equalizes risk across positions",
            volatility_scaling=True,
            priority=6,
        ))

        return variations

    def _timing_variations(
        self,
        chars: StrategyCharacteristics,
        metrics: Dict[str, float] = None
    ) -> List[SmartVariation]:
        """Generate variations based on strategy timing."""
        variations = []

        # If strategy uses fixed holding period, stop-loss often doesn't help
        if chars.rebalance_type == RebalanceType.FIXED_PERIOD:
            # Don't add stop-loss variations - they don't help
            # Instead, focus on position sizing
            variations.append(SmartVariation(
                name="Reduced Max Position",
                description="Limit max position to 15% instead of full weight",
                rationale="Fixed-period strategies can't use stops; reduce position risk instead",
                max_position_pct=0.15,
                expected_sharpe_impact="neutral",
                expected_drawdown_impact="positive",
                priority=7,
            ))

        # Short holding period strategies
        if chars.hold_period and chars.hold_period <= 5:
            variations.append(SmartVariation(
                name="Tighter Entry Filter",
                description="Require stronger signal for entry",
                rationale="Short-term trades need high conviction; filter weak signals",
                confirmation_period=0,  # Would need code modification
                expected_sharpe_impact="positive",
                expected_drawdown_impact="positive",
                priority=6,
            ))

        # Long holding period strategies
        elif chars.hold_period and chars.hold_period >= 20:
            variations.append(SmartVariation(
                name="Monthly Rebalance Overlay",
                description="Only rebalance at month-end regardless of signals",
                rationale="Reduces turnover and trading costs for long-term strategies",
                expected_sharpe_impact="neutral",
                expected_drawdown_impact="neutral",
                priority=5,
            ))

        return variations

    def _adaptive_variations(
        self,
        chars: StrategyCharacteristics,
        metrics: Dict[str, float]
    ) -> List[SmartVariation]:
        """Generate variations based on base strategy metrics."""
        variations = []

        # High drawdown - add protective measures
        if metrics.get('max_drawdown', 0) < -0.4:  # More than 40% drawdown
            variations.append(SmartVariation(
                name="Drawdown Control",
                description="Reduce exposure by 50% after 15% drawdown",
                rationale=f"Base strategy had {metrics['max_drawdown']:.0%} drawdown; add protection",
                max_position_pct=0.5,
                expected_sharpe_impact="neutral",
                expected_drawdown_impact="positive",
                priority=9,
            ))

        # Low win rate - need to let winners run
        if metrics.get('win_rate', 0.5) < 0.45:
            variations.append(SmartVariation(
                name="Let Winners Run",
                description="No take profit; use trailing stop instead",
                rationale=f"Win rate is {metrics['win_rate']:.0%}; need big winners to compensate",
                trailing_stop=0.12,
                expected_sharpe_impact="positive",
                expected_drawdown_impact="neutral",
                priority=8,
            ))

        # High win rate - can be more aggressive
        if metrics.get('win_rate', 0.5) > 0.6:
            variations.append(SmartVariation(
                name="Aggressive Sizing",
                description="Increase position sizes given high win rate",
                rationale=f"Win rate is {metrics['win_rate']:.0%}; can afford larger positions",
                max_position_pct=1.2,
                expected_sharpe_impact="positive",
                expected_drawdown_impact="negative",
                priority=6,
            ))

        # Low Sharpe but positive - might benefit from vol scaling
        if 0 < metrics.get('sharpe_ratio', 0) < 0.5:
            variations.append(SmartVariation(
                name="Risk Parity",
                description="Equal risk contribution from each position",
                rationale=f"Sharpe is {metrics['sharpe_ratio']:.2f}; vol scaling may improve risk-adjusted returns",
                inverse_volatility=True,
                volatility_scaling=True,
                expected_sharpe_impact="positive",
                expected_drawdown_impact="positive",
                priority=8,
            ))

        # High correlation to market - add hedging
        if abs(metrics.get('correlation', 0)) > 0.7:
            variations.append(SmartVariation(
                name="Market Hedge Overlay",
                description="Consider market-neutral variation",
                rationale=f"Correlation is {metrics['correlation']:.2f}; strategy may just be beta exposure",
                expected_sharpe_impact="neutral",
                expected_drawdown_impact="positive",
                priority=7,
            ))

        return variations


def analyze_and_suggest_variations(
    code: str,
    category: str = None,
    base_metrics: Dict[str, float] = None,
) -> Tuple[StrategyCharacteristics, List[SmartVariation]]:
    """
    Convenience function to analyze a strategy and get variation suggestions.

    Args:
        code: Strategy code
        category: Strategy category hint
        base_metrics: Metrics from base backtest

    Returns:
        Tuple of (characteristics, variations)
    """
    generator = SmartVariationGenerator()
    analyzer = generator.analyzer

    chars = analyzer.analyze(code, category)
    variations = generator.generate_variations(code, category, base_metrics)

    return chars, variations


def print_analysis_report(
    code: str,
    category: str = None,
    base_metrics: Dict[str, float] = None,
):
    """Print a formatted analysis report."""
    chars, variations = analyze_and_suggest_variations(code, category, base_metrics)

    print("=" * 70)
    print("STRATEGY ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nDetected Characteristics (confidence: {chars.confidence:.0%})")
    print(f"  Strategy Type: {chars.strategy_type.value}")
    print(f"  Rebalance Type: {chars.rebalance_type.value}")
    print(f"  Lookback Period: {chars.lookback_period or 'unknown'}")
    print(f"  Hold Period: {chars.hold_period or 'unknown'}")
    print(f"  Max Positions: {chars.max_positions or 'unknown'}")
    print(f"  Indicators: {', '.join(chars.indicators) or 'none detected'}")
    print(f"  Has Explicit Exit: {chars.has_explicit_exit}")
    print(f"  Uses Stop Loss: {chars.uses_stop_loss}")

    print(f"\nSuggested Variations ({len(variations)} total)")
    print("-" * 70)

    for i, var in enumerate(variations[:10], 1):  # Top 10
        print(f"\n{i}. {var.name} (priority: {var.priority})")
        print(f"   {var.description}")
        print(f"   Rationale: {var.rationale}")
        if var.stop_loss:
            print(f"   Stop Loss: {var.stop_loss:.0%}")
        if var.trailing_stop:
            print(f"   Trailing Stop: {var.trailing_stop:.0%}")
        if var.take_profit:
            print(f"   Take Profit: {var.take_profit:.0%}")
        if var.time_stop:
            print(f"   Time Stop: {var.time_stop} days")
        if var.volatility_scaling:
            print(f"   Volatility Scaling: Yes")
        print(f"   Expected Impact: Sharpe={var.expected_sharpe_impact}, DD={var.expected_drawdown_impact}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example: Analyze a mean reversion strategy
    code = '''
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
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
                    signals.loc[prices.index[j], sym] = weight
    return signals.fillna(0)
'''

    # Example with base metrics
    base_metrics = {
        "sharpe_ratio": 0.83,
        "max_drawdown": -0.85,
        "win_rate": 0.50,
        "correlation": 0.22,
    }

    print_analysis_report(code, category="mean_reversion", base_metrics=base_metrics)
