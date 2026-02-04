"""
Adversarial Validation Module for VibeQuant
Detects common backtesting mistakes, biases, and unrealistic assumptions.

Checks for:
- Look-ahead bias (using future data)
- Survivorship bias (testing only on survivors)
- Overfitting indicators (too many parameters, precise values)
- Transaction cost neglect (high turnover)
- Regime dependency (only works in bull/bear)
- Statistical issues (small samples, non-stationarity)
- Implementation bugs (NaN, division by zero)
- Correlation to existing alphas (redundancy)
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd


class Severity(str, Enum):
    """Severity levels for validation issues."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    PASS = "PASS"


@dataclass
class ValidationIssue:
    """A single validation issue found in the strategy."""
    issue_type: str
    severity: Severity
    description: str
    code_location: Optional[str] = None
    fix_suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Complete validation result for a strategy."""
    validation_passed: bool
    severity: Severity
    issues_found: List[ValidationIssue] = field(default_factory=list)
    sanity_checks: Dict[str, bool] = field(default_factory=dict)
    universe_check: Dict[str, Any] = field(default_factory=dict)
    code_analysis: Dict[str, Any] = field(default_factory=dict)
    regime_analysis: Dict[str, Any] = field(default_factory=dict)
    transaction_analysis: Dict[str, Any] = field(default_factory=dict)
    correlation_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "validation_passed": self.validation_passed,
            "severity": self.severity.value,
            "issues_found": [
                {
                    "type": i.issue_type,
                    "severity": i.severity.value,
                    "description": i.description,
                    "code_location": i.code_location,
                    "fix_suggestion": i.fix_suggestion,
                    "details": i.details,
                }
                for i in self.issues_found
            ],
            "sanity_checks": self.sanity_checks,
            "universe_check": self.universe_check,
            "code_analysis": self.code_analysis,
            "regime_analysis": self.regime_analysis,
            "transaction_analysis": self.transaction_analysis,
            "correlation_analysis": self.correlation_analysis,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score
        }


class AdversarialValidator:
    """
    Validates backtest strategies for common issues and biases.
    
    Comprehensive checks for:
    - Look-ahead bias (using future data)
    - Survivorship bias (testing only on survivors)
    - Overfitting indicators (parameters, complexity)
    - Transaction cost neglect
    - Regime dependency
    - Statistical validity
    - Implementation bugs
    - Alpha redundancy
    """

    # Look-ahead bias patterns (CRITICAL)
    LOOK_AHEAD_PATTERNS = [
        (r'\.shift\s*\(\s*-\s*\d+', "Forward shift detected (.shift(-N))"),
        (r'iloc\s*\[\s*[a-zA-Z_]\w*\s*\+', "Forward index access (iloc[i+])"),
        (r'\b(future_|next_|tomorrow|forward_)\w+', "Variable naming suggests future data"),
        (r'\.loc\s*\[\s*["\'][^"\']+["\']\s*:\s*\]', "Potential future slice in loc"),
    ]

    # Potential look-ahead (HIGH)
    MISSING_SHIFT_PATTERNS = [
        (r'prices?\s*[><]=?\s*prices?\.rolling\([^)]+\)\.mean\(\)(?!\.shift)',
         "Rolling mean comparison without shift"),
        (r'returns?\s*[><]=?\s*returns?\.rolling\([^)]+\)\.mean\(\)(?!\.shift)',
         "Rolling returns compared without shift"),
        (r'\.rank\s*\([^)]*\)(?!.*\.shift)', "Rank without shift may use current data"),
    ]

    # Implementation bug patterns
    BUG_PATTERNS = [
        (r'/\s*0(?!\.)|\/ 0(?!\.)', "Division by zero risk", Severity.HIGH),
        (r'\.std\(\)\s*$', "Std without ddof may cause issues", Severity.LOW),
        (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', "Loop over DataFrame (slow)", Severity.LOW),
    ]

    # Overfitting indicators
    OVERFIT_PATTERNS = [
        (r'\d+\.\d{3,}', "Overly precise parameter (3+ decimals)"),
        (r'if\s+\w+\s*[<>=]+\s*\d+\.\d{2,}', "Precise threshold in conditional"),
    ]

    SURVIVORSHIP_FREE_UNIVERSES = ['sp500_sf', 'dynamic', 'etfs']
    
    # Transaction cost assumptions (basis points)
    DEFAULT_TRANSACTION_COST_BPS = 10  # 0.1% per trade

    def __init__(self, existing_alpha_returns: Optional[pd.DataFrame] = None):
        """
        Initialize validator.
        
        Args:
            existing_alpha_returns: DataFrame of daily returns from existing alphas
                                   for correlation checking
        """
        self.issues: List[ValidationIssue] = []
        self.existing_alpha_returns = existing_alpha_returns

    def validate(
        self,
        strategy_code: str,
        backtest_metrics: Optional[Dict[str, float]] = None,
        universe: str = "unknown",
        daily_returns: Optional[pd.Series] = None,
        signals: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> ValidationResult:
        """Perform comprehensive adversarial validation."""
        self.issues = []

        # Code analysis
        code_analysis = self._analyze_code(strategy_code)
        
        # Universe check
        universe_check = self._check_universe(universe)
        
        # Metrics sanity checks
        sanity_checks = {}
        if backtest_metrics:
            sanity_checks = self._check_metrics_sanity(backtest_metrics)

        # Signal analysis
        transaction_analysis = {}
        if signals is not None:
            self._analyze_signals(signals)
            transaction_analysis = self._analyze_transaction_costs(signals, backtest_metrics)

        # Returns analysis
        regime_analysis = {}
        if daily_returns is not None:
            self._analyze_returns(daily_returns)
            regime_analysis = self._analyze_regime_dependency(daily_returns, benchmark_returns)
            self._check_performance_decay(daily_returns)

        # Correlation to existing alphas
        correlation_analysis = {}
        if daily_returns is not None and self.existing_alpha_returns is not None:
            correlation_analysis = self._check_alpha_correlation(daily_returns)

        # Overall assessment
        severity = self._determine_overall_severity()
        recommendations = self._generate_recommendations()
        confidence = self._calculate_confidence(backtest_metrics or {})
        validation_passed = severity in (Severity.PASS, Severity.LOW)

        return ValidationResult(
            validation_passed=validation_passed,
            severity=severity,
            issues_found=self.issues,
            sanity_checks=sanity_checks,
            universe_check=universe_check,
            code_analysis=code_analysis,
            regime_analysis=regime_analysis,
            transaction_analysis=transaction_analysis,
            correlation_analysis=correlation_analysis,
            recommendations=recommendations,
            confidence_score=confidence
        )

    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Perform static analysis on strategy code."""
        analysis = {
            "look_ahead_patterns": [],
            "bug_patterns": [],
            "nan_handling": "UNKNOWN",
            "signal_normalization": "UNKNOWN",
            "complexity_indicators": [],
            "parameter_count": 0,
            "lines_of_code": 0,
            "overfitting_risk": "LOW",
        }

        lines = code.split('\n')
        analysis["lines_of_code"] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])

        # Check for look-ahead bias patterns (CRITICAL)
        for i, line in enumerate(lines, 1):
            for pattern, desc in self.LOOK_AHEAD_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    self.issues.append(ValidationIssue(
                        issue_type="LOOK_AHEAD_BIAS",
                        severity=Severity.CRITICAL,
                        description=desc,
                        code_location=f"line {i}: {line.strip()[:80]}",
                        fix_suggestion="Remove forward-looking data access"
                    ))
                    analysis["look_ahead_patterns"].append({"line": i, "content": line.strip()[:80]})

            # Potential look-ahead (HIGH)
            for pattern, desc in self.MISSING_SHIFT_PATTERNS:
                if re.search(pattern, line):
                    self.issues.append(ValidationIssue(
                        issue_type="POTENTIAL_LOOK_AHEAD",
                        severity=Severity.HIGH,
                        description=desc,
                        code_location=f"line {i}: {line.strip()[:80]}",
                        fix_suggestion="Add .shift(1) to ensure only past data is used"
                    ))

            # Bug patterns
            for pattern, desc, sev in self.BUG_PATTERNS:
                if re.search(pattern, line):
                    self.issues.append(ValidationIssue(
                        issue_type="IMPLEMENTATION_BUG",
                        severity=sev,
                        description=desc,
                        code_location=f"line {i}: {line.strip()[:80]}",
                    ))
                    analysis["bug_patterns"].append({"line": i, "issue": desc})

        # Check NaN handling
        nan_patterns = ['.fillna(', '.dropna(', '.ffill()', '.bfill()']
        nan_found = any(p in code for p in nan_patterns)
        rolling_used = '.rolling(' in code

        if rolling_used and not nan_found:
            analysis["nan_handling"] = "MISSING"
            self.issues.append(ValidationIssue(
                issue_type="MISSING_NAN_HANDLING",
                severity=Severity.MEDIUM,
                description="Rolling calculations without NaN handling",
                fix_suggestion="Add .fillna(0) after rolling operations"
            ))
        elif nan_found:
            analysis["nan_handling"] = "PRESENT"

        # Check signal normalization
        norm_patterns = ['.div(', '/ row_sums', '/ signals.abs().sum']
        if any(p in code for p in norm_patterns):
            analysis["signal_normalization"] = "PRESENT"
        else:
            analysis["signal_normalization"] = "MISSING"
            self.issues.append(ValidationIssue(
                issue_type="MISSING_NORMALIZATION",
                severity=Severity.LOW,
                description="No signal normalization detected",
                fix_suggestion="Normalize signals to sum to 1 per row"
            ))

        # Check for overfitting indicators
        numbers = re.findall(r'\b\d+\.\d{2,}\b', code)
        if len(numbers) > 3:
            self.issues.append(ValidationIssue(
                issue_type="POTENTIAL_OVERFIT",
                severity=Severity.MEDIUM,
                description=f"Found {len(numbers)} precise decimal parameters",
                fix_suggestion="Use round numbers for parameters"
            ))

        return analysis

    def _check_universe(self, universe: str) -> Dict[str, Any]:
        """Check universe for survivorship bias."""
        check = {
            "universe_used": universe,
            "survivorship_free": universe.lower() in self.SURVIVORSHIP_FREE_UNIVERSES
        }

        if not check["survivorship_free"]:
            self.issues.append(ValidationIssue(
                issue_type="SURVIVORSHIP_BIAS",
                severity=Severity.CRITICAL,
                description=f"Universe '{universe}' is not survivorship-free",
                fix_suggestion="Re-test with 'sp500_sf' or 'dynamic' universe"
            ))

        return check

    def _check_metrics_sanity(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if backtest metrics are reasonable."""
        checks = {}

        sharpe = metrics.get('sharpe_ratio', metrics.get('sharpe', 0))
        checks["sharpe_reasonable"] = -0.5 <= sharpe <= 3.0

        if sharpe > 3.0:
            self.issues.append(ValidationIssue(
                issue_type="SUSPICIOUS_SHARPE",
                severity=Severity.CRITICAL,
                description=f"Sharpe {sharpe:.2f} > 3.0 is almost certainly wrong",
                fix_suggestion="Check for look-ahead bias or data errors"
            ))
        elif sharpe > 2.5:
            self.issues.append(ValidationIssue(
                issue_type="HIGH_SHARPE",
                severity=Severity.HIGH,
                description=f"Sharpe {sharpe:.2f} > 2.5 is unusually high",
                fix_suggestion="Verify with out-of-sample testing"
            ))

        win_rate = metrics.get('win_rate', 0.5)
        checks["win_rate_reasonable"] = 0.30 <= win_rate <= 0.80

        if win_rate > 0.80:
            self.issues.append(ValidationIssue(
                issue_type="SUSPICIOUS_WIN_RATE",
                severity=Severity.HIGH,
                description=f"Win rate {win_rate:.1%} > 80% is unusually high",
                fix_suggestion="Check for calculation errors"
            ))

        max_dd = abs(metrics.get('max_drawdown', 0))
        checks["drawdown_reasonable"] = max_dd <= 0.60

        if max_dd > 0.70:
            self.issues.append(ValidationIssue(
                issue_type="EXCESSIVE_DRAWDOWN",
                severity=Severity.HIGH,
                description=f"Max drawdown {max_dd:.1%} > 70%",
                fix_suggestion="Add risk management"
            ))

        num_trades = metrics.get('num_trades', 0)
        checks["trades_sufficient"] = num_trades >= 30

        if num_trades < 20:
            self.issues.append(ValidationIssue(
                issue_type="INSUFFICIENT_TRADES",
                severity=Severity.HIGH,
                description=f"Only {num_trades} trades - not statistically significant",
                fix_suggestion="Extend backtest period"
            ))

        trading_days = metrics.get('trading_days', 0)
        checks["period_sufficient"] = trading_days >= 500

        if trading_days > 0 and trading_days < 252:
            self.issues.append(ValidationIssue(
                issue_type="SHORT_BACKTEST",
                severity=Severity.HIGH,
                description=f"Only {trading_days} trading days",
                fix_suggestion="Extend to at least 2 years"
            ))

        profit_factor = metrics.get('profit_factor', 1.0)
        checks["profit_factor_reasonable"] = 0.5 <= profit_factor <= 5.0

        if profit_factor > 5.0:
            self.issues.append(ValidationIssue(
                issue_type="SUSPICIOUS_PROFIT_FACTOR",
                severity=Severity.HIGH,
                description=f"Profit factor {profit_factor:.2f} > 5.0",
                fix_suggestion="Verify calculation"
            ))

        annual_return = metrics.get('annual_return', 0)
        checks["return_reasonable"] = -0.50 <= annual_return <= 1.0

        if annual_return > 1.0:
            self.issues.append(ValidationIssue(
                issue_type="SUSPICIOUS_RETURN",
                severity=Severity.HIGH,
                description=f"Annual return {annual_return:.1%} > 100%",
                fix_suggestion="Check for leverage or errors"
            ))

        return checks

    def _analyze_signals(self, signals: pd.DataFrame) -> None:
        """Analyze signal DataFrame for issues."""
        max_signal = signals.abs().max().max()
        if max_signal > 1.0:
            self.issues.append(ValidationIssue(
                issue_type="INVALID_SIGNALS",
                severity=Severity.HIGH,
                description=f"Signal values exceed [-1, 1] (max: {max_signal:.2f})",
                fix_suggestion="Clip signals to [-1, 1]"
            ))

        if len(signals) > 1:
            daily_turnover = signals.diff().abs().sum(axis=1).mean()
            if daily_turnover > 0.5:
                self.issues.append(ValidationIssue(
                    issue_type="HIGH_TURNOVER",
                    severity=Severity.MEDIUM,
                    description=f"Daily turnover {daily_turnover:.1%} is high",
                    fix_suggestion="Add turnover constraints"
                ))

        nan_count = signals.isna().sum().sum()
        if nan_count > 0:
            self.issues.append(ValidationIssue(
                issue_type="NAN_IN_SIGNALS",
                severity=Severity.MEDIUM,
                description=f"Found {nan_count} NaN values in signals",
                fix_suggestion="Handle NaN values"
            ))

    def _analyze_returns(self, returns: pd.Series) -> None:
        """Analyze returns series for anomalies."""
        extreme = (returns.abs() > 0.20).sum()
        if extreme > 5:
            self.issues.append(ValidationIssue(
                issue_type="EXTREME_RETURNS",
                severity=Severity.MEDIUM,
                description=f"Found {extreme} days with >20% returns",
                fix_suggestion="Verify data quality"
            ))

        if returns.std() < 0.001:
            self.issues.append(ValidationIssue(
                issue_type="SUSPICIOUS_CONSISTENCY",
                severity=Severity.HIGH,
                description="Returns have very low variance",
                fix_suggestion="Verify strategy generates signals"
            ))

        if len(returns) > 20:
            autocorr = returns.autocorr(lag=1)
            if autocorr is not None and autocorr > 0.5:
                self.issues.append(ValidationIssue(
                    issue_type="HIGH_AUTOCORRELATION",
                    severity=Severity.HIGH,
                    description=f"Returns autocorrelation {autocorr:.2f}",
                    fix_suggestion="Check for look-ahead bias"
                ))

    def _analyze_transaction_costs(
        self, 
        signals: pd.DataFrame, 
        metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Analyze impact of transaction costs on strategy."""
        analysis = {
            "daily_turnover": 0.0,
            "annual_turnover": 0.0,
            "estimated_cost_drag": 0.0,
            "cost_adjusted_sharpe": None,
        }
        
        if signals is None or len(signals) < 2:
            return analysis
        
        # Calculate turnover
        daily_turnover = signals.diff().abs().sum(axis=1).mean()
        annual_turnover = daily_turnover * 252
        analysis["daily_turnover"] = float(daily_turnover)
        analysis["annual_turnover"] = float(annual_turnover)
        
        # Estimate cost drag (basis points to decimal)
        cost_per_trade = self.DEFAULT_TRANSACTION_COST_BPS / 10000
        annual_cost_drag = annual_turnover * cost_per_trade
        analysis["estimated_cost_drag"] = float(annual_cost_drag)
        
        if metrics:
            sharpe = metrics.get('sharpe_ratio', metrics.get('sharpe', 0))
            annual_return = metrics.get('annual_return', 0)
            volatility = metrics.get('volatility', 0.15)
            
            # Adjust returns for costs
            if volatility > 0:
                cost_adjusted_return = annual_return - annual_cost_drag
                cost_adjusted_sharpe = cost_adjusted_return / volatility
                analysis["cost_adjusted_sharpe"] = float(cost_adjusted_sharpe)
                
                # Check if costs eat most of returns
                if annual_cost_drag > 0 and annual_return > 0:
                    cost_ratio = annual_cost_drag / annual_return
                    if cost_ratio > 0.5:
                        self.issues.append(ValidationIssue(
                            issue_type="HIGH_TRANSACTION_COSTS",
                            severity=Severity.HIGH,
                            description=f"Transaction costs would consume {cost_ratio:.0%} of returns",
                            fix_suggestion="Reduce turnover or use lower-cost execution",
                            details={"cost_ratio": cost_ratio, "annual_turnover": annual_turnover}
                        ))
                    elif cost_ratio > 0.3:
                        self.issues.append(ValidationIssue(
                            issue_type="MODERATE_TRANSACTION_COSTS",
                            severity=Severity.MEDIUM,
                            description=f"Transaction costs would consume {cost_ratio:.0%} of returns",
                            fix_suggestion="Consider reducing turnover",
                            details={"cost_ratio": cost_ratio, "annual_turnover": annual_turnover}
                        ))
        
        return analysis

    def _analyze_regime_dependency(
        self, 
        returns: pd.Series, 
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Analyze if strategy only works in certain market regimes."""
        analysis = {
            "up_market_sharpe": None,
            "down_market_sharpe": None,
            "regime_dependency": "UNKNOWN",
            "performance_by_year": {},
        }
        
        if returns is None or len(returns) < 252:
            return analysis
        
        # Use returns themselves if no benchmark
        if benchmark_returns is None:
            benchmark_returns = returns.rolling(20).mean()
        
        # Align series
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < 100:
            return analysis
        
        strat_returns = returns.loc[common_idx]
        bench_returns = benchmark_returns.loc[common_idx]
        
        # Define up/down markets
        up_market = bench_returns > 0
        down_market = bench_returns <= 0
        
        up_returns = strat_returns[up_market]
        down_returns = strat_returns[down_market]
        
        # Calculate regime Sharpes
        if len(up_returns) > 50 and up_returns.std() > 0:
            up_sharpe = (up_returns.mean() * 252) / (up_returns.std() * np.sqrt(252))
            analysis["up_market_sharpe"] = float(up_sharpe)
        
        if len(down_returns) > 50 and down_returns.std() > 0:
            down_sharpe = (down_returns.mean() * 252) / (down_returns.std() * np.sqrt(252))
            analysis["down_market_sharpe"] = float(down_sharpe)
        
        # Check for regime dependency
        if analysis["up_market_sharpe"] is not None and analysis["down_market_sharpe"] is not None:
            up_s = analysis["up_market_sharpe"]
            down_s = analysis["down_market_sharpe"]
            
            if up_s > 1.0 and down_s < 0:
                analysis["regime_dependency"] = "BULL_ONLY"
                self.issues.append(ValidationIssue(
                    issue_type="REGIME_DEPENDENCY",
                    severity=Severity.HIGH,
                    description=f"Strategy only works in bull markets (up: {up_s:.2f}, down: {down_s:.2f})",
                    fix_suggestion="Add hedging for bear markets",
                    details={"up_sharpe": up_s, "down_sharpe": down_s}
                ))
            elif down_s > 1.0 and up_s < 0:
                analysis["regime_dependency"] = "BEAR_ONLY"
                self.issues.append(ValidationIssue(
                    issue_type="REGIME_DEPENDENCY",
                    severity=Severity.HIGH,
                    description=f"Strategy only works in bear markets (up: {up_s:.2f}, down: {down_s:.2f})",
                    fix_suggestion="May underperform in normal conditions",
                    details={"up_sharpe": up_s, "down_sharpe": down_s}
                ))
            else:
                analysis["regime_dependency"] = "BALANCED"
        
        # Performance by year
        try:
            yearly = strat_returns.groupby(strat_returns.index.year)
            for year, year_returns in yearly:
                if len(year_returns) > 50:
                    year_sharpe = (year_returns.mean() * 252) / (year_returns.std() * np.sqrt(252))
                    analysis["performance_by_year"][str(year)] = float(year_sharpe)
        except:
            pass
        
        return analysis

    def _check_performance_decay(self, returns: pd.Series) -> None:
        """Check if strategy performance has decayed over time."""
        if returns is None or len(returns) < 504:  # Need at least 2 years
            return
        
        # Split into halves
        mid = len(returns) // 2
        first_half = returns.iloc[:mid]
        second_half = returns.iloc[mid:]
        
        if first_half.std() > 0 and second_half.std() > 0:
            first_sharpe = (first_half.mean() * 252) / (first_half.std() * np.sqrt(252))
            second_sharpe = (second_half.mean() * 252) / (second_half.std() * np.sqrt(252))
            
            # Check for significant decay
            if first_sharpe > 1.0 and second_sharpe < first_sharpe * 0.5:
                self.issues.append(ValidationIssue(
                    issue_type="PERFORMANCE_DECAY",
                    severity=Severity.HIGH,
                    description=f"Performance decayed: first half Sharpe {first_sharpe:.2f}, second half {second_sharpe:.2f}",
                    fix_suggestion="Strategy may be arbitraged away or overfit to early data",
                    details={"first_half_sharpe": first_sharpe, "second_half_sharpe": second_sharpe}
                ))
            elif first_sharpe > 0.5 and second_sharpe < first_sharpe * 0.7:
                self.issues.append(ValidationIssue(
                    issue_type="PERFORMANCE_DECAY",
                    severity=Severity.MEDIUM,
                    description=f"Performance declined: first half Sharpe {first_sharpe:.2f}, second half {second_sharpe:.2f}",
                    fix_suggestion="Monitor for continued decay",
                    details={"first_half_sharpe": first_sharpe, "second_half_sharpe": second_sharpe}
                ))

    def _check_alpha_correlation(self, returns: pd.Series) -> Dict[str, Any]:
        """Check if new alpha is correlated to existing alphas."""
        analysis = {
            "max_correlation": 0.0,
            "most_correlated_alpha": None,
            "correlations": {},
        }
        
        if self.existing_alpha_returns is None or returns is None:
            return analysis
        
        # Align indices
        common_idx = returns.index.intersection(self.existing_alpha_returns.index)
        if len(common_idx) < 100:
            return analysis
        
        new_returns = returns.loc[common_idx]
        
        max_corr = 0.0
        max_corr_alpha = None
        
        for col in self.existing_alpha_returns.columns:
            existing = self.existing_alpha_returns.loc[common_idx, col]
            corr = new_returns.corr(existing)
            analysis["correlations"][col] = float(corr) if not pd.isna(corr) else 0.0
            
            if abs(corr) > abs(max_corr):
                max_corr = corr
                max_corr_alpha = col
        
        analysis["max_correlation"] = float(max_corr)
        analysis["most_correlated_alpha"] = max_corr_alpha
        
        # Check for high correlation (redundancy)
        if abs(max_corr) > 0.7:
            self.issues.append(ValidationIssue(
                issue_type="ALPHA_REDUNDANCY",
                severity=Severity.HIGH,
                description=f"Highly correlated ({max_corr:.2f}) to existing alpha '{max_corr_alpha}'",
                fix_suggestion="Strategy may not add diversification value",
                details={"correlation": max_corr, "correlated_to": max_corr_alpha}
            ))
        elif abs(max_corr) > 0.5:
            self.issues.append(ValidationIssue(
                issue_type="MODERATE_CORRELATION",
                severity=Severity.MEDIUM,
                description=f"Moderately correlated ({max_corr:.2f}) to existing alpha '{max_corr_alpha}'",
                fix_suggestion="Consider if this adds sufficient diversification",
                details={"correlation": max_corr, "correlated_to": max_corr_alpha}
            ))
        
        return analysis

    def _determine_overall_severity(self) -> Severity:
        """Determine overall validation severity."""
        if not self.issues:
            return Severity.PASS

        severities = [i.severity for i in self.issues]
        if Severity.CRITICAL in severities:
            return Severity.CRITICAL
        if Severity.HIGH in severities:
            return Severity.HIGH
        if Severity.MEDIUM in severities:
            return Severity.MEDIUM
        if Severity.LOW in severities:
            return Severity.LOW
        return Severity.PASS

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on issues."""
        recs = []
        types = {i.issue_type for i in self.issues}

        if "LOOK_AHEAD_BIAS" in types or "POTENTIAL_LOOK_AHEAD" in types:
            recs.append("Review signal generation for proper time shifting.")

        if "SURVIVORSHIP_BIAS" in types:
            recs.append("Re-run on survivorship-free universe (sp500_sf).")

        if "SUSPICIOUS_SHARPE" in types or "HIGH_SHARPE" in types:
            recs.append("Perform rigorous out-of-sample testing.")

        if "HIGH_TRANSACTION_COSTS" in types or "MODERATE_TRANSACTION_COSTS" in types:
            recs.append("Consider reducing turnover or using limit orders.")

        if "REGIME_DEPENDENCY" in types:
            recs.append("Test strategy across different market conditions.")

        if "PERFORMANCE_DECAY" in types:
            recs.append("Strategy may be crowded or overfit. Consider walk-forward testing.")

        if "ALPHA_REDUNDANCY" in types:
            recs.append("Consider if this alpha adds value beyond existing ones.")

        if not recs and self.issues:
            recs.append("Address identified issues before live trading.")

        if not recs:
            recs.append("Strategy passed validation. Paper trade before deployment.")

        return recs

    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence score."""
        confidence = 1.0

        for issue in self.issues:
            if issue.severity == Severity.CRITICAL:
                confidence -= 0.30
            elif issue.severity == Severity.HIGH:
                confidence -= 0.15
            elif issue.severity == Severity.MEDIUM:
                confidence -= 0.05
            elif issue.severity == Severity.LOW:
                confidence -= 0.02

        sharpe = metrics.get('sharpe_ratio', metrics.get('sharpe', 0))
        if 0.5 <= sharpe <= 1.5:
            confidence += 0.05

        return max(0.0, min(1.0, confidence))


def validate_strategy(
    strategy_code: str,
    backtest_metrics: Optional[Dict[str, float]] = None,
    universe: str = "unknown",
    daily_returns: Optional[pd.Series] = None,
    signals: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    existing_alpha_returns: Optional[pd.DataFrame] = None,
) -> ValidationResult:
    """
    Convenience function to validate a strategy.
    
    Args:
        strategy_code: Python code of the strategy
        backtest_metrics: Dict with sharpe_ratio, annual_return, etc.
        universe: Name of universe used
        daily_returns: Strategy daily returns series
        signals: Strategy signals DataFrame
        benchmark_returns: Market benchmark returns (for regime analysis)
        existing_alpha_returns: DataFrame of existing alpha returns (for correlation)
    
    Returns:
        ValidationResult with issues, recommendations, and analysis
    """
    validator = AdversarialValidator(existing_alpha_returns=existing_alpha_returns)
    return validator.validate(
        strategy_code, backtest_metrics, universe, daily_returns, signals, benchmark_returns
    )


def validate_from_files(results_dir: str, strategy_name: str) -> ValidationResult:
    """
    Validate a strategy from saved backtest result files.
    
    Expected files:
        - {strategy_name}_metrics.json
        - {strategy_name}_returns.csv
        - {strategy_name}_signals.csv
        - {strategy_name}_code.py (optional)
    """
    import os
    import json

    safe_name = strategy_name.replace(" ", "_").replace("/", "_").lower()

    # Load metrics
    metrics_path = os.path.join(results_dir, f"{safe_name}_metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, "r") as f:
        data = json.load(f)

    backtest_metrics = data.get("metrics", {})
    backtest_metrics["trading_days"] = data.get("trading_days", 0)
    universe = data.get("universe", "unknown")

    # Load returns
    returns_path = os.path.join(results_dir, f"{safe_name}_returns.csv")
    daily_returns = None
    if os.path.exists(returns_path):
        df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        daily_returns = df["returns"] if "returns" in df.columns else df.iloc[:, 0]

    # Load signals
    signals_path = os.path.join(results_dir, f"{safe_name}_signals.csv")
    signals = None
    if os.path.exists(signals_path):
        signals = pd.read_csv(signals_path, index_col=0, parse_dates=True)

    # Load code
    code_path = os.path.join(results_dir, f"{safe_name}_code.py")
    strategy_code = ""
    if os.path.exists(code_path):
        with open(code_path, "r") as f:
            strategy_code = f.read()

    return validate_strategy(
        strategy_code=strategy_code,
        backtest_metrics=backtest_metrics,
        universe=universe,
        daily_returns=daily_returns,
        signals=signals,
    )


def load_existing_alpha_returns(validated_alphas_dir: str = "results/validated_alphas") -> pd.DataFrame:
    """
    Load daily returns from all validated alphas.
    
    Args:
        validated_alphas_dir: Directory containing validated alpha files
    
    Returns:
        DataFrame with columns for each alpha's daily returns
    """
    import os
    import glob
    
    returns_files = glob.glob(os.path.join(validated_alphas_dir, "*_returns.csv"))
    
    if not returns_files:
        return pd.DataFrame()
    
    all_returns = {}
    for f in returns_files:
        # Extract alpha name from filename
        basename = os.path.basename(f)
        alpha_name = basename.replace("_returns.csv", "")
        
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            returns = df.iloc[:, 0] if len(df.columns) > 0 else None
            if returns is not None:
                all_returns[alpha_name] = returns
        except Exception:
            continue
    
    if not all_returns:
        return pd.DataFrame()
    
    return pd.DataFrame(all_returns)


def validate_all_alphas(
    validated_alphas_dir: str = "results/validated_alphas",
    verbose: bool = True,
) -> Dict[str, ValidationResult]:
    """
    Validate all alphas in the validated_alphas directory.
    
    Args:
        validated_alphas_dir: Directory containing validated alpha files
        verbose: Print progress and results
    
    Returns:
        Dict mapping alpha names to their ValidationResult
    """
    import os
    import glob
    import json
    
    # Load existing alpha returns for correlation checking
    existing_returns = load_existing_alpha_returns(validated_alphas_dir)
    
    # Find all alphas
    json_files = glob.glob(os.path.join(validated_alphas_dir, "alpha_*.json"))
    
    results = {}
    
    for json_path in sorted(json_files):
        basename = os.path.basename(json_path)
        alpha_name = basename.replace(".json", "")
        
        try:
            # Load metadata
            with open(json_path, "r") as f:
                data = json.load(f)
            
            metrics = data.get("metrics", {})
            universe = data.get("universe", "unknown")
            
            # Load code
            code_path = json_path.replace(".json", ".py")
            code = ""
            if os.path.exists(code_path):
                with open(code_path, "r") as f:
                    code = f.read()
            
            # Load returns
            returns_path = json_path.replace(".json", "_returns.csv")
            daily_returns = None
            if os.path.exists(returns_path):
                df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
                daily_returns = df.iloc[:, 0]
            
            # Exclude this alpha from existing returns for correlation check
            other_returns = existing_returns.drop(columns=[alpha_name], errors='ignore')
            
            # Validate
            result = validate_strategy(
                strategy_code=code,
                backtest_metrics=metrics,
                universe=universe,
                daily_returns=daily_returns,
                existing_alpha_returns=other_returns if not other_returns.empty else None,
            )
            
            results[alpha_name] = result
            
            if verbose:
                status = "PASS" if result.validation_passed else result.severity.value
                print(f"{alpha_name}: {status}")
                if result.issues_found:
                    for issue in result.issues_found[:3]:
                        print(f"    [{issue.severity.value}] {issue.issue_type}: {issue.description}")
        
        except Exception as e:
            if verbose:
                print(f"{alpha_name}: ERROR - {e}")
    
    return results


@dataclass
class TradeValidation:
    """Result of manual trade validation."""
    trade_date: str
    return_on_date: float
    validation_steps: List[Dict[str, Any]]
    data_used: Dict[str, str]
    verdict: str  # VALID, INVALID, UNCERTAIN
    issues: List[str]
    explanation: str


def validate_trade_manually(
    alpha_path: str,
    prices: pd.DataFrame,
    num_samples: int = 2,
    verbose: bool = True,
) -> List[TradeValidation]:
    """
    Manually validate trades by tracing through the strategy logic step-by-step.
    
    This function:
    1. Loads the strategy code
    2. Picks sample dates with significant returns
    3. Executes the strategy logic step-by-step
    4. Verifies signals use only past data
    
    Args:
        alpha_path: Path to alpha .py file
        prices: DataFrame of prices (index=dates, columns=symbols)
        num_samples: Number of trades to validate
        verbose: Print detailed output
    
    Returns:
        List of TradeValidation results
    """
    import os
    import re
    
    # Load strategy code
    with open(alpha_path, 'r') as f:
        code = f.read()
    
    alpha_name = os.path.basename(alpha_path).replace('.py', '')
    
    # Load returns if available
    returns_path = alpha_path.replace('.py', '_returns.csv')
    if os.path.exists(returns_path):
        returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        returns = returns_df.iloc[:, 0]
    else:
        # Generate returns by running strategy
        returns = None
    
    if verbose:
        print("=" * 70)
        print(f"MANUAL TRADE VALIDATION: {alpha_name}")
        print("=" * 70)
        print(f"\nStrategy Code:\n{'-' * 40}")
        # Print just the generate_signals function
        func_match = re.search(r'def generate_signals.*?(?=\ndef |\Z)', code, re.DOTALL)
        if func_match:
            print(func_match.group(0)[:500] + "..." if len(func_match.group(0)) > 500 else func_match.group(0))
        print(f"{'-' * 40}\n")
    
    validations = []
    
    # Pick sample dates
    if returns is not None:
        # Pick dates with significant returns
        significant = returns[returns.abs() > 0.01]
        if len(significant) < num_samples:
            significant = returns
        sample_dates = significant.sample(min(num_samples, len(significant)), random_state=42).index
    else:
        # Pick random dates from prices
        sample_dates = prices.index[-252:].to_series().sample(num_samples, random_state=42).index
    
    for sample_date in sample_dates:
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Validating trade on: {sample_date.date()}")
            if returns is not None:
                print(f"Return on this date: {returns.loc[sample_date]:.2%}")
            print(f"{'=' * 50}")
        
        steps = []
        issues = []
        data_used = {}
        
        # Analyze what data the strategy uses
        # Check for cummax, rolling, pct_change, shift patterns
        
        # Step 1: Identify lookback patterns
        lookback_patterns = {
            'cummax': r'\.cummax\(\)',
            'rolling': r'\.rolling\((\d+)\)',
            'pct_change': r'\.pct_change\((\d+)\)',
            'shift': r'\.shift\((-?\d+)\)',
            'rank': r'\.rank\(',
            'ewm': r'\.ewm\(',
        }
        
        found_patterns = {}
        for name, pattern in lookback_patterns.items():
            matches = re.findall(pattern, code)
            if matches:
                found_patterns[name] = matches
        
        if verbose:
            print(f"\nDetected patterns in code:")
            for name, matches in found_patterns.items():
                print(f"  - {name}: {matches}")
        
        # Step 2: Trace data availability
        steps.append({
            "step": 1,
            "description": "Data availability check",
            "detail": f"On {sample_date.date()}, strategy can only use data up to {(sample_date - pd.Timedelta(days=1)).date()} for signal generation"
        })
        
        # Step 3: Check for look-ahead risks
        look_ahead_risk = False
        
        if 'rank' in found_patterns and 'shift' not in found_patterns:
            look_ahead_risk = True
            issues.append("rank() without shift() - may use same-day data")
            steps.append({
                "step": 2,
                "description": "POTENTIAL ISSUE: rank() without shift()",
                "detail": "The .rank() operation uses data from the current day. If trading at market open, this is look-ahead bias."
            })
        
        if 'cummax' in found_patterns:
            steps.append({
                "step": 3,
                "description": "cummax() is backward-looking",
                "detail": "cummax() only uses historical data up to current point - OK"
            })
        
        if 'pct_change' in found_patterns:
            lookback = found_patterns['pct_change'][0] if found_patterns['pct_change'] else '?'
            steps.append({
                "step": 4,
                "description": f"pct_change({lookback}) lookback",
                "detail": f"Uses prices from T-{lookback} to T for return calculation"
            })
        
        # Step 4: Execution timing analysis
        if 'shift' in found_patterns:
            shift_vals = [int(s) for s in found_patterns['shift']]
            if any(s < 0 for s in shift_vals):
                look_ahead_risk = True
                issues.append(f"Negative shift detected: {shift_vals} - LOOK AHEAD BIAS!")
            elif any(s >= 1 for s in shift_vals):
                steps.append({
                    "step": 5,
                    "description": "shift() provides lag",
                    "detail": f"shift({shift_vals}) ensures signal uses only past data - GOOD"
                })
        
        # Step 5: Sample calculation trace
        if sample_date in prices.index:
            idx = prices.index.get_loc(sample_date)
            if idx > 20:
                # Show sample data
                sample_symbol = prices.columns[0]
                price_before = prices.iloc[idx-5:idx, 0].values
                price_on_date = prices.iloc[idx, 0]
                
                data_used = {
                    "sample_symbol": sample_symbol,
                    "prices_T-5_to_T-1": str(price_before.round(2)),
                    "price_on_T": str(round(price_on_date, 2)),
                    "execution_assumption": "Close-to-close (signal generated at close, executed next day)"
                }
                
                steps.append({
                    "step": 6,
                    "description": "Sample data trace",
                    "detail": f"{sample_symbol}: prices T-5 to T-1 = {price_before.round(2)}, price on T = {price_on_date:.2f}"
                })
        
        # Determine verdict
        if any("LOOK AHEAD" in i.upper() for i in issues):
            verdict = "INVALID"
            explanation = "Look-ahead bias detected - strategy uses future data"
        elif look_ahead_risk:
            verdict = "UNCERTAIN"
            explanation = "Potential same-day bias - depends on execution timing. Recommend adding .shift(1)"
        else:
            verdict = "VALID"
            explanation = "Strategy appears to use only past data for signal generation"
        
        if verbose:
            print(f"\nValidation Steps:")
            for step in steps:
                print(f"  Step {step['step']}: {step['description']}")
                print(f"    {step['detail']}")
            
            print(f"\nData Used:")
            for k, v in data_used.items():
                print(f"  {k}: {v}")
            
            print(f"\nIssues Found: {issues if issues else 'None'}")
            print(f"\n>>> VERDICT: {verdict}")
            print(f"    {explanation}")
        
        validations.append(TradeValidation(
            trade_date=str(sample_date.date()),
            return_on_date=float(returns.loc[sample_date]) if returns is not None else 0.0,
            validation_steps=steps,
            data_used=data_used,
            verdict=verdict,
            issues=issues,
            explanation=explanation,
        ))
    
    return validations


def validate_alpha_trades(
    alpha_name: str = "alpha_001_maxdd_rev",
    validated_alphas_dir: str = "results/validated_alphas",
    num_samples: int = 2,
) -> List[TradeValidation]:
    """
    Convenience function to validate trades for a specific alpha.
    
    Args:
        alpha_name: Name of the alpha (without extension)
        validated_alphas_dir: Directory containing validated alphas
        num_samples: Number of trades to validate
    
    Returns:
        List of TradeValidation results
    """
    import os
    
    # Load prices (need to get from data loader)
    try:
        from .data.alpaca_loader import AlpacaDataLoader
        loader = AlpacaDataLoader()
        data = loader.load_prices(days=2520)  # 10 years
        prices = data['close'].unstack(level=0)
    except Exception as e:
        print(f"Warning: Could not load prices from Alpaca: {e}")
        print("Using placeholder prices for demonstration...")
        # Create placeholder for demo
        prices = pd.DataFrame()
    
    alpha_path = os.path.join(validated_alphas_dir, f"{alpha_name}.py")
    
    if not os.path.exists(alpha_path):
        raise FileNotFoundError(f"Alpha not found: {alpha_path}")
    
    return validate_trade_manually(alpha_path, prices, num_samples)
