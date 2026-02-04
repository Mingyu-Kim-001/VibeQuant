"""
Adversarial Validation Module for VibeQuant
Detects common backtesting mistakes, biases, and unrealistic assumptions.
"""

import re
from typing import Dict, List, Optional, Any
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


@dataclass
class ValidationResult:
    """Complete validation result for a strategy."""
    validation_passed: bool
    severity: Severity
    issues_found: List[ValidationIssue] = field(default_factory=list)
    sanity_checks: Dict[str, bool] = field(default_factory=dict)
    universe_check: Dict[str, Any] = field(default_factory=dict)
    code_analysis: Dict[str, Any] = field(default_factory=dict)
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
                    "fix_suggestion": i.fix_suggestion
                }
                for i in self.issues_found
            ],
            "sanity_checks": self.sanity_checks,
            "universe_check": self.universe_check,
            "code_analysis": self.code_analysis,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score
        }


class AdversarialValidator:
    """
    Validates backtest strategies for common issues and biases.
    
    Checks for:
    - Look-ahead bias (using future data)
    - Survivorship bias
    - Data snooping / overfitting
    - Implementation bugs
    - Unrealistic assumptions
    - Statistical validity issues
    """

    LOOK_AHEAD_PATTERNS = [
        (r'\.shift\s*\(\s*-\s*\d+', "Forward shift detected (.shift(-N))"),
        (r'iloc\s*\[\s*[a-zA-Z_]\w*\s*\+', "Forward index access (iloc[i+])"),
        (r'\b(future_|next_|tomorrow|forward_)\w+', "Variable naming suggests future data"),
    ]

    MISSING_SHIFT_PATTERNS = [
        (r'prices?\s*[><]=?\s*prices?\.rolling\([^)]+\)\.mean\(\)(?!\.shift)',
         "Rolling mean comparison without shift"),
        (r'returns?\s*[><]=?\s*returns?\.rolling\([^)]+\)\.mean\(\)(?!\.shift)',
         "Rolling returns compared without shift"),
    ]

    SURVIVORSHIP_FREE_UNIVERSES = ['sp500_sf', 'dynamic', 'etfs']

    def __init__(self):
        self.issues: List[ValidationIssue] = []

    def validate(
        self,
        strategy_code: str,
        backtest_metrics: Optional[Dict[str, float]] = None,
        universe: str = "unknown",
        daily_returns: Optional[pd.Series] = None,
        signals: Optional[pd.DataFrame] = None,
    ) -> ValidationResult:
        """Perform comprehensive adversarial validation."""
        self.issues = []

        code_analysis = self._analyze_code(strategy_code)
        universe_check = self._check_universe(universe)
        
        sanity_checks = {}
        if backtest_metrics:
            sanity_checks = self._check_metrics_sanity(backtest_metrics)

        if signals is not None:
            self._analyze_signals(signals)

        if daily_returns is not None:
            self._analyze_returns(daily_returns)

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
            recommendations=recommendations,
            confidence_score=confidence
        )

    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Perform static analysis on strategy code."""
        analysis = {
            "look_ahead_patterns": [],
            "nan_handling": "UNKNOWN",
            "signal_normalization": "UNKNOWN",
            "complexity_indicators": []
        }

        lines = code.split('\n')

        # Check for look-ahead bias patterns
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

            for pattern, desc in self.MISSING_SHIFT_PATTERNS:
                if re.search(pattern, line):
                    self.issues.append(ValidationIssue(
                        issue_type="POTENTIAL_LOOK_AHEAD",
                        severity=Severity.HIGH,
                        description=desc,
                        code_location=f"line {i}: {line.strip()[:80]}",
                        fix_suggestion="Add .shift(1) to ensure only past data is used"
                    ))

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
) -> ValidationResult:
    """Convenience function to validate a strategy."""
    return AdversarialValidator().validate(
        strategy_code, backtest_metrics, universe, daily_returns, signals
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
