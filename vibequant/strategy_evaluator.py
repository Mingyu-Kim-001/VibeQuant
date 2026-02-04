"""
Strategy Evaluator - Utility functions for deterministic evaluation.

This module wraps existing evaluation logic from agents/base.py and provides
file I/O utilities. It does NOT change any evaluation thresholds or logic.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Re-export from base.py - single source of truth for criteria
from .agents.base import PASSING_CRITERIA, check_passing_criteria


@dataclass
class EvaluationResult:
    """Result of strategy evaluation."""
    verdict: str  # PASS, FAIL, RETEST, SUSPICIOUS
    passed: bool
    score: float
    passed_criteria: Dict[str, bool]
    failure_reasons: List[str]
    warnings: List[str]


def evaluate_backtest_metrics(
    metrics: Dict[str, float],
    universe: str = "unknown",
    survivorship_free: bool = False,
) -> EvaluationResult:
    """
    Evaluate backtest metrics against passing criteria.
    
    Uses the same criteria as agents/base.py check_passing_criteria().
    
    Args:
        metrics: Backtest metrics dict
        universe: Universe used for backtest
        survivorship_free: Whether universe is survivorship-free
    
    Returns:
        EvaluationResult with verdict and details
    """
    sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe", 0))
    
    # Check survivorship-free requirement
    is_sf = survivorship_free or universe.lower() in ["sp500_sf", "dynamic", "etfs"]
    if not is_sf:
        return EvaluationResult(
            verdict="RETEST",
            passed=False,
            score=0,
            passed_criteria={"survivorship_free": False},
            failure_reasons=[f"Must test on survivorship-free universe, got '{universe}'"],
            warnings=[],
        )
    
    # Check for suspicious results (same as prompts)
    if sharpe > 3.0:
        return EvaluationResult(
            verdict="SUSPICIOUS",
            passed=False,
            score=0,
            passed_criteria={"sharpe_reasonable": False},
            failure_reasons=[f"Sharpe {sharpe:.2f} > 3.0 - likely overfit or bug"],
            warnings=[],
        )
    
    # Use existing criteria check from base.py
    passed, failure_reasons = check_passing_criteria(metrics)
    
    warnings = []
    if 2.5 < sharpe <= 3.0:
        warnings.append(f"Sharpe {sharpe:.2f} is unusually high - verify with out-of-sample data")
    
    score = min(10, sharpe * 5) if sharpe > 0 else 0
    
    return EvaluationResult(
        verdict="PASS" if passed else "FAIL",
        passed=passed,
        score=score,
        passed_criteria={
            "sharpe": sharpe >= PASSING_CRITERIA["min_sharpe_ratio"],
            "profit_factor": metrics.get("profit_factor", 0) >= PASSING_CRITERIA["min_profit_factor"],
            "num_trades": metrics.get("num_trades", 0) >= PASSING_CRITERIA["min_trades"],
            "survivorship_free": True,
        },
        failure_reasons=failure_reasons,
        warnings=warnings,
    )


def save_validated_alpha(
    output_dir: str,
    strategy_name: str,
    strategy_code: str,
    hypothesis: str,
    category: str,
    metrics: Dict[str, float],
    parameters: Dict[str, Any],
    universe: str = "sp500_sf",
) -> Dict[str, str]:
    """
    Save a validated alpha to disk.
    
    Args:
        output_dir: Directory to save alpha files
        strategy_name: Name of the strategy
        strategy_code: Python code
        hypothesis: Original hypothesis
        category: Strategy category
        metrics: Backtest metrics
        parameters: Strategy parameters
        universe: Universe used
    
    Returns:
        Dict with paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine next alpha ID
    next_id = 1
    if os.path.exists(output_dir):
        existing = [f for f in os.listdir(output_dir) if f.endswith('.py') and f.startswith('alpha_')]
        if existing:
            ids = [int(f.split('_')[1]) for f in existing if f.split('_')[1].isdigit()]
            next_id = max(ids) + 1 if ids else 1
    
    # Create safe filename
    safe_name = strategy_name.lower().replace(" ", "_").replace("-", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")[:30]
    
    paths = {}
    
    # Save code file
    sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe", 0))
    code_filename = f"alpha_{next_id:03d}_{safe_name}.py"
    code_path = os.path.join(output_dir, code_filename)
    
    code_content = f'''"""
Alpha: {strategy_name}
Category: {category}
Sharpe: {sharpe:.2f} ({universe}, survivorship-free)
Validated: {datetime.now().isoformat()}
Hypothesis: {hypothesis[:200]}...
"""

{strategy_code}
'''
    
    with open(code_path, "w") as f:
        f.write(code_content)
    paths["code"] = code_path
    
    # Save metadata JSON
    json_filename = f"alpha_{next_id:03d}_{safe_name}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    metadata = {
        "id": next_id,
        "name": strategy_name,
        "hypothesis": hypothesis,
        "category": category,
        "validated_at": datetime.now().isoformat(),
        "universe": universe,
        "survivorship_free": True,
        "metrics": {
            "sharpe": sharpe,
            "annual_return": metrics.get("annual_return", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "win_rate": metrics.get("win_rate", 0),
            "num_trades": metrics.get("num_trades", 0),
        },
        "parameters": parameters,
    }
    
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    paths["metadata"] = json_path
    
    return paths


def extract_learnings(
    strategy_name: str,
    category: str,
    metrics: Dict[str, float],
    passed: bool,
) -> Dict[str, List[str]]:
    """
    Extract learnings from backtest results.
    
    Args:
        strategy_name: Name of strategy
        category: Strategy category
        metrics: Backtest metrics
        passed: Whether strategy passed
    
    Returns:
        Dict with successful_patterns, failed_patterns, technical_notes
    """
    learnings = {
        "successful_patterns": [],
        "failed_patterns": [],
        "technical_notes": [],
    }
    
    sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe", 0))
    annual_return = metrics.get("annual_return", 0)
    max_drawdown = metrics.get("max_drawdown", 0)
    win_rate = metrics.get("win_rate", 0)
    profit_factor = metrics.get("profit_factor", 0)
    
    if passed:
        learnings["successful_patterns"].append(
            f"{strategy_name}: Sharpe {sharpe:.2f}, Return {annual_return:.1%}"
        )
        if win_rate > 0.55:
            learnings["successful_patterns"].append(f"{strategy_name}: Good win rate {win_rate:.1%}")
        if profit_factor > 1.5:
            learnings["successful_patterns"].append(f"{strategy_name}: Strong PF {profit_factor:.2f}")
    else:
        if sharpe < PASSING_CRITERIA["min_sharpe_ratio"]:
            learnings["failed_patterns"].append(f"{strategy_name} ({category}): Low Sharpe {sharpe:.2f}")
        if max_drawdown < -0.5:
            learnings["failed_patterns"].append(f"{strategy_name}: High DD {max_drawdown:.1%}")
    
    if abs(max_drawdown) > 0.6:
        learnings["technical_notes"].append(f"{category} may have high drawdown risk")
    
    return learnings


def get_passing_criteria() -> Dict[str, float]:
    """Return the passing criteria dict from base.py."""
    return PASSING_CRITERIA.copy()
