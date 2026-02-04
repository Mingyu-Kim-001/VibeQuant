"""
Strategy Memory System
Persistent storage for learned strategies, hypotheses, and backtest results.
Critical for avoiding repeated failures and building on successes.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib


class StrategyStatus(str, Enum):
    """Status of a strategy in the memory system."""
    PROPOSED = "proposed"       # Hypothesis generated, not yet coded
    CODED = "coded"             # Code written, not yet backtested
    BACKTESTING = "backtesting" # Currently being backtested
    PASSED = "passed"           # Passed backtest criteria
    FAILED = "failed"           # Failed backtest or validation
    DEPLOYED = "deployed"       # In production/paper trading
    RETIRED = "retired"         # Previously deployed, now retired


@dataclass
class BacktestResult:
    """Results from a strategy backtest."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    start_date: str
    end_date: str
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    correlation_to_spy: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_trade_return: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HypothesisRecord:
    """Record of a hypothesis proposed by the Insight Agent."""
    id: str
    hypothesis: str
    rationale: str
    category: str  # momentum, mean_reversion, factor, event, ml, etc.
    created_at: str
    status: str = "proposed"
    related_strategies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    priority_score: float = 0.5  # 0-1, higher = more promising
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyRecord:
    """Complete record of a strategy including code and results."""
    id: str
    name: str
    hypothesis_id: str
    description: str
    category: str
    status: str
    code: str
    created_at: str
    updated_at: str
    version: int = 1
    backtest_results: Optional[Dict] = None
    feedback: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    parent_strategy_id: Optional[str] = None  # For iterative improvements


class StrategyMemory:
    """
    Persistent memory system for the VibeQuant multi-agent system.

    Responsibilities:
    1. Store and retrieve strategy records
    2. Track hypotheses and their outcomes
    3. Maintain learned patterns (what works, what doesn't)
    4. Provide context for agent decision-making
    5. Support iterative improvement of strategies
    """

    def __init__(self, memory_dir: Optional[str] = None):
        """
        Initialize the memory system.

        Args:
            memory_dir: Directory to store memory files
        """
        self.memory_dir = Path(memory_dir or "./memory")
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.strategies_file = self.memory_dir / "strategies.json"
        self.hypotheses_file = self.memory_dir / "hypotheses.json"
        self.learnings_file = self.memory_dir / "learnings.json"
        self.metrics_file = self.memory_dir / "metrics.json"

        # In-memory caches
        self._strategies: Dict[str, StrategyRecord] = {}
        self._hypotheses: Dict[str, HypothesisRecord] = {}
        self._learnings: Dict[str, List[str]] = {
            "successful_patterns": [],
            "failed_patterns": [],
            "market_insights": [],
            "technical_notes": [],
        }
        self._metrics: Dict[str, Any] = {
            "total_hypotheses": 0,
            "total_strategies": 0,
            "successful_strategies": 0,
            "failed_strategies": 0,
            "best_sharpe": 0.0,
            "best_return": 0.0,
        }

        # Load existing data
        self._load_all()

    def _load_all(self) -> None:
        """Load all memory from disk."""
        self._load_strategies()
        self._load_hypotheses()
        self._load_learnings()
        self._load_metrics()

    def _load_strategies(self) -> None:
        """Load strategies from disk."""
        if self.strategies_file.exists():
            with open(self.strategies_file, "r") as f:
                data = json.load(f)
                for record in data:
                    self._strategies[record["id"]] = StrategyRecord(**record)

    def _load_hypotheses(self) -> None:
        """Load hypotheses from disk."""
        if self.hypotheses_file.exists():
            with open(self.hypotheses_file, "r") as f:
                data = json.load(f)
                for record in data:
                    self._hypotheses[record["id"]] = HypothesisRecord(**record)

    def _load_learnings(self) -> None:
        """Load learnings from disk."""
        if self.learnings_file.exists():
            with open(self.learnings_file, "r") as f:
                self._learnings = json.load(f)

    def _load_metrics(self) -> None:
        """Load metrics from disk."""
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                self._metrics = json.load(f)

    def _save_strategies(self) -> None:
        """Save strategies to disk."""
        data = [asdict(s) for s in self._strategies.values()]
        with open(self.strategies_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _save_hypotheses(self) -> None:
        """Save hypotheses to disk."""
        data = [asdict(h) for h in self._hypotheses.values()]
        with open(self.hypotheses_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _save_learnings(self) -> None:
        """Save learnings to disk."""
        with open(self.learnings_file, "w") as f:
            json.dump(self._learnings, f, indent=2)

    def _save_metrics(self) -> None:
        """Save metrics to disk."""
        with open(self.metrics_file, "w") as f:
            json.dump(self._metrics, f, indent=2)

    def _generate_id(self, prefix: str, content: str) -> str:
        """Generate a unique ID based on content."""
        hash_input = f"{content}_{datetime.now().isoformat()}"
        hash_str = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{prefix}_{hash_str}"

    # =========================================================================
    # HYPOTHESIS MANAGEMENT
    # =========================================================================

    def add_hypothesis(
        self,
        hypothesis: str,
        rationale: str,
        category: str,
        tags: Optional[List[str]] = None,
        priority_score: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> HypothesisRecord:
        """
        Add a new hypothesis to memory.

        Args:
            hypothesis: The hypothesis statement
            rationale: Reasoning behind the hypothesis
            category: Category (momentum, mean_reversion, factor, etc.)
            tags: Optional tags for categorization
            priority_score: Priority score (0-1)
            metadata: Additional metadata

        Returns:
            The created HypothesisRecord
        """
        hyp_id = self._generate_id("HYP", hypothesis)

        record = HypothesisRecord(
            id=hyp_id,
            hypothesis=hypothesis,
            rationale=rationale,
            category=category,
            created_at=datetime.now().isoformat(),
            tags=tags or [],
            priority_score=priority_score,
            metadata=metadata or {},
        )

        self._hypotheses[hyp_id] = record
        self._metrics["total_hypotheses"] += 1
        self._save_hypotheses()
        self._save_metrics()

        return record

    def get_hypothesis(self, hypothesis_id: str) -> Optional[HypothesisRecord]:
        """Get a hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)

    def get_untested_hypotheses(
        self,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[HypothesisRecord]:
        """Get hypotheses that haven't been converted to strategies."""
        results = [
            h for h in self._hypotheses.values()
            if h.status == "proposed" and (category is None or h.category == category)
        ]
        # Sort by priority score (highest first)
        results.sort(key=lambda x: x.priority_score, reverse=True)
        return results[:limit]

    def update_hypothesis_status(
        self,
        hypothesis_id: str,
        status: str,
        strategy_id: Optional[str] = None,
    ) -> None:
        """Update the status of a hypothesis."""
        if hypothesis_id in self._hypotheses:
            self._hypotheses[hypothesis_id].status = status
            if strategy_id:
                self._hypotheses[hypothesis_id].related_strategies.append(strategy_id)
            self._save_hypotheses()

    # =========================================================================
    # STRATEGY MANAGEMENT
    # =========================================================================

    def add_strategy(
        self,
        name: str,
        hypothesis_id: str,
        description: str,
        category: str,
        code: str,
        parameters: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        parent_strategy_id: Optional[str] = None,
    ) -> StrategyRecord:
        """
        Add a new strategy to memory.

        Args:
            name: Strategy name
            hypothesis_id: ID of the source hypothesis
            description: Strategy description
            category: Strategy category
            code: The Python code implementing the strategy
            parameters: Strategy parameters
            tags: Optional tags
            parent_strategy_id: ID of parent strategy if this is an improvement

        Returns:
            The created StrategyRecord
        """
        strategy_id = self._generate_id("STR", name)
        now = datetime.now().isoformat()

        # Determine version
        version = 1
        if parent_strategy_id and parent_strategy_id in self._strategies:
            version = self._strategies[parent_strategy_id].version + 1

        record = StrategyRecord(
            id=strategy_id,
            name=name,
            hypothesis_id=hypothesis_id,
            description=description,
            category=category,
            status=StrategyStatus.CODED.value,
            code=code,
            created_at=now,
            updated_at=now,
            version=version,
            parameters=parameters or {},
            tags=tags or [],
            parent_strategy_id=parent_strategy_id,
        )

        self._strategies[strategy_id] = record
        self._metrics["total_strategies"] += 1
        self.update_hypothesis_status(hypothesis_id, "coded", strategy_id)
        self._save_strategies()
        self._save_metrics()

        return record

    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Get a strategy by ID."""
        return self._strategies.get(strategy_id)

    def get_strategies_by_status(
        self,
        status: str,
        limit: int = 50,
    ) -> List[StrategyRecord]:
        """Get strategies filtered by status."""
        results = [s for s in self._strategies.values() if s.status == status]
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results[:limit]

    def get_successful_strategies(
        self,
        min_sharpe: float = 1.0,
        limit: int = 20,
    ) -> List[StrategyRecord]:
        """Get strategies that passed backtesting with good metrics."""
        results = []
        for s in self._strategies.values():
            if s.status == StrategyStatus.PASSED.value and s.backtest_results:
                sharpe = s.backtest_results.get("sharpe_ratio", 0)
                if sharpe >= min_sharpe:
                    results.append(s)

        results.sort(
            key=lambda x: x.backtest_results.get("sharpe_ratio", 0),
            reverse=True,
        )
        return results[:limit]

    def get_failed_strategies(
        self,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> List[StrategyRecord]:
        """Get failed strategies to learn from mistakes."""
        results = [
            s for s in self._strategies.values()
            if s.status == StrategyStatus.FAILED.value
            and (category is None or s.category == category)
        ]
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results[:limit]

    def update_strategy_backtest(
        self,
        strategy_id: str,
        backtest_results: BacktestResult,
        passed: bool,
        failure_reasons: Optional[List[str]] = None,
        improvement_suggestions: Optional[List[str]] = None,
    ) -> None:
        """
        Update a strategy with backtest results.

        Args:
            strategy_id: Strategy ID
            backtest_results: Results from backtest
            passed: Whether the strategy passed criteria
            failure_reasons: Reasons for failure if applicable
            improvement_suggestions: Suggestions for improvement
        """
        if strategy_id not in self._strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        strategy = self._strategies[strategy_id]
        strategy.backtest_results = asdict(backtest_results)
        strategy.updated_at = datetime.now().isoformat()

        if passed:
            strategy.status = StrategyStatus.PASSED.value
            self._metrics["successful_strategies"] += 1

            # Update best metrics
            if backtest_results.sharpe_ratio > self._metrics["best_sharpe"]:
                self._metrics["best_sharpe"] = backtest_results.sharpe_ratio
            if backtest_results.annual_return > self._metrics["best_return"]:
                self._metrics["best_return"] = backtest_results.annual_return
        else:
            strategy.status = StrategyStatus.FAILED.value
            strategy.failure_reasons = failure_reasons or []
            self._metrics["failed_strategies"] += 1

        strategy.improvement_suggestions = improvement_suggestions or []

        # Update hypothesis status
        self.update_hypothesis_status(
            strategy.hypothesis_id,
            "passed" if passed else "failed",
        )

        self._save_strategies()
        self._save_metrics()

    def add_strategy_feedback(
        self,
        strategy_id: str,
        feedback: str,
    ) -> None:
        """Add feedback to a strategy."""
        if strategy_id in self._strategies:
            self._strategies[strategy_id].feedback.append(feedback)
            self._strategies[strategy_id].updated_at = datetime.now().isoformat()
            self._save_strategies()

    # =========================================================================
    # LEARNINGS MANAGEMENT
    # =========================================================================

    def add_learning(
        self,
        learning_type: Literal[
            "successful_patterns",
            "failed_patterns",
            "market_insights",
            "technical_notes",
        ],
        learning: str,
    ) -> None:
        """
        Add a learning to memory.

        Args:
            learning_type: Type of learning
            learning: The learning text
        """
        if learning_type in self._learnings:
            if learning not in self._learnings[learning_type]:
                self._learnings[learning_type].append(learning)
                self._save_learnings()

    def get_learnings(
        self,
        learning_type: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Get learnings, optionally filtered by type."""
        if learning_type:
            return {learning_type: self._learnings.get(learning_type, [])}
        return self._learnings

    # =========================================================================
    # CONTEXT GENERATION FOR AGENTS
    # =========================================================================

    def get_context_for_insight_agent(self) -> Dict[str, Any]:
        """
        Generate context for the Insight Agent.
        Includes: failed patterns to avoid, successful patterns to build on,
        untested hypotheses, and overall metrics.
        """
        return {
            "metrics": self._metrics,
            "failed_patterns": self._learnings["failed_patterns"][-20:],
            "successful_patterns": self._learnings["successful_patterns"][-20:],
            "recent_failures": [
                {
                    "name": s.name,
                    "category": s.category,
                    "hypothesis": self._hypotheses.get(s.hypothesis_id, {}).hypothesis
                    if s.hypothesis_id in self._hypotheses else "Unknown",
                    "reasons": s.failure_reasons,
                }
                for s in self.get_failed_strategies(limit=10)
            ],
            "successful_strategies": [
                {
                    "name": s.name,
                    "category": s.category,
                    "sharpe": s.backtest_results.get("sharpe_ratio", 0),
                    "return": s.backtest_results.get("annual_return", 0),
                }
                for s in self.get_successful_strategies(limit=5)
            ],
            "untested_categories": self._get_untested_categories(),
        }

    def get_context_for_research_agent(
        self,
        hypothesis_id: str,
    ) -> Dict[str, Any]:
        """
        Generate context for the Research Agent.
        Includes: similar strategies, successful code patterns, and technical notes.
        """
        hypothesis = self.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return {}

        similar_strategies = [
            s for s in self._strategies.values()
            if s.category == hypothesis.category
        ]

        return {
            "hypothesis": asdict(hypothesis),
            "similar_strategies": [
                {"name": s.name, "code": s.code, "results": s.backtest_results}
                for s in similar_strategies[:3]
            ],
            "technical_notes": self._learnings["technical_notes"][-10:],
            "successful_patterns": self._learnings["successful_patterns"][-10:],
        }

    def get_context_for_feedback_agent(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """
        Generate context for the Feedback Agent.
        Includes: strategy details, backtest results, and comparison benchmarks.
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return {}

        # Get benchmark metrics from successful strategies
        successful = self.get_successful_strategies()
        avg_sharpe = (
            sum(s.backtest_results.get("sharpe_ratio", 0) for s in successful) /
            len(successful) if successful else 0
        )

        return {
            "strategy": asdict(strategy),
            "benchmark_metrics": {
                "avg_sharpe_of_successful": avg_sharpe,
                "best_sharpe": self._metrics["best_sharpe"],
                "best_return": self._metrics["best_return"],
            },
            "passing_criteria": {
                "min_sharpe": 0.5,
                "max_drawdown": -0.35,
                "min_trades": 20,
                "min_profit_factor": 1.0,
            },
        }

    def _get_untested_categories(self) -> List[str]:
        """Get categories with few tested strategies."""
        categories = [
            "momentum", "mean_reversion", "factor", "event",
            "ml", "volatility", "pairs", "seasonal",
        ]

        category_counts = {}
        for s in self._strategies.values():
            category_counts[s.category] = category_counts.get(s.category, 0) + 1

        return [c for c in categories if category_counts.get(c, 0) < 3]

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_summary_report(self) -> str:
        """Generate a summary report of the memory state."""
        report = []
        report.append("=" * 60)
        report.append("VIBEQUANT MEMORY SUMMARY")
        report.append("=" * 60)

        report.append(f"\nTotal Hypotheses: {self._metrics['total_hypotheses']}")
        report.append(f"Total Strategies: {self._metrics['total_strategies']}")
        report.append(f"Successful: {self._metrics['successful_strategies']}")
        report.append(f"Failed: {self._metrics['failed_strategies']}")

        if self._metrics['total_strategies'] > 0:
            success_rate = (
                self._metrics['successful_strategies'] /
                self._metrics['total_strategies'] * 100
            )
            report.append(f"Success Rate: {success_rate:.1f}%")

        report.append(f"\nBest Sharpe: {self._metrics['best_sharpe']:.2f}")
        report.append(f"Best Annual Return: {self._metrics['best_return']:.2%}")

        # Top strategies
        top_strategies = self.get_successful_strategies(limit=5)
        if top_strategies:
            report.append("\n--- TOP STRATEGIES ---")
            for i, s in enumerate(top_strategies, 1):
                sharpe = s.backtest_results.get("sharpe_ratio", 0)
                ret = s.backtest_results.get("annual_return", 0)
                report.append(f"{i}. {s.name} (Sharpe: {sharpe:.2f}, Return: {ret:.1%})")

        # Recent learnings
        if self._learnings["successful_patterns"]:
            report.append("\n--- RECENT SUCCESSFUL PATTERNS ---")
            for pattern in self._learnings["successful_patterns"][-3:]:
                report.append(f"  - {pattern}")

        if self._learnings["failed_patterns"]:
            report.append("\n--- RECENT FAILED PATTERNS ---")
            for pattern in self._learnings["failed_patterns"][-3:]:
                report.append(f"  - {pattern}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    memory = StrategyMemory(memory_dir="./test_memory")

    # Add a hypothesis
    hyp = memory.add_hypothesis(
        hypothesis="Price momentum over 20 days predicts future returns",
        rationale="Stocks with positive momentum tend to continue trending",
        category="momentum",
        tags=["momentum", "trend-following"],
        priority_score=0.7,
    )
    print(f"Created hypothesis: {hyp.id}")

    # Add a strategy
    strategy = memory.add_strategy(
        name="Simple Momentum Strategy",
        hypothesis_id=hyp.id,
        description="Buy stocks with positive 20-day momentum",
        category="momentum",
        code="def signal(prices): return prices.pct_change(20) > 0",
        parameters={"lookback": 20},
    )
    print(f"Created strategy: {strategy.id}")

    # Add backtest results
    results = BacktestResult(
        total_return=0.25,
        annual_return=0.18,
        sharpe_ratio=1.2,
        max_drawdown=-0.15,
        win_rate=0.55,
        num_trades=100,
        start_date="2023-01-01",
        end_date="2024-01-01",
    )
    memory.update_strategy_backtest(strategy.id, results, passed=True)

    # Add a learning
    memory.add_learning(
        "successful_patterns",
        "20-day momentum works well in trending markets",
    )

    # Print summary
    print(memory.get_summary_report())
