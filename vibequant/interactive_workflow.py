"""
VibeQuant Interactive Workflow
Designed to work with Claude Code as the LLM brain.
Each agent has an isolated context window.
"""

import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import uuid

from dotenv import load_dotenv

from .agents.base import AgentState, check_passing_criteria, PASSING_CRITERIA
from .agents.prompts import (
    format_insight_prompt,
    format_research_prompt,
    format_feedback_prompt,
)
from .memory.strategy_memory import StrategyMemory, BacktestResult
from .data.alpaca_loader import AlpacaDataLoader
from .backtest.engine import BacktestEngine, BacktestConfig, execute_strategy_code

load_dotenv()


class InteractiveWorkflow:
    """
    Interactive workflow that uses Claude Code as the LLM brain.

    Each agent runs with isolated context - only seeing:
    1. Its specific system prompt
    2. Relevant memory context
    3. Current task input

    No conversation history is shared between agents.
    """

    def __init__(
        self,
        memory_dir: str = "./memory",
        max_iterations: int = 10,
        min_successful_strategies: int = 3,
    ):
        self.memory_dir = memory_dir
        self.max_iterations = max_iterations
        self.min_successful_strategies = min_successful_strategies

        # Initialize components
        self.memory = StrategyMemory(memory_dir)
        self.data_loader = AlpacaDataLoader()
        self.backtest_engine = BacktestEngine(BacktestConfig())

        # State tracking
        self.iteration = 0
        self.successful_strategies: List[Dict] = []
        self.failed_strategies: List[Dict] = []
        self.current_hypothesis: Optional[Dict] = None
        self.current_strategy: Optional[Dict] = None
        self.run_id = str(uuid.uuid4())[:8]

    def _print_separator(self, title: str, char: str = "=") -> None:
        """Print a section separator."""
        print(f"\n{char * 70}")
        print(f" {title}")
        print(f"{char * 70}\n")

    def _print_agent_prompt(self, agent_name: str, prompt: str, input_data: Optional[str] = None) -> None:
        """Print the agent prompt in a clear format."""
        self._print_separator(f"AGENT: {agent_name}", "=")
        print("CONTEXT FOR THIS AGENT (isolated window):")
        print("-" * 50)
        print(prompt)
        if input_data:
            print("\n" + "-" * 50)
            print("INPUT DATA:")
            print("-" * 50)
            print(input_data)
        print("\n" + "-" * 50)
        print("YOUR TASK: Respond with valid JSON as specified above.")
        print("-" * 50)

    def _get_user_response(self) -> str:
        """Get multi-line JSON response from user."""
        print("\n[Paste your JSON response, then enter 'END' on a new line]")
        lines = []
        while True:
            try:
                line = input()
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            except EOFError:
                break
        return "\n".join(lines)

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from response, handling code blocks."""
        try:
            # Clean up the response
            content = response.strip()

            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            print(f"\n[ERROR] Failed to parse JSON: {e}")
            print("Please ensure your response is valid JSON.")
            return None

    def check_termination(self) -> tuple[bool, str]:
        """Check if workflow should terminate."""
        if self.iteration >= self.max_iterations:
            return True, f"Reached maximum iterations ({self.max_iterations})"

        if len(self.successful_strategies) >= self.min_successful_strategies:
            return True, f"Found {len(self.successful_strategies)} successful strategies"

        return False, ""

    def run_orchestrator(self) -> str:
        """Orchestrator decision - simplified for interactive mode."""
        self._print_separator(f"ORCHESTRATOR - Iteration {self.iteration + 1}/{self.max_iterations}")

        print(f"Run ID: {self.run_id}")
        print(f"Successful strategies: {len(self.successful_strategies)}")
        print(f"Failed strategies: {len(self.failed_strategies)}")
        print(f"Target: {self.min_successful_strategies} successful strategies")

        # Check termination
        should_terminate, reason = self.check_termination()
        if should_terminate:
            print(f"\n[TERMINATING] {reason}")
            return "terminate"

        # Show memory summary
        print("\n--- Memory Summary ---")
        metrics = self.memory._metrics
        print(f"Total hypotheses tested: {metrics.get('total_hypotheses', 0)}")
        print(f"Total strategies: {metrics.get('total_strategies', 0)}")
        print(f"Best Sharpe achieved: {metrics.get('best_sharpe', 0):.2f}")

        print("\n[CONTINUING] Starting new iteration...")
        self.iteration += 1
        return "continue"

    def run_insight_agent(self) -> Optional[Dict]:
        """Run the Insight Agent with isolated context."""
        context = self.memory.get_context_for_insight_agent()

        prompt = format_insight_prompt(
            failed_patterns=context.get("failed_patterns", [])[-10:],
            successful_patterns=context.get("successful_patterns", [])[-10:],
            recent_failures=context.get("recent_failures", [])[-5:],
            untested_categories=context.get("untested_categories", []),
        )

        self._print_agent_prompt("INSIGHT AGENT", prompt)

        response = self._get_user_response()
        hypothesis = self._parse_json_response(response)

        if hypothesis:
            # Store in memory
            hyp_record = self.memory.add_hypothesis(
                hypothesis=hypothesis.get("hypothesis", ""),
                rationale=hypothesis.get("rationale", ""),
                category=hypothesis.get("category", "other"),
                tags=hypothesis.get("data_requirements", []),
                priority_score=hypothesis.get("priority_score", 0.5),
                metadata=hypothesis.get("expected_characteristics", {}),
            )
            hypothesis["id"] = hyp_record.id
            self.current_hypothesis = hypothesis

            print(f"\n[SUCCESS] Hypothesis recorded: {hyp_record.id}")
            return hypothesis

        return None

    def run_research_agent(self) -> Optional[Dict]:
        """Run the Research Agent with isolated context."""
        if not self.current_hypothesis:
            print("[ERROR] No hypothesis to research")
            return None

        context = self.memory.get_context_for_research_agent(
            self.current_hypothesis.get("id", "")
        )

        prompt = format_research_prompt(
            hypothesis=self.current_hypothesis,
            similar_strategies=context.get("similar_strategies", [])[:2],
            technical_notes=context.get("technical_notes", [])[-5:],
        )

        self._print_agent_prompt("RESEARCH AGENT", prompt)

        response = self._get_user_response()
        strategy_output = self._parse_json_response(response)

        if strategy_output:
            # Extract and clean code
            code = strategy_output.get("code", "")
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            # Store in memory
            strategy_record = self.memory.add_strategy(
                name=strategy_output.get("strategy_name", "Unnamed Strategy"),
                hypothesis_id=self.current_hypothesis.get("id", ""),
                description=strategy_output.get("description", ""),
                category=self.current_hypothesis.get("category", "other"),
                code=code,
                parameters=strategy_output.get("parameters", {}),
                tags=strategy_output.get("data_requirements", []),
            )

            strategy_output["id"] = strategy_record.id
            strategy_output["code"] = code
            strategy_output["hypothesis_id"] = self.current_hypothesis.get("id", "")
            self.current_strategy = strategy_output

            print(f"\n[SUCCESS] Strategy recorded: {strategy_record.id}")
            print(f"Code length: {len(code)} characters")
            return strategy_output

        return None

    def run_backtest_agent(self) -> Optional[Dict]:
        """Run backtest automatically (no LLM needed)."""
        if not self.current_strategy:
            print("[ERROR] No strategy to backtest")
            return None

        self._print_separator("BACKTEST AGENT (Automated)")

        code = self.current_strategy.get("code", "")
        strategy_name = self.current_strategy.get("strategy_name", "Unknown")

        print(f"Strategy: {strategy_name}")
        print(f"Loading market data...")

        try:
            # Load data
            symbols = self.data_loader.get_popular_etfs()[:20]
            data = self.data_loader.get_bars(
                symbols=symbols,
                timeframe="1Day",
                lookback_days=504,  # ~2 years
            )

            if data.empty:
                raise ValueError("No data loaded from Alpaca API")

            # Pivot to wide format
            prices = data["close"].unstack(level=0)
            print(f"Data shape: {prices.shape} (days x symbols)")
            print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

            # Execute strategy
            print("Generating signals...")
            signals = execute_strategy_code(code, prices)

            # Get benchmark
            benchmark_prices = prices.get("SPY")

            # Run backtest
            print("Running backtest...")
            results = self.backtest_engine.run(prices, signals, benchmark_prices)

            # Build metrics dict
            metrics = {
                "total_return": results.total_return,
                "annual_return": results.annual_return,
                "sharpe_ratio": results.sharpe_ratio,
                "sortino_ratio": results.sortino_ratio,
                "calmar_ratio": results.calmar_ratio,
                "max_drawdown": results.max_drawdown,
                "max_drawdown_duration": results.max_drawdown_duration,
                "volatility": results.volatility,
                "var_95": results.var_95,
                "cvar_95": results.cvar_95,
                "num_trades": results.num_trades,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor,
                "avg_trade_return": results.avg_trade_return,
                "best_trade": results.best_trade,
                "worst_trade": results.worst_trade,
                "alpha": results.alpha,
                "beta": results.beta,
                "correlation": results.correlation,
                "information_ratio": results.information_ratio,
                "benchmark_return": results.benchmark_return,
            }

            # Print results
            print("\n" + "=" * 50)
            print("BACKTEST RESULTS")
            print("=" * 50)
            print(f"Total Return:      {metrics['total_return']:>10.2%}")
            print(f"Annual Return:     {metrics['annual_return']:>10.2%}")
            print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
            print(f"Sortino Ratio:     {metrics['sortino_ratio']:>10.2f}")
            print(f"Max Drawdown:      {metrics['max_drawdown']:>10.2%}")
            print(f"Volatility:        {metrics['volatility']:>10.2%}")
            print(f"Number of Trades:  {metrics['num_trades']:>10d}")
            print(f"Win Rate:          {metrics['win_rate']:>10.2%}")
            print(f"Profit Factor:     {metrics['profit_factor']:>10.2f}")
            print(f"Alpha:             {metrics['alpha']:>10.4f}")
            print(f"Beta:              {metrics['beta']:>10.2f}")
            print(f"Correlation (SPY): {metrics['correlation']:>10.2f}")
            print("=" * 50)

            # Check passing criteria
            passed, failures = check_passing_criteria(metrics)
            print(f"\nPassing Criteria Check: {'PASS' if passed else 'FAIL'}")
            if failures:
                print("Failures:")
                for f in failures:
                    print(f"  - {f}")

            return {
                "success": True,
                "strategy_id": self.current_strategy.get("id", ""),
                "metrics": metrics,
                "start_date": results.start_date,
                "end_date": results.end_date,
                "passed_criteria": passed,
                "criteria_failures": failures,
            }

        except Exception as e:
            print(f"\n[ERROR] Backtest failed: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "strategy_id": self.current_strategy.get("id", ""),
                "error": str(e),
            }

    def run_feedback_agent(self, backtest_results: Dict) -> Optional[Dict]:
        """Run the Feedback Agent with isolated context."""
        if not backtest_results.get("success", False):
            # Auto-fail for execution errors
            error_msg = backtest_results.get("error", "Unknown error")
            self.memory.add_learning(
                "failed_patterns",
                f"Execution error in {self.current_strategy.get('strategy_name', 'Unknown')}: {error_msg}"
            )
            self.failed_strategies.append({
                "id": self.current_strategy.get("id", ""),
                "name": self.current_strategy.get("strategy_name", ""),
                "reason": f"Execution failed: {error_msg}",
            })
            print(f"\n[AUTO-FAIL] Strategy failed to execute: {error_msg}")
            return {"passed": False, "reason": "Execution error"}

        metrics = backtest_results.get("metrics", {})
        context = self.memory.get_context_for_feedback_agent(
            self.current_strategy.get("id", "")
        )

        prompt = format_feedback_prompt(
            strategy_name=self.current_strategy.get("strategy_name", ""),
            category=self.current_hypothesis.get("category", ""),
            hypothesis=self.current_hypothesis.get("hypothesis", ""),
            backtest_results=metrics,
            avg_sharpe_successful=context.get("benchmark_metrics", {}).get("avg_sharpe_of_successful", 0),
            best_sharpe=context.get("benchmark_metrics", {}).get("best_sharpe", 0),
            best_return=context.get("benchmark_metrics", {}).get("best_return", 0),
        )

        # Add criteria check results to prompt
        criteria_info = f"""
## AUTOMATED CRITERIA CHECK
Passed: {backtest_results.get('passed_criteria', False)}
Failures: {backtest_results.get('criteria_failures', [])}

Current Passing Criteria:
- Min Sharpe Ratio: {PASSING_CRITERIA['min_sharpe_ratio']}
- Max Drawdown: {PASSING_CRITERIA['max_drawdown']:.0%}
- Min Trades: {PASSING_CRITERIA['min_trades']}
- Min Profit Factor: {PASSING_CRITERIA['min_profit_factor']}
"""

        self._print_agent_prompt("FEEDBACK AGENT", prompt, criteria_info)

        response = self._get_user_response()
        feedback = self._parse_json_response(response)

        if feedback:
            # Use automated criteria check as ground truth
            passed = backtest_results.get("passed_criteria", False) and feedback.get("passed", False)

            # Update memory
            bt_result = BacktestResult(
                total_return=metrics.get("total_return", 0),
                annual_return=metrics.get("annual_return", 0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0),
                max_drawdown=metrics.get("max_drawdown", 0),
                win_rate=metrics.get("win_rate", 0),
                num_trades=metrics.get("num_trades", 0),
                start_date=backtest_results.get("start_date", ""),
                end_date=backtest_results.get("end_date", ""),
                alpha=metrics.get("alpha", 0),
                beta=metrics.get("beta", 0),
                correlation_to_spy=metrics.get("correlation", 0),
                sortino_ratio=metrics.get("sortino_ratio", 0),
                calmar_ratio=metrics.get("calmar_ratio", 0),
                profit_factor=metrics.get("profit_factor", 0),
            )

            failure_reasons = feedback.get("failure_reasons", [])
            if not backtest_results.get("passed_criteria", False):
                failure_reasons.extend(backtest_results.get("criteria_failures", []))

            self.memory.update_strategy_backtest(
                strategy_id=self.current_strategy.get("id", ""),
                backtest_results=bt_result,
                passed=passed,
                failure_reasons=failure_reasons,
                improvement_suggestions=feedback.get("improvement_suggestions", []),
            )

            # Store learnings
            learnings = feedback.get("learnings_for_memory", {})
            for pattern in learnings.get("successful_patterns", []):
                self.memory.add_learning("successful_patterns", pattern)
            for pattern in learnings.get("failed_patterns", []):
                self.memory.add_learning("failed_patterns", pattern)
            for note in learnings.get("technical_notes", []):
                self.memory.add_learning("technical_notes", note)

            # Track results
            if passed:
                self.successful_strategies.append({
                    "id": self.current_strategy.get("id", ""),
                    "name": self.current_strategy.get("strategy_name", ""),
                    "sharpe": metrics.get("sharpe_ratio", 0),
                    "return": metrics.get("annual_return", 0),
                })
                print(f"\n[SUCCESS] Strategy PASSED! Added to successful strategies.")
            else:
                self.failed_strategies.append({
                    "id": self.current_strategy.get("id", ""),
                    "name": self.current_strategy.get("strategy_name", ""),
                    "reason": "; ".join(failure_reasons[:3]),
                })
                print(f"\n[FAILED] Strategy did not pass criteria.")

            return feedback

        return None

    def run(self) -> Dict[str, Any]:
        """Run the complete interactive workflow."""
        self._print_separator("VIBEQUANT INTERACTIVE WORKFLOW", "=")
        print(f"Run ID: {self.run_id}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Target Strategies: {self.min_successful_strategies}")
        print("\nThis workflow will prompt you (Claude Code) to act as each agent.")
        print("Each agent has ISOLATED context - no conversation history is shared.")
        print("=" * 70)

        while True:
            # Orchestrator decision
            decision = self.run_orchestrator()
            if decision == "terminate":
                break

            # Clear current work items
            self.current_hypothesis = None
            self.current_strategy = None

            # Run agent pipeline
            print("\n" + "#" * 70)
            print(f"# ITERATION {self.iteration}")
            print("#" * 70)

            # 1. Insight Agent
            hypothesis = self.run_insight_agent()
            if not hypothesis:
                print("[SKIP] No valid hypothesis, continuing to next iteration...")
                continue

            # 2. Research Agent
            strategy = self.run_research_agent()
            if not strategy:
                print("[SKIP] No valid strategy code, continuing to next iteration...")
                continue

            # 3. Backtest Agent (automated)
            backtest_results = self.run_backtest_agent()
            if not backtest_results:
                print("[SKIP] Backtest failed, continuing to next iteration...")
                continue

            # 4. Feedback Agent
            feedback = self.run_feedback_agent(backtest_results)
            if not feedback:
                print("[SKIP] No valid feedback, continuing to next iteration...")
                continue

        # Final summary
        self._print_separator("WORKFLOW COMPLETE", "=")
        print(f"Total Iterations: {self.iteration}")
        print(f"Successful Strategies: {len(self.successful_strategies)}")
        print(f"Failed Strategies: {len(self.failed_strategies)}")

        if self.successful_strategies:
            print("\n--- SUCCESSFUL STRATEGIES ---")
            for s in self.successful_strategies:
                print(f"  - {s['name']}: Sharpe={s['sharpe']:.2f}, Return={s['return']:.2%}")

        print("\n" + self.memory.get_summary_report())

        return {
            "run_id": self.run_id,
            "iterations": self.iteration,
            "successful_strategies": self.successful_strategies,
            "failed_strategies": self.failed_strategies,
        }


def run_interactive(
    memory_dir: str = "./memory",
    max_iterations: int = 10,
    min_successful_strategies: int = 3,
) -> Dict[str, Any]:
    """
    Convenience function to run the interactive workflow.

    Args:
        memory_dir: Directory for memory storage
        max_iterations: Maximum iterations
        min_successful_strategies: Target number of successful strategies

    Returns:
        Workflow results
    """
    workflow = InteractiveWorkflow(
        memory_dir=memory_dir,
        max_iterations=max_iterations,
        min_successful_strategies=min_successful_strategies,
    )
    return workflow.run()


if __name__ == "__main__":
    run_interactive(max_iterations=3)
