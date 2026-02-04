"""
VibeQuant LangGraph Workflow
Multi-agent orchestration for automated alpha discovery.
"""

import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Literal
from dataclasses import asdict
import uuid

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .agents.base import AgentState, AgentOutput, check_passing_criteria
from .agents.prompts import (
    format_orchestrator_prompt,
    format_insight_prompt,
    format_research_prompt,
    format_backtest_prompt,
    format_feedback_prompt,
)
from .memory.strategy_memory import StrategyMemory, BacktestResult
from .data.alpaca_loader import AlpacaDataLoader
from .backtest.engine import BacktestEngine, BacktestConfig, execute_strategy_code

load_dotenv()


class VibeQuantWorkflow:
    """
    Main workflow orchestrator using LangGraph.

    Implements the following flow:
    Orchestrator -> Insight -> Research -> Backtest -> Feedback -> (loop or end)
    """

    def __init__(
        self,
        llm_provider: Literal["openai", "anthropic"] = "anthropic",
        model_name: Optional[str] = None,
        memory_dir: str = "./memory",
        max_iterations: int = 10,
        min_successful_strategies: int = 3,
    ):
        """
        Initialize the workflow.

        Args:
            llm_provider: LLM provider to use
            model_name: Specific model name (defaults to best available)
            memory_dir: Directory for memory storage
            max_iterations: Maximum workflow iterations
            min_successful_strategies: Target number of successful strategies
        """
        self.max_iterations = max_iterations
        self.min_successful_strategies = min_successful_strategies

        # Initialize LLM
        if llm_provider == "anthropic":
            self.llm = ChatAnthropic(
                model=model_name or "claude-sonnet-4-20250514",
                temperature=0.7,
                max_tokens=4096,
            )
        else:
            self.llm = ChatOpenAI(
                model=model_name or "gpt-4-turbo-preview",
                temperature=0.7,
                max_tokens=4096,
            )

        # Initialize components
        self.memory = StrategyMemory(memory_dir)
        self.data_loader = AlpacaDataLoader()
        self.backtest_engine = BacktestEngine(BacktestConfig())

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the graph with our state schema
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("insight", self._insight_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("backtest", self._backtest_node)
        workflow.add_node("feedback", self._feedback_node)

        # Set entry point
        workflow.set_entry_point("orchestrator")

        # Add conditional edges from orchestrator
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_from_orchestrator,
            {
                "insight": "insight",
                "research": "research",
                "backtest": "backtest",
                "feedback": "feedback",
                "end": END,
            },
        )

        # Add edges for the main flow
        workflow.add_edge("insight", "research")
        workflow.add_edge("research", "backtest")
        workflow.add_edge("backtest", "feedback")
        workflow.add_edge("feedback", "orchestrator")

        return workflow.compile()

    def _route_from_orchestrator(self, state: AgentState) -> str:
        """Determine next node based on orchestrator decision."""
        workflow_state = state.get("workflow_state", "idle")

        # Check termination conditions
        if state.get("iteration", 0) >= self.max_iterations:
            return "end"

        if len(state.get("successful_strategies", [])) >= self.min_successful_strategies:
            return "end"

        if workflow_state == "error":
            return "end"

        # Route based on current state
        if workflow_state in ["idle", "generating_hypothesis"]:
            return "insight"
        elif workflow_state == "researching":
            return "research"
        elif workflow_state == "backtesting":
            return "backtest"
        elif workflow_state == "evaluating":
            return "feedback"

        # Default to starting new iteration
        return "insight"

    def _orchestrator_node(self, state: AgentState) -> Dict[str, Any]:
        """Orchestrator decision node."""
        print(f"\n{'='*60}")
        print(f"ORCHESTRATOR - Iteration {state.get('iteration', 0) + 1}")
        print(f"{'='*60}")

        memory_context = self.memory.get_context_for_insight_agent()

        prompt = format_orchestrator_prompt(
            iteration=state.get("iteration", 0),
            max_iterations=self.max_iterations,
            num_successful=len(state.get("successful_strategies", [])),
            num_failed=len(state.get("failed_strategies", [])),
            workflow_state=state.get("workflow_state", "idle"),
            memory_summary=json.dumps(memory_context.get("metrics", {}), indent=2),
            min_successful_strategies=self.min_successful_strategies,
        )

        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Analyze the current state and decide the next action."),
        ])

        try:
            # Parse JSON response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            decision = json.loads(content)
            print(f"Decision: {decision.get('action', 'continue')}")
            print(f"Next Phase: {decision.get('next_phase', 'insight')}")

            new_state = {
                "iteration": state.get("iteration", 0) + 1,
                "workflow_state": "generating_hypothesis",
                "memory_context": memory_context,
            }

            if decision.get("action") == "terminate":
                new_state["workflow_state"] = "complete"

            return new_state

        except Exception as e:
            print(f"Orchestrator error: {e}")
            return {
                "workflow_state": "generating_hypothesis",
                "iteration": state.get("iteration", 0) + 1,
            }

    def _insight_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate new trading hypothesis."""
        print(f"\n{'='*60}")
        print("INSIGHT AGENT - Generating Hypothesis")
        print(f"{'='*60}")

        context = state.get("memory_context", {})

        prompt = format_insight_prompt(
            failed_patterns=context.get("failed_patterns", []),
            successful_patterns=context.get("successful_patterns", []),
            recent_failures=context.get("recent_failures", []),
            untested_categories=context.get("untested_categories", []),
        )

        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Generate a novel, testable trading hypothesis."),
        ])

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            hypothesis = json.loads(content)

            # Store in memory
            hyp_record = self.memory.add_hypothesis(
                hypothesis=hypothesis.get("hypothesis", ""),
                rationale=hypothesis.get("rationale", ""),
                category=hypothesis.get("category", "other"),
                tags=hypothesis.get("data_requirements", []),
                priority_score=hypothesis.get("priority_score", 0.5),
                metadata=hypothesis.get("expected_characteristics", {}),
            )

            print(f"Generated: {hypothesis.get('hypothesis', 'N/A')[:100]}...")
            print(f"Category: {hypothesis.get('category', 'N/A')}")

            return {
                "current_hypothesis": {**hypothesis, "id": hyp_record.id},
                "workflow_state": "researching",
                "hypotheses_generated": [{**hypothesis, "id": hyp_record.id}],
            }

        except Exception as e:
            print(f"Insight error: {e}")
            traceback.print_exc()
            return {
                "workflow_state": "error",
                "error_message": str(e),
            }

    def _research_node(self, state: AgentState) -> Dict[str, Any]:
        """Convert hypothesis to executable code."""
        print(f"\n{'='*60}")
        print("RESEARCH AGENT - Generating Strategy Code")
        print(f"{'='*60}")

        hypothesis = state.get("current_hypothesis", {})
        if not hypothesis:
            return {"workflow_state": "error", "error_message": "No hypothesis to research"}

        context = self.memory.get_context_for_research_agent(hypothesis.get("id", ""))

        prompt = format_research_prompt(
            hypothesis=hypothesis,
            similar_strategies=context.get("similar_strategies", []),
            technical_notes=context.get("technical_notes", []),
        )

        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Write the strategy code based on the hypothesis."),
        ])

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            strategy_output = json.loads(content)

            # Extract code - handle case where code might be in a code block
            code = strategy_output.get("code", "")
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            # Store in memory
            strategy_record = self.memory.add_strategy(
                name=strategy_output.get("strategy_name", "Unnamed Strategy"),
                hypothesis_id=hypothesis.get("id", ""),
                description=strategy_output.get("description", ""),
                category=hypothesis.get("category", "other"),
                code=code,
                parameters=strategy_output.get("parameters", {}),
                tags=strategy_output.get("data_requirements", []),
            )

            print(f"Strategy: {strategy_output.get('strategy_name', 'N/A')}")
            print(f"Code length: {len(code)} chars")

            return {
                "current_strategy": {
                    **strategy_output,
                    "id": strategy_record.id,
                    "code": code,
                    "hypothesis_id": hypothesis.get("id", ""),
                },
                "workflow_state": "backtesting",
                "strategies_coded": [{
                    **strategy_output,
                    "id": strategy_record.id,
                    "hypothesis_id": hypothesis.get("id", ""),
                }],
            }

        except Exception as e:
            print(f"Research error: {e}")
            traceback.print_exc()
            return {
                "workflow_state": "error",
                "error_message": str(e),
            }

    def _backtest_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute backtest on the strategy."""
        print(f"\n{'='*60}")
        print("BACKTEST AGENT - Running Backtest")
        print(f"{'='*60}")

        strategy = state.get("current_strategy", {})
        if not strategy:
            return {"workflow_state": "error", "error_message": "No strategy to backtest"}

        code = strategy.get("code", "")

        try:
            # Load data
            print("Loading market data...")
            symbols = self.data_loader.get_popular_etfs()[:20]  # Start with ETFs
            data = self.data_loader.get_bars(
                symbols=symbols,
                timeframe="1Day",
                lookback_days=504,  # 2 years
            )

            if data.empty:
                raise ValueError("No data loaded from Alpaca")

            # Pivot data to wide format (symbols as columns)
            prices = data["close"].unstack(level=0)
            print(f"Data shape: {prices.shape}")

            # Execute strategy code
            print("Generating signals...")
            signals = execute_strategy_code(code, prices)

            # Get benchmark
            benchmark_prices = None
            if "SPY" in prices.columns:
                benchmark_prices = prices["SPY"]

            # Run backtest
            print("Running backtest...")
            results = self.backtest_engine.run(prices, signals, benchmark_prices)

            # Convert to dict for state
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

            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Annual Return: {metrics['annual_return']:.2%}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Trades: {metrics['num_trades']}")

            return {
                "current_backtest_results": {
                    "success": True,
                    "strategy_id": strategy.get("id", ""),
                    "metrics": metrics,
                    "start_date": results.start_date,
                    "end_date": results.end_date,
                },
                "workflow_state": "evaluating",
                "backtests_run": [{
                    "strategy_id": strategy.get("id", ""),
                    "metrics": metrics,
                    "success": True,
                }],
            }

        except Exception as e:
            print(f"Backtest error: {e}")
            traceback.print_exc()

            return {
                "current_backtest_results": {
                    "success": False,
                    "strategy_id": strategy.get("id", ""),
                    "error": str(e),
                },
                "workflow_state": "evaluating",
                "backtests_run": [{
                    "strategy_id": strategy.get("id", ""),
                    "success": False,
                    "error": str(e),
                }],
            }

    def _feedback_node(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate results and update memory."""
        print(f"\n{'='*60}")
        print("FEEDBACK AGENT - Evaluating Results")
        print(f"{'='*60}")

        strategy = state.get("current_strategy", {})
        hypothesis = state.get("current_hypothesis", {})
        backtest_results = state.get("current_backtest_results", {})

        if not backtest_results.get("success", False):
            # Backtest failed
            error_msg = backtest_results.get("error", "Unknown error")
            print(f"Backtest failed: {error_msg}")

            self.memory.add_learning(
                "failed_patterns",
                f"Code execution error in {strategy.get('strategy_name', 'Unknown')}: {error_msg}",
            )

            return {
                "failed_strategies": [{
                    "id": strategy.get("id", ""),
                    "name": strategy.get("strategy_name", ""),
                    "reason": f"Backtest execution failed: {error_msg}",
                }],
                "workflow_state": "generating_hypothesis",
            }

        metrics = backtest_results.get("metrics", {})
        context = self.memory.get_context_for_feedback_agent(strategy.get("id", ""))

        prompt = format_feedback_prompt(
            strategy_name=strategy.get("strategy_name", ""),
            category=hypothesis.get("category", ""),
            hypothesis=hypothesis.get("hypothesis", ""),
            backtest_results=metrics,
            avg_sharpe_successful=context.get("benchmark_metrics", {}).get("avg_sharpe_of_successful", 0),
            best_sharpe=context.get("benchmark_metrics", {}).get("best_sharpe", 0),
            best_return=context.get("benchmark_metrics", {}).get("best_return", 0),
        )

        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Evaluate the backtest results and provide detailed feedback."),
        ])

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            feedback = json.loads(content)

            # Check passing criteria
            passed, failure_reasons = check_passing_criteria(metrics)

            # Override LLM decision with hard criteria check
            if not passed:
                feedback["passed"] = False
                feedback["failure_reasons"] = failure_reasons

            print(f"Verdict: {'PASSED' if feedback.get('passed', False) else 'FAILED'}")
            print(f"Score: {feedback.get('score', 0)}/10")

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

            self.memory.update_strategy_backtest(
                strategy_id=strategy.get("id", ""),
                backtest_results=bt_result,
                passed=feedback.get("passed", False),
                failure_reasons=feedback.get("failure_reasons", []),
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

            # Add feedback to strategy
            for fb in feedback.get("feedback", []):
                self.memory.add_strategy_feedback(strategy.get("id", ""), fb)

            if feedback.get("passed", False):
                return {
                    "successful_strategies": [{
                        "id": strategy.get("id", ""),
                        "name": strategy.get("strategy_name", ""),
                        "sharpe": metrics.get("sharpe_ratio", 0),
                        "return": metrics.get("annual_return", 0),
                    }],
                    "current_feedback": feedback,
                    "workflow_state": "generating_hypothesis",
                    "feedbacks_given": [feedback],
                }
            else:
                return {
                    "failed_strategies": [{
                        "id": strategy.get("id", ""),
                        "name": strategy.get("strategy_name", ""),
                        "reason": "; ".join(feedback.get("failure_reasons", [])),
                    }],
                    "current_feedback": feedback,
                    "workflow_state": "generating_hypothesis",
                    "feedbacks_given": [feedback],
                }

        except Exception as e:
            print(f"Feedback error: {e}")
            traceback.print_exc()
            return {
                "workflow_state": "generating_hypothesis",
            }

    def run(self, initial_state: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run the complete workflow.

        Args:
            initial_state: Optional initial state override

        Returns:
            Final state with all results
        """
        # Initialize state
        run_id = str(uuid.uuid4())[:8]
        state = initial_state or {
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "workflow_state": "idle",
            "error_message": None,
            "current_hypothesis": None,
            "current_strategy": None,
            "current_backtest_results": None,
            "current_feedback": None,
            "memory_context": {},
            "hypotheses_generated": [],
            "strategies_coded": [],
            "backtests_run": [],
            "feedbacks_given": [],
            "successful_strategies": [],
            "failed_strategies": [],
            "run_id": run_id,
            "started_at": datetime.now().isoformat(),
        }

        print("\n" + "=" * 60)
        print(f"VIBEQUANT WORKFLOW STARTED - Run ID: {run_id}")
        print("=" * 60)

        # Execute the graph
        final_state = self.graph.invoke(state)

        # Print summary
        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETE")
        print("=" * 60)
        print(f"Iterations: {final_state.get('iteration', 0)}")
        print(f"Successful Strategies: {len(final_state.get('successful_strategies', []))}")
        print(f"Failed Strategies: {len(final_state.get('failed_strategies', []))}")

        if final_state.get("successful_strategies"):
            print("\nSuccessful Strategies:")
            for s in final_state["successful_strategies"]:
                print(f"  - {s.get('name', 'Unknown')}: Sharpe={s.get('sharpe', 0):.2f}")

        # Print memory summary
        print("\n" + self.memory.get_summary_report())

        return final_state


def create_workflow(
    llm_provider: str = "anthropic",
    model_name: Optional[str] = None,
    memory_dir: str = "./memory",
    max_iterations: int = 10,
) -> VibeQuantWorkflow:
    """
    Factory function to create a VibeQuant workflow.

    Args:
        llm_provider: "openai" or "anthropic"
        model_name: Optional specific model name
        memory_dir: Directory for memory storage
        max_iterations: Maximum iterations to run

    Returns:
        Configured VibeQuantWorkflow instance
    """
    return VibeQuantWorkflow(
        llm_provider=llm_provider,
        model_name=model_name,
        memory_dir=memory_dir,
        max_iterations=max_iterations,
    )


if __name__ == "__main__":
    # Quick test
    workflow = create_workflow(max_iterations=2)
    results = workflow.run()
