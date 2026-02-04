"""
Run a single alpha discovery iteration step by step.
This script allows Claude Code to act as each agent.
"""

import json
import sys
from datetime import datetime
from vibequant.memory.strategy_memory import StrategyMemory, BacktestResult
from vibequant.data.alpaca_loader import AlpacaDataLoader
from vibequant.backtest.engine import BacktestEngine, BacktestConfig, execute_strategy_code
from vibequant.agents.base import check_passing_criteria, PASSING_CRITERIA
from vibequant.agents.prompts import (
    format_insight_prompt,
    format_research_prompt,
    format_feedback_prompt,
)


def print_separator(title, char="="):
    print(f"\n{char * 70}")
    print(f" {title}")
    print(f"{char * 70}\n")


def main():
    # Initialize components
    memory = StrategyMemory("./memory")
    data_loader = AlpacaDataLoader()
    backtest_engine = BacktestEngine(BacktestConfig())

    step = sys.argv[1] if len(sys.argv) > 1 else "insight"

    if step == "insight":
        # =====================================================
        # STEP 1: INSIGHT AGENT
        # =====================================================
        print_separator("INSIGHT AGENT - Generate Hypothesis")

        context = memory.get_context_for_insight_agent()
        prompt = format_insight_prompt(
            failed_patterns=context.get("failed_patterns", [])[-10:],
            successful_patterns=context.get("successful_patterns", [])[-10:],
            recent_failures=context.get("recent_failures", [])[-5:],
            untested_categories=context.get("untested_categories", []),
        )

        print("CONTEXT FOR INSIGHT AGENT:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)
        print("\nPlease provide hypothesis JSON response.")

    elif step == "save_hypothesis":
        # Save the hypothesis from command line arg
        hypothesis_json = sys.argv[2]
        hypothesis = json.loads(hypothesis_json)

        hyp_record = memory.add_hypothesis(
            hypothesis=hypothesis.get("hypothesis", ""),
            rationale=hypothesis.get("rationale", ""),
            category=hypothesis.get("category", "other"),
            tags=hypothesis.get("data_requirements", []),
            priority_score=hypothesis.get("priority_score", 0.5),
            metadata=hypothesis.get("expected_characteristics", {}),
        )

        print(f"Hypothesis saved with ID: {hyp_record.id}")

        # Save to temp file for next step
        hypothesis["id"] = hyp_record.id
        with open("./memory/current_hypothesis.json", "w") as f:
            json.dump(hypothesis, f, indent=2)

    elif step == "research":
        # =====================================================
        # STEP 2: RESEARCH AGENT
        # =====================================================
        print_separator("RESEARCH AGENT - Write Strategy Code")

        with open("./memory/current_hypothesis.json", "r") as f:
            hypothesis = json.load(f)

        context = memory.get_context_for_research_agent(hypothesis.get("id", ""))
        prompt = format_research_prompt(
            hypothesis=hypothesis,
            similar_strategies=context.get("similar_strategies", [])[:2],
            technical_notes=context.get("technical_notes", [])[-5:],
        )

        print("CONTEXT FOR RESEARCH AGENT:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)
        print("\nPlease provide strategy code JSON response.")

    elif step == "save_strategy":
        # Save the strategy
        strategy_json = sys.argv[2]
        strategy = json.loads(strategy_json)

        with open("./memory/current_hypothesis.json", "r") as f:
            hypothesis = json.load(f)

        code = strategy.get("code", "")
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        strategy_record = memory.add_strategy(
            name=strategy.get("strategy_name", "Unnamed Strategy"),
            hypothesis_id=hypothesis.get("id", ""),
            description=strategy.get("description", ""),
            category=hypothesis.get("category", "other"),
            code=code,
            parameters=strategy.get("parameters", {}),
            tags=strategy.get("data_requirements", []),
        )

        print(f"Strategy saved with ID: {strategy_record.id}")

        # Save for next step
        strategy["id"] = strategy_record.id
        strategy["code"] = code
        strategy["hypothesis_id"] = hypothesis.get("id", "")
        with open("./memory/current_strategy.json", "w") as f:
            json.dump(strategy, f, indent=2)

    elif step == "backtest":
        # =====================================================
        # STEP 3: BACKTEST AGENT (Automated)
        # =====================================================
        print_separator("BACKTEST AGENT - Running Backtest")

        with open("./memory/current_strategy.json", "r") as f:
            strategy = json.load(f)

        code = strategy.get("code", "")
        print(f"Strategy: {strategy.get('strategy_name', 'Unknown')}")
        print(f"Code length: {len(code)} chars")
        print("\nLoading market data from Alpaca...")

        # Load data
        symbols = data_loader.get_popular_etfs()[:20]
        print(f"Symbols: {symbols}")

        from datetime import datetime
        data = data_loader.get_bars(
            symbols=symbols,
            timeframe="1Day",
            start=datetime(2016, 1, 1),  # Start from 2016
        )

        if data.empty:
            print("ERROR: No data loaded!")
            return

        prices = data["close"].unstack(level=0)
        print(f"Data shape: {prices.shape}")
        print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

        print("\nGenerating signals...")
        try:
            signals = execute_strategy_code(code, prices)
            print(f"Signals shape: {signals.shape}")
        except Exception as e:
            print(f"ERROR executing strategy code: {e}")
            import traceback
            traceback.print_exc()

            # Save error result
            result = {"success": False, "error": str(e)}
            with open("./memory/current_backtest.json", "w") as f:
                json.dump(result, f, indent=2)
            return

        print("\nRunning backtest...")
        benchmark_prices = prices.get("SPY")
        results = backtest_engine.run(prices, signals, benchmark_prices)

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

        passed, failures = check_passing_criteria(metrics)
        print(f"\nPassing Criteria: {'PASS' if passed else 'FAIL'}")
        if failures:
            for f in failures:
                print(f"  - {f}")

        # Save results
        result = {
            "success": True,
            "strategy_id": strategy.get("id", ""),
            "metrics": metrics,
            "start_date": results.start_date,
            "end_date": results.end_date,
            "passed_criteria": passed,
            "criteria_failures": failures,
        }
        with open("./memory/current_backtest.json", "w") as f:
            json.dump(result, f, indent=2)

    elif step == "feedback":
        # =====================================================
        # STEP 4: FEEDBACK AGENT
        # =====================================================
        print_separator("FEEDBACK AGENT - Evaluate Results")

        with open("./memory/current_hypothesis.json", "r") as f:
            hypothesis = json.load(f)
        with open("./memory/current_strategy.json", "r") as f:
            strategy = json.load(f)
        with open("./memory/current_backtest.json", "r") as f:
            backtest_results = json.load(f)

        if not backtest_results.get("success", False):
            print(f"Backtest failed: {backtest_results.get('error', 'Unknown')}")
            print("Strategy automatically marked as FAILED")
            return

        metrics = backtest_results.get("metrics", {})
        context = memory.get_context_for_feedback_agent(strategy.get("id", ""))

        prompt = format_feedback_prompt(
            strategy_name=strategy.get("strategy_name", ""),
            category=hypothesis.get("category", ""),
            hypothesis=hypothesis.get("hypothesis", ""),
            backtest_results=metrics,
            avg_sharpe_successful=context.get("benchmark_metrics", {}).get("avg_sharpe_of_successful", 0),
            best_sharpe=context.get("benchmark_metrics", {}).get("best_sharpe", 0),
            best_return=context.get("benchmark_metrics", {}).get("best_return", 0),
        )

        print("CONTEXT FOR FEEDBACK AGENT:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)
        print(f"\nAutomated Criteria Check: {'PASS' if backtest_results.get('passed_criteria') else 'FAIL'}")
        print(f"Criteria Failures: {backtest_results.get('criteria_failures', [])}")
        print(f"\nCurrent Passing Criteria:")
        for k, v in PASSING_CRITERIA.items():
            print(f"  - {k}: {v}")
        print("\nPlease provide feedback JSON response.")

    elif step == "save_feedback":
        # Save feedback and update memory
        feedback_json = sys.argv[2]
        feedback = json.loads(feedback_json)

        with open("./memory/current_hypothesis.json", "r") as f:
            hypothesis = json.load(f)
        with open("./memory/current_strategy.json", "r") as f:
            strategy = json.load(f)
        with open("./memory/current_backtest.json", "r") as f:
            backtest_results = json.load(f)

        metrics = backtest_results.get("metrics", {})
        passed = backtest_results.get("passed_criteria", False) and feedback.get("passed", False)

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

        memory.update_strategy_backtest(
            strategy_id=strategy.get("id", ""),
            backtest_results=bt_result,
            passed=passed,
            failure_reasons=failure_reasons,
            improvement_suggestions=feedback.get("improvement_suggestions", []),
        )

        # Store learnings
        learnings = feedback.get("learnings_for_memory", {})
        for pattern in learnings.get("successful_patterns", []):
            memory.add_learning("successful_patterns", pattern)
        for pattern in learnings.get("failed_patterns", []):
            memory.add_learning("failed_patterns", pattern)
        for note in learnings.get("technical_notes", []):
            memory.add_learning("technical_notes", note)

        if passed:
            print(f"\n{'='*50}")
            print("STRATEGY PASSED!")
            print(f"{'='*50}")
            print(f"Name: {strategy.get('strategy_name')}")
            print(f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
        else:
            print(f"\n{'='*50}")
            print("STRATEGY FAILED")
            print(f"{'='*50}")
            print(f"Reasons: {failure_reasons}")

        print("\n" + memory.get_summary_report())


if __name__ == "__main__":
    main()
