"""
VibeQuant: Multi-Agent Alpha Discovery System
Main entry point for running the automated alpha discovery workflow.

Supports two modes:
1. Autonomous mode: Uses LLM API (requires ANTHROPIC_API_KEY or OPENAI_API_KEY)
2. Interactive mode: Uses Claude Code as the brain (no external API needed)
"""

import argparse
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def check_environment(interactive: bool = False, auto: bool = False):
    """Check that required environment variables are set."""
    required_vars = [
        "ALPACA_PAPER_API_KEY",
        "ALPACA_PAPER_SECRET_KEY",
    ]

    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"ERROR: Missing required environment variables: {missing}")
        print("Please add them to your .env file")
        sys.exit(1)

    # Interactive and auto modes don't need LLM API keys
    if interactive or auto:
        return None

    # Autonomous mode needs at least one LLM provider
    llm_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ]

    has_llm = any(os.getenv(v) for v in llm_vars)
    if not has_llm:
        print(f"ERROR: Need at least one LLM API key for autonomous mode: {llm_vars}")
        print("Please add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file")
        print("Or use --interactive mode to use Claude Code as the brain")
        sys.exit(1)

    return "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VibeQuant: Multi-Agent Alpha Discovery System"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode using Claude Code as the brain (no API key needed)",
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run fully automated mode - no LLM or user input needed",
    )

    parser.add_argument(
        "--survivorship-free",
        action="store_true",
        help="Use survivorship-bias-free S&P 500 universe (requires S&P500.csv)",
    )

    parser.add_argument(
        "--universe-size",
        type=int,
        default=200,
        help="Maximum number of stocks in universe, filtered by liquidity (default: 200, 0=no limit)",
    )

    parser.add_argument(
        "--min-volume",
        type=int,
        default=500000,
        help="Minimum average daily volume filter (default: 500000)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Maximum number of iterations (default: 10)",
    )

    parser.add_argument(
        "--target-strategies",
        type=int,
        default=3,
        help="Target number of successful strategies (default: 3)",
    )

    parser.add_argument(
        "--memory-dir",
        type=str,
        default="./memory",
        help="Directory for memory storage (default: ./memory)",
    )

    parser.add_argument(
        "--llm",
        type=str,
        choices=["anthropic", "openai"],
        default=None,
        help="LLM provider for autonomous mode (default: auto-detect from env)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name to use in autonomous mode",
    )

    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Run a quick data loading test",
    )

    parser.add_argument(
        "--test-backtest",
        action="store_true",
        help="Run a quick backtest test",
    )

    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="Display current memory state and exit",
    )

    args = parser.parse_args()

    # Check environment
    llm_provider = check_environment(interactive=args.interactive, auto=args.auto)
    if not args.interactive and not args.auto:
        llm_provider = args.llm or llm_provider

    print("\n" + "=" * 60)
    print("VIBEQUANT - Multi-Agent Alpha Discovery System")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.auto:
        mode_str = "FULLY AUTOMATED (no LLM)"
        if hasattr(args, 'survivorship_free') and args.survivorship_free:
            mode_str += " + SURVIVORSHIP-BIAS-FREE"
    elif args.interactive:
        mode_str = "INTERACTIVE (Claude Code)"
    else:
        mode_str = f"AUTONOMOUS ({llm_provider})"
    print(f"Mode: {mode_str}")
    print(f"Max Iterations: {args.iterations}")
    print(f"Target Strategies: {args.target_strategies}")
    if args.auto:
        universe_str = f"{args.universe_size} most liquid" if args.universe_size > 0 else "all available"
        print(f"Universe: {universe_str} (min vol: {args.min_volume:,})")
    print(f"Memory Directory: {args.memory_dir}")
    print("=" * 60 + "\n")

    # Handle test modes
    if args.test_data:
        print("Running data loading test...")
        from vibequant.data import AlpacaDataLoader

        loader = AlpacaDataLoader()
        etfs = loader.get_popular_etfs()[:5]
        print(f"Testing with: {etfs}")

        bars = loader.get_bars(etfs, timeframe="1Day", lookback_days=30)
        print(f"Loaded {len(bars)} bars")
        print(bars.head())

        stats = loader.validate_data(bars)
        print(f"Validation: {stats}")
        return

    if args.test_backtest:
        print("Running backtest test...")
        import pandas as pd
        import numpy as np
        from vibequant.backtest import BacktestEngine

        # Create dummy data
        dates = pd.date_range("2022-01-01", "2024-01-01", freq="B")
        symbols = ["AAPL", "MSFT", "GOOGL"]

        np.random.seed(42)
        prices = pd.DataFrame(
            index=dates,
            data={
                s: 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
                for s in symbols
            },
        )

        # Simple momentum signals
        sma = prices.rolling(window=20).mean()
        signals = (prices > sma).astype(float)
        signals = signals / signals.sum(axis=1).replace(0, 1).values.reshape(-1, 1)
        signals = signals.fillna(0)

        engine = BacktestEngine()
        results = engine.run(prices, signals)

        print(f"Total Return: {results.total_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Trades: {results.num_trades}")
        return

    if args.show_memory:
        from vibequant.memory import StrategyMemory

        memory = StrategyMemory(args.memory_dir)
        print(memory.get_summary_report())
        return

    # Run the workflow
    import json

    if args.auto:
        # Fully automated mode - no LLM or user input needed
        from vibequant.auto_workflow import AutomatedWorkflow

        workflow = AutomatedWorkflow(
            memory_dir=args.memory_dir,
            max_iterations=args.iterations,
            min_successful=args.target_strategies,
            survivorship_bias_free=args.survivorship_free,
            include_etfs=True,
            universe_size=args.universe_size,
            min_avg_volume=args.min_volume,
        )
        results = workflow.run()

    elif args.interactive:
        # Interactive mode - uses Claude Code as the brain
        from vibequant.interactive_workflow import InteractiveWorkflow

        workflow = InteractiveWorkflow(
            memory_dir=args.memory_dir,
            max_iterations=args.iterations,
            min_successful_strategies=args.target_strategies,
        )
        results = workflow.run()

    else:
        # Autonomous mode - uses LLM API
        from vibequant.workflow import create_workflow

        workflow = create_workflow(
            llm_provider=llm_provider,
            model_name=args.model,
            memory_dir=args.memory_dir,
            max_iterations=args.iterations,
        )
        results = workflow.run()

    # Save results summary
    run_id = results.get("run_id", "unknown")
    results_file = os.path.join(args.memory_dir, f"run_{run_id}.json")

    summary = {
        "run_id": run_id,
        "mode": "interactive" if args.interactive else "autonomous",
        "started_at": results.get("started_at", datetime.now().isoformat()),
        "completed_at": datetime.now().isoformat(),
        "iterations": results.get("iterations", results.get("iteration", 0)),
        "successful_strategies": results.get("successful_strategies", []),
        "failed_strategies_count": len(results.get("failed_strategies", [])),
    }

    os.makedirs(args.memory_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
