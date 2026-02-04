"""
VibeQuant: Multi-Agent Alpha Discovery System
Designed for Alpaca Trading API
"""

__version__ = "0.1.0"

from .variation_tester import (
    StrategyVariationTester,
    test_strategy_variations,
    test_smart_variations,
    UNIVERSE_CONFIGS,
    VARIATION_CONFIGS,
)

from .smart_variations import (
    SmartVariationGenerator,
    SmartVariation,
    StrategyAnalyzer,
    StrategyCharacteristics,
    StrategyType,
    RebalanceType,
    analyze_and_suggest_variations,
    print_analysis_report,
)

from .universe_search import (
    UniverseSearcher,
    search_universes,
    QUANT_UNIVERSES,
    RUSSELL_1000,
    RUSSELL_2000_SAMPLE,
    RUSSELL_3000_SAMPLE,
    NASDAQ_100,
)

from .alpha101 import (
    ALPHA_FUNCTIONS,
    alpha_to_long_only_signals,
    alpha001, alpha002, alpha006, alpha012, alpha033,
    alpha041, alpha042, alpha053, alpha054, alpha101,
)

from .adversarial_validation import (
    AdversarialValidator,
    ValidationResult,
    ValidationIssue,
    Severity,
    validate_strategy,
    validate_from_files,
)

from .strategy_evaluator import (
    evaluate_backtest_metrics,
    save_validated_alpha,
    extract_learnings,
    get_passing_criteria,
    EvaluationResult,
)
