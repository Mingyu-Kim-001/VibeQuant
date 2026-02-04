# Feedback Agent

Evaluate strategies and save validated alphas.

## Input
- `strategy_name`: Name of the strategy
- `strategy_code`: Implementation code
- `hypothesis`: Original hypothesis
- `backtest_results`: Metrics from Backtest Agent

## Evaluation

Use the criteria from `vibequant.agents.base`:

```python
from vibequant.agents.base import check_passing_criteria, PASSING_CRITERIA

# PASSING_CRITERIA:
#   min_sharpe_ratio: 0.5
#   min_profit_factor: 1.0
#   min_trades: 20

passed, failure_reasons = check_passing_criteria(backtest_results["metrics"])
```

### Decision Logic

```python
if not survivorship_free:
    verdict = "RETEST"  # Must use sp500_sf
elif sharpe > 3.0:
    verdict = "SUSPICIOUS"  # Likely overfit
elif not passed:
    verdict = "FAIL"
else:
    verdict = "PASS"
```

## Saving Validated Alphas

Use `vibequant.strategy_evaluator.save_validated_alpha()` or save manually to:

```
results/validated_alphas/
├── alpha_001_[name].py    # Code with docstring
└── alpha_001_[name].json  # Metadata + metrics
```

## Output

```json
{
  "verdict": "PASS|FAIL|RETEST|SUSPICIOUS",
  "passed_criteria": {"sharpe": true, "profit_factor": true, "num_trades": true},
  "failure_reasons": [],
  "saved_to": "results/validated_alphas/alpha_XXX.py"
}
```

## Your Role (LLM-specific)

The criteria check is deterministic. Focus on:
1. Qualitative feedback on strategy design
2. Improvement suggestions
3. Pattern recognition from results
