# Orchestrator Agent

Coordinate the alpha discovery pipeline.

## Pipeline

```
Insight → Research → Backtest → Feedback
   │          │          │          │
   ▼          ▼          ▼          ▼
hypothesis   code     metrics    verdict
```

## Automated Workflow

For fully automated discovery, use:

```python
from vibequant.auto_workflow import AutomatedWorkflow

workflow = AutomatedWorkflow(
    max_iterations=50,
    min_successful=3,
    survivorship_bias_free=True,
)
results = workflow.run()
```

## Passing Criteria

From `vibequant.agents.base.PASSING_CRITERIA`:
- Sharpe Ratio >= 0.5
- Profit Factor >= 1.0
- Number of Trades >= 20
- Universe must be survivorship-free (sp500_sf, dynamic, etfs)

## Output Location

```
results/validated_alphas/
├── alpha_001_*.py   # Code
├── alpha_001_*.json # Metadata
└── ...
```

## Termination

Stop when:
1. Found target number of validated alphas
2. Max iterations reached
3. All parameter combinations exhausted

## Your Role (LLM-specific)

Most orchestration is automated. Use LLM for:
1. Hypothesis generation (Insight Agent)
2. Code generation (Research Agent)
3. High-level decisions when stuck
