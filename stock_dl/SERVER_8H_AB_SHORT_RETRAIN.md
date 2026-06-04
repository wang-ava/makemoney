# 8h A/B Short-Term Retrain

Goal: retrain both A and B for short-term tradable profit, not a generic stock ranking score.

Primary target: `next_open_to_next_close`.

Meaning: use information available before/near the next open, buy at the next open, and train/evaluate on the next close return. This directly matches the "morning buy, afternoon close profit" objective.

## Why the old target was a problem

The old close-to-close target can show a good validation IC while still failing live execution. It rewards ranking stocks by next close return from today's close, but your actual trade is often entered at a later executable price. If the alpha is already gone by the open, the model can look good on paper and still lose money.

Current audit results before retraining were red:

- Scheme A: tradable IC `0.0332`, Top70 tradable excess `-0.1938%`, last20 Top70 excess `-0.1012%`.
- Scheme B: tradable IC `0.0246`, Top70 tradable excess `-0.2534%`, last20 Top70 excess `-0.1604%`.

Conclusion: raw TopK auto-buy is not justified until the retrained short-target audit turns green.

## 8h budget

The server script builds the panel once, copies it to B, then trains both schemes.

Expected budget on one RTX 5090:

- Shared panel build: 25-60 minutes.
- A deep model: capped at 150 minutes.
- A eval/backtest/audit: 10-20 minutes.
- B deep model: capped at 150 minutes.
- B LightGBM: capped at 65 minutes.
- B blend/eval/backtest/audit: 15-30 minutes.

Total planned runtime: about 6.7-7.6 hours, with a small buffer inside the 8h allocation.

## Run on the server

From the `stock_dl` directory:

```bash
bash scripts/run_server_8h_ab_short.sh
```

With Slurm:

```bash
sbatch jobs/submit_server_8h_ab_short.sbatch
```

If the cluster requires a named 5090 resource, change this line in the sbatch file:

```bash
#SBATCH --gres=gpu:1
```

to the cluster-specific form, for example:

```bash
#SBATCH --gres=gpu:RTX5090:1
```

## Acceptance gates

Do not use the new model for automatic buying unless all are true on the audit report:

- `tradable_ic_mean >= 0.05`
- Top30 `tradable_topk_excess_last20 > 0`
- Top30 `realizable_t1_topk_excess_last20 > 0`

Use B only if its blended audit beats A. If both stay red, the issue is model signal quality, not merely order placement.

## What is actually trained

Each sample is a 20-trading-day window for one stock. The model sees technical, liquidity, money-flow, valuation, market-index, industry-relative, volatility, momentum, and cross-sectional rank/z-score features. It predicts a cross-sectional score for the signal date.

The server A/B configs use:

- target: `next_open_to_next_close`
- target column: `label_rank`
- loss: IC loss plus top/bottom ranking and direction auxiliary losses
- early stopping: `val_topk_excess`
- TopK for training selection: 30

This means checkpoint selection now follows the actual long-only use case: whether the top predicted stocks outperform the same-day universe on the short tradable leg.

The label also filters out samples that are not realistically buyable at the next open: missing/zero next-day volume or amount, invalid next open/close, and next-open gaps above the configured limit-up threshold.

## Noon-entry warning

"Today noon buy, next day profit" cannot be modeled honestly from daily OHLC alone. A noon-entry model needs intraday/noon features and an executable noon price. With only daily data, using high/low/close from the same day as noon features would leak future information. The current retrain therefore focuses on the honest daily-data short horizon: next open to next close.

## Causality warning

The current daily-data causal chain is:

1. Build features from signal date `d` after the close.
2. Generate orders before the next open.
3. Buy at `d+1` open.
4. Audit mark-to-market at `d+1` close and T+1 realizable return at `d+2` close.

So this is suitable for "previous close signal, next morning buy." It is not suitable for "same morning signal from same-day data" unless intraday data is added.
