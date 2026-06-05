# 模型重训与上线检查

日期：2026-06-04

## 结论

当前模型不是完全无效，但原训练目标和短线实盘执行不一致。旧标签主要是 `close_to_next_close`，包含隔夜跳空；实际盘后出信号后，能交易的是次日开盘或盘中之后的收益。因此重新训练应先改标签，再谈模型结构。

## 当前旧模型审计结果

使用 `scripts/16_model_audit.py` 对现有 A/B 输出做可交易标签复核：

| 方案 | 旧验证标签IC | 可交易标签IC | 可交易标签最近20日IC | Top70可交易超额 | Top70最近20日可交易超额 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| A | 0.1051 | 0.0332 | 0.0267 | -0.1938% | -0.1012% | red |
| B | 0.0774 | 0.0246 | 0.0055 | -0.2534% | -0.1604% | red |

这说明旧模型在原始 close-to-close 标签上看起来有排序能力，但换成真实可交易的次日开盘到收盘收益后，Top70 组合没有正超额收益。继续按 raw TopK 自动交易没有统计依据。

## 已修正

1. `add_features` 新增 `label_mode`。
2. 默认训练标签改为 `next_open_to_next_close`，即次日开盘买入到次日收盘收益。
3. `feature_columns` 会排除所有 `label_` 前缀字段，避免新增诊断标签误入特征造成泄漏。
4. 构建面板、推理、交易指南、持仓扫描都已透传 `label_mode`。
5. 新增 `scripts/16_model_audit.py`，用于评估模型在可交易标签上的 IC、TopK 超额收益、最近窗口表现和回测回撤。

## 重训命令

方案 A：

```bash
cd stock_dl
python scripts/01_build_panel.py --config configs/a_share_pure_dl.yaml
python scripts/03_train.py --config configs/a_share_pure_dl.yaml
python scripts/04_eval_ic.py --config configs/a_share_pure_dl.yaml
python scripts/05_backtest.py --config configs/a_share_pure_dl.yaml
python scripts/16_model_audit.py --config configs/a_share_pure_dl.yaml
```

方案 B：

```bash
cd stock_dl
python scripts/01_build_panel.py --config configs/a_share_with_lgbm.yaml
python scripts/03_train.py --config configs/a_share_with_lgbm.yaml
python scripts/10_train_lgbm.py --config configs/a_share_with_lgbm.yaml
python scripts/11_blend_scores.py --config configs/a_share_with_lgbm.yaml
python scripts/04_eval_ic.py --config configs/a_share_with_lgbm.yaml
python scripts/05_backtest.py --config configs/a_share_with_lgbm.yaml
python scripts/16_model_audit.py --config configs/a_share_with_lgbm.yaml
```

## 上线前硬门槛

- `model_audit_summary.json.status` 不能是 `red`。
- 可交易标签最近 20 日 IC 应大于 `0.03`。
- Top70 可交易最近 20 日超额收益应大于 `0`。
- 回测最近 20 日收益不能低于 `-5%`。
- 如果最近 20 日变红，只保留观察列表，不自动下单。

## 执行层规则

- 不再执行 raw Top70 全量重同步。
- 每日换手控制在 1 到 3 只。
- 买入前过滤主力净流出、流动性弱、冲高回落、高开追涨。
- 模型只提供候选池，最终下单必须通过风险过滤和价格过滤。
