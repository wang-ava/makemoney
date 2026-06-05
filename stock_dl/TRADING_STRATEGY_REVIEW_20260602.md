# A/B 交易策略复盘与 14 脚本修复说明

生成日期：2026-06-03  
复盘对象：`outputs_scheme_a/trading_guides`、`outputs_scheme_b/trading_guides` 中 20260529 与 20260601 的 `full_positions.csv` / `trading_orders.csv`。  
关注点：不比较 A/B 模型本身，只检查交易策略，也就是动态持仓、换手、持股数量、权重、现金与执行文件。

## 一句话结论

6月1日策略亏损的主要问题不是 A/B 哪个模型更好，而是执行层把“目标持仓表 full_positions”当成“实际订单表”使用，绕过了动态换手；同时旧版 14 脚本存在手数/股数、目标数量/本次买卖数量、交易日推断、现金约束、A/B 分数尺度不一致等问题。现在已经修复为：只用 `trading_orders.csv` 下单，`full_positions.csv` 也会把未执行目标标成 `skip`；高波动日少持、少换手、保留现金，A/B 使用同一套交易策略。

## 当日收盘复盘

口径：信号日收盘后生成 guide，下一交易日执行；收益按“建议买入价到执行日收盘价”的组合加权收益估算。

| guide | 信号日 | 执行日 | 原持仓目标数 | 估算收益 | 胜率 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| A 当前 20260529 | 2026-05-29 | 2026-06-01 | 30 | +1.99% | 83.3% | 6月1日大盘弱，但组合表现好 |
| A 当前 20260601 | 2026-06-01 | 2026-06-02 | 30 | -1.54% | 6.7% | 若按 full_positions 全买，会明显亏 |
| B 20260529 | 2026-05-29 | 2026-06-01 | 41 | +2.72% | 95.1% | 旧文件里数量列实际是股数 |
| B 20260601 | 2026-06-01 | 2026-06-02 | 41 | -1.62% | 7.3% | 原文件像空仓重买 41 只 |
| A old 20260601 | 2026-06-01 | 2026-06-02 | 41 | -1.91% | 4.9% | “候选外全清 + 买入放大”最激进 |

同期基准：2026-06-02 上证 +0.43%、沪深300 +1.45%、创业板 +2.66%。也就是说，6月2日并不是市场整体下跌导致亏损，而是当天选入池与交易执行规则共同造成了负收益。

## 发现的问题

1. `full_positions.csv` 被误用为下单表。  
   旧版 `full_positions` 是“理想目标持仓”，不是“今天实际要买卖”。例如 A 的 20260601 修复前实际 `trading_orders.csv` 只应买 7、卖 2，但 `full_positions.csv` 里 30 行都显示 `buy`。如果按 full_positions 全部买入，就绕开了动态换手。

2. 旧版订单数量用错了字段。  
   对已有持仓加仓时，买入清单部分地方写的是“目标总手数”，不是“本次买入手数”；减仓时也可能写成“当前总手数”，不是“本次卖出手数”。这会把加仓 20 手误写成买 49 手，或者把部分减仓误写成清仓。

3. B 的 20260601 不是连续持仓逻辑。  
   原 B 的 20260601 guide 里 `当前手数` 全为 0，等于把 5月29 买入的持仓忘掉，从空仓重新买 41 只。这不是动态持仓，是重新建仓。

4. 高波动日持仓逻辑和回测不一致。  
   14 脚本旧逻辑把 `vol_pct=1.0` 理解成“多持”，会把持仓数推高；`engine.py` 更合理地按高波动少持、低波动多持处理。两边不一致会让回测和实盘指南脱节。

5. A/B 分数尺度不同，却共用绝对分差阈值。  
   A 是原始 DL 分数，B 是 rank 融合分数。旧动态 K 用 `gap > 0.10` 这种绝对分差，对 A/B 不是同一条规则。现在改为先转成当日横截面百分位，再判断换手。

6. `score_gap_trigger` 配置原来没有在 14 / engine 里真正阻止换手。  
   旧逻辑最后强制 `max(1, k)`，即使持仓不差、候选优势不明显，也至少换 1 只。现在允许返回 0，避免噪音交易。

7. 候选外清理和买入放大绕开了动态 K。  
   旧版 `sell_outsiders=all` 或无限清理会卖出很多候选外股票，然后 `amplify_buys/topup/hand_expand` 又把现金全部买回去，实际换手远大于 `k_trade`。这正是 6月1日后亏损放大的核心执行风险。

8. 现金约束缺失。  
   若卖出后现金不足，旧版只提示资金缺口，但仍把买单写入 CSV。现在默认不补现金，资金不足时自动删掉低优先级买单。

## 修复后的交易策略

### 交易日期

信号日 `D` 收盘后生成 guide，实际交易日是下一个交易日，而不是简单日历 `D+1`。例如 2026-05-29 的下一交易日是 2026-06-01，不是 2026-05-30。

### 股票池与排序

1. 先由模型产生当日 `score`。A/B 只负责给股票排序。
2. 过滤不可买股票、涨跌幅过大股票、低成交额股票、上市时间过短股票、市值极端股票。
3. 再做 momentum 过滤：默认保留 `rank_ret_5d >= 0.2` 的股票。
4. 按过滤后的 `score` 从高到低排序。

### 持仓数量

配置：

```yaml
n_hold: 30
adaptive_min_hold: 24
adaptive_max_hold: 32
adaptive_vol_weight: 0.4
adaptive_ic_weight: 0.3
adaptive_conf_weight: 0.3
```

计算思想：高波动少持，低波动适度多持；模型越有区分度，允许稍多一点目标持仓。20260601 的 `vol_pct=1.0`、置信度约 1.0，所以目标持仓数被压到 28 只。

### 目标仓位

配置：

```yaml
min_position_ratio: 0.70
base_position_ratio: 0.85
max_position_ratio: 0.92
cash_reserve_ratio: 0.08
```

目标仓位公式：

```text
target = base
       + 0.08 * (confidence - 0.5)
       + 0.12 * ((1 - vol_pct) - 0.5)
然后 clip 到 [0.70, 0.92]
```

20260601 的市场波动分位是 100%，即 `vol_pct=1.0`，高波动扣仓位；模型置信度高会加一点仓位，最后目标仓位约 83%。

### 权重分配

默认使用 `score_weighted`：

```text
weight_i = score_i / sum(top_n_scores)
target_value_i = portfolio_value * target_position * weight_i
target_hands_i = floor(target_value_i / price_i / 100)
```

这样高分股票权重大，但不会全部等权；同时所有下单数量按 100 股整数倍取整。

### 动态换手

配置：

```yaml
k_trade: 2
dynamic_k: true
dynamic_k_use_rank_scores: true
dynamic_k_max: 3
dynamic_k_step: 1
score_gap_trigger: true
score_gap_trigger_threshold: 0.05
score_gap_low: 0.02
score_gap_high: 0.10
```

逻辑：

1. 先把当日所有分数转成横截面百分位，避免 A/B 分数尺度不同。
2. 当前持仓分数低于候选池中位数的，记为“烂股”。
3. `k` 从基准 `2` 开始。
4. 若烂股数大于等于 1，`k = max(k, 烂股数)`。
5. 若最佳候选明显高于最差持仓，且百分位 gap > 0.10，`k += 1`。
6. 若没有烂股且百分位 gap <= 0.05，`k = 0`，当天不做噪音换手。
7. 最终 `k <= 3`，优先降低手动执行滑点和真实持仓偏离风险。

### 候选外持仓

配置：

```yaml
sell_outsiders: true
sell_outsiders_mode: bad
sell_outsiders_unlimited: false
sell_outsiders_max: 2
```

只清理“候选外且分数低于候选中位数”的持仓；每天最多额外清理 2 只，并且不再绕开动态 K 大规模卖出。

例外：如果真实持仓和目标池严重错位，触发 `resync_on_mismatch`，则进入重同步模式，卖出所有非目标持仓，并尽量按目标池重建组合。这个模式用于纠正“实际账户和系统假设完全相反”的情况，不受日常 `dynamic_k_max` / `sell_outsiders_max` 限制。

### 买入放大与补仓

全部关闭：

```yaml
amplify_buys: false
hand_expand_enabled: false
topup_enabled: false
enforce_cash_limit: true
```

原因：6月1日亏损的核心之一就是换手被放大。先让 `k_trade` 真正约束每日交易，再谈收益增强。现金不足时，自动删除低优先级买单，不默认补现金。

## 修复后 20260601 新指南

### 方案 A

输入：沿用 A 的 20260529_old 持仓，组合估值约 1,024,453 元。  
输出目录：`outputs_scheme_a/trading_guides/20260601/`

| 项目 | 值 |
|---|---:|
| 交易日 | 2026-06-02 |
| 目标持仓 | 28 只 |
| 目标仓位 | 83% |
| 动态 K | 3 |
| 实际买入 | 3 只 |
| 实际卖出 | 2 只 |
| full_positions 状态 | 3 buy、2 sell、25 skip |

实际订单：

| 操作 | 股票 | 手数 |
|---|---|---:|
| 卖出 | 002206.SZ | 36 |
| 卖出 | 300822.SZ | 17 |
| 买入 | 300214.SZ | 23 |
| 买入 | 000570.SZ | 66 |
| 买入 | 002998.SZ | 5 |

### 方案 B

输入：沿用 B 的 20260529 持仓，组合估值约 1,025,794 元。  
输出目录：`outputs_scheme_b/trading_guides/20260601/`

| 项目 | 值 |
|---|---:|
| 交易日 | 2026-06-02 |
| 目标持仓 | 28 只 |
| 目标仓位 | 83% |
| 动态 K | 6（旧指南；当前配置已收紧为最多 3） |
| 实际买入 | 4 只 |
| 实际卖出 | 2 只 |
| full_positions 状态 | 4 buy、2 sell、24 skip |

实际订单：

| 操作 | 股票 | 手数 |
|---|---|---:|
| 卖出 | 000936.SZ | 35 |
| 卖出 | 002609.SZ | 28 |
| 买入 | 300214.SZ | 40 |
| 买入 | 300307.SZ | 45 |
| 买入 | 002695.SZ | 26 |
| 买入 | 601996.SH | 113 |

## 使用规则

1. 实盘下单只看 `trading_orders.csv`，不要把 full_positions 当下单清单。
2. `full_positions.csv` 现在可以辅助检查目标仓位，但只有 `操作=buy/sell` 且 `执行状态=本次执行` 的行才是本次动作；`skip` 行不是今天要下单。
3. 每天生成 guide 时必须传入真实当前持仓，格式是 `股票代码:股数`；如沿用历史手数口径，显式加 `--holdings-unit hands`。不要让 B 方案空仓重买，除非真实账户就是空仓。
4. 若使用旧版 `*_old` 文件，注意旧 `目标手数` 很多实际是“股数”；新版文件会同时给出股数和手数，下单优先看“股数”列。
5. 若当天已有持仓市值高于目标仓位，应优先少买或不买；现金约束已经默认开启。

## 验证情况

已通过：

```bash
python3 -m py_compile stock_dl/scripts/14_generate_trading_guide.py stock_dl/src/backtest/engine.py stock_dl/scripts/test_dynamic_k_scenarios.py
python3 stock_dl/scripts/test_dynamic_k_scenarios.py
```

已重生成：

```text
stock_dl/outputs_scheme_a/trading_guides/20260601/
stock_dl/outputs_scheme_b/trading_guides/20260601/
```

未完成的回测验证：

1. A 方案 `05_backtest.py` 未能运行，因为本地缺少 `outputs_scheme_a/val_predictions.csv`。
2. B 方案 `05_backtest.py` 未能运行，因为 `outputs_scheme_b/panel.parquet` 不是可读 parquet 文件，报 `Parquet magic bytes not found in footer`。

后续如果要做严格最优参数，需要先修复/重建这两个回测输入，再跑网格搜索；当前修复优先解决实盘执行风险。
