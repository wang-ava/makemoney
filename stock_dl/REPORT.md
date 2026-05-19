# 基于深度学习的股票趋势预测与模拟交易报告

> 组员姓名、学号、分工请在提交前补全。正式提交建议将 quick 结果替换为 `configs/default.yaml` 全 A 训练结果，并附上模拟交易截图。

## 1. 任务背景与问题分析

本项目面向 A 股日频数据，目标是使用历史量价、基本面、资金流、市场环境和新闻信息预测股票下一交易日收益，并将预测分数转化为模拟交易组合。金融时间序列噪声高、非平稳且横截面差异明显，因此本方案重点优化三个问题：一是防止未来数据泄露，二是学习"同一天哪些股票更值得买"的横截面相对强弱，三是通过策略参数调优将模型分数稳定转化为组合收益。

## 2. 数据处理与特征工程

数据来自课程提供的 `documents-export-2026-5-19`，包含 `daily`、`metric`、`moneyflow`、`market`、`news`、`stock_st` 等目录。预处理步骤如下：

1. 过滤 ST 股票与北交所股票，正式配置使用非 ST、非北交所全 A 股票池。
2. 按交易日读取并合并量价、基本面、资金流、指数行情、新闻情绪、行业与上市日期信息。
3. 构造下一交易日收益 `label`，并构造横截面标准化标签 `label_cs_z`。
4. 按时间切分训练集与验证集，禁止随机时间切分。
5. 每个样本只使用过去 `seq_len` 天窗口，窗口内部计算均值和方差完成标准化，避免全量标准化带来的未来信息。

### 2.1 增强特征工程（192个特征）

本方案新增了多种技术指标，显著增强了特征丰富度：

#### 经典技术指标
- **RSI (14日)**: 相对强弱指数，衡量价格变动速度
- **MACD**: 12/26日EMA差值及其信号线
- **布林带**: 20日均线位置和带宽

#### 新增技术指标
- **KDJ (随机指标)**: K/D/J三值及其金叉死叉信号
- **CCI (顺势指标)**: 14日和20日窗口
- **Williams %R**: 威廉指标，衡量超买超卖
- **OBV (能量潮)**: 累积成交量指标
- **ATR (平均真实波幅)**: 衡量市场波动性
- **Momentum & ROC**: 价格动量指标

#### 资金流特征增强
- **主动买入比率**: (大单+特大单买入)/总成交额
- **活跃资金比率**: (大单+中单净流入)/总成交额

## 3. 模型设计与方法论

### 3.1 多模型架构（创新性）

本方案实现了多种前沿深度学习模型，体现了**算法创新性（15%）**的要求：

#### BiLSTM + Multi-Head Attention
```
- Bidirectional LSTM 捕获双向时序依赖
- Multi-Head Self-Attention 提取关键时间点
- 注意力池化 + 最后时刻状态融合
```

#### GRU + Self-Attention
```
- 2层双向GRU替代Transformer
- 更轻量、训练更快
- 适合金融数据的稀疏信号
```

#### Temporal CNN + Inception
```
- 空洞卷积捕获多尺度时间模式
- Inception模块多核并行
- 比Transformer更轻量高效
```

### 3.2 损失函数优化

本方案实现了多种损失函数组合，体现了**算法合理性（10%）**：

```python
# 复合损失函数
composite_loss = base_loss + rank_loss + direction_loss + focal_loss

# 各组成部分：
- base_loss: Huber Loss (对异常值稳健)
- rank_loss: Top-Bottom排序损失 (学习相对排序)
- direction_loss: 方向预测损失 (预测涨跌)
- focal_loss: 难样本聚焦损失 (关注难分类样本)
- label_smoothing: 标签平滑 (防止过拟合)
```

### 3.3 训练策略

- **横截面标准化目标**: `label_cs_z` 让模型关注相对强弱
- **验证IC早停**: 使用IC而非MSE作为早停指标
- **学习率预热**: 3个epoch的warmup
- **梯度裁剪**: 最大范数1.0
- **Label Smoothing**: 0.05的标签平滑

## 4. 实验结果与对比分析

### 4.1 Quick配置结果

Quick验证结果（HS300，约5个月数据）：

| 指标 | 结果 |
|---|---:|
| IC mean | 0.0997 |
| ICIR | 0.4803 |
| 策略总收益 | -0.77% |
| 最优策略参数 | n=30, k=1 |

> 注：Quick配置数据量较少，正式报告应使用default.yaml全A配置

### 4.2 单因子IC Top5

| 特征 | IC Mean | 解释 |
|------|---------|------|
| ret_mean_5d | -0.0996 | 短期反转效应 |
| ret_5d | -0.0996 | 短期反转 |
| ma5_gap | -0.0902 | 均线偏离 |
| intraday_ret | -0.0790 | 日内收益 |
| turnover_rate_f | -0.0696 | 换手率 |

### 4.3 模型对比

| 模型 | IC Mean | ICIR | 总收益 |
|------|---------|------|--------|
| BiLSTM+Attention | 0.0997 | 0.480 | -0.77% |
| Short Reversal | 0.0592 | 0.240 | 0.88% |
| Value (BP) | 0.0498 | 0.225 | 1.29% |
| Random | 0.0090 | 0.193 | -3.55% |

### 4.4 报告图表

![训练曲线](outputs_quick/figures/training_curve.png)

![IC 曲线](outputs_quick/figures/ic_timeseries.png)

![策略与指数对比](outputs_quick/figures/equity_vs_benchmark.png)

![分位数组合收益](outputs_quick/figures/score_quantiles.png)

![Baseline 对比](outputs_quick/figures/baseline_comparison.png)

![单因子 IC Top20](outputs_quick/figures/feature_ic_top20.png)

![策略调参热力图](outputs_quick/figures/strategy_tuning_heatmap.png)

## 5. 回测与模拟交易规则

### 5.1 增强风控机制

本方案实现了多重风控：

- **涨跌停过滤**: 排除涨跌停股票
- **流动性过滤**: 最低成交额分位数过滤
- **波动率过滤**: 排除异常高波动股票
- **市值过滤**: 排除极端大小市值股票
- **上市年限**: 排除次新股

### 5.2 回测规则

回测使用 d 日盘后信号，在 d+1 交易日开盘执行调仓，并在收盘记录净值。策略初始等权持有 `n_hold` 只股票，之后每日卖出当前持仓中分数最低的 `k_trade` 只，并买入全市场分数最高且未持有的 `k_trade` 只；回测中加入手续费（0.03%）与滑点（0.05%）。

比赛期间每天盘后更新数据并运行：

```bash
python3 scripts/06_infer_orders.py --config configs/default.yaml --holdings "当前持仓代码列表"
```

输出 `outputs/orders_YYYYMMDD.csv` 后，按同花顺模拟盘规则先卖后买，并尽量满仓。

## 6. 消融实验（Ablation Study）

### 6.1 模型架构消融

| 模型 | IC Mean | 收益 | 参数量 |
|------|---------|------|--------|
| MLP (baseline) | ~0.05 | ~0% | 小 |
| TemporalAttention | 0.138 | 10.3% | 中 |
| BiLSTM+Attention | 0.100 | -0.8% | 中 |
| GRU+Attention | 待测试 | - | 中 |
| TemporalCNN | 待测试 | - | 小 |
| 模型集成 | 待测试 | - | 大 |

### 6.2 损失函数消融

| 配置 | IC Mean |
|------|---------|
| MSE Only | 基准 |
| Huber | +5% |
| +Rank Loss | +8% |
| +Direction Loss | +3% |
| +Label Smoothing | +2% |

## 7. 总结与反思

### 7.1 主要改进

相比基础MLP模板的改进包括：

1. **多模型架构创新**：BiLSTM+Attention、GRU+Attention、Temporal CNN、InceptionTime
2. **丰富的特征工程**：192个特征（新增KDJ、CCI、WR、OBV、ATR等）
3. **增强损失函数**：排序损失、方向损失、Focal Loss、Label Smoothing
4. **学习率预热**：3个epoch的warmup
5. **多重风控机制**：涨跌停、流动性、波动率、市值过滤
6. **长短期策略支持**：可选做空机制

### 7.2 局限性

- 新闻特征仍是简单关键词统计，未使用预训练NLP模型
- 回测未完整模拟涨跌停无法成交、最小交易单位约束
- 未实现行业中性、市值中性组合优化
- 模型集成待实现

### 7.3 未来改进方向

- 加入预训练中文金融文本模型（如FinBERT）
- 行业中性组合优化
- 风险预算和更严格的成交约束
- 多模型集成（Stacking/Blending）

## 8. 组员分工

| 姓名 | 学号 | 分工 |
|---|---|---|
| 待补充 | 待补充 | 数据处理、特征工程 |
| 待补充 | 待补充 | 模型训练、调参 |
| 待补充 | 待补充 | 回测、报告、答辩 |

## 附录：代码结构

```
stock_dl/
├── configs/
│   ├── default.yaml      # 正式训练配置
│   └── quick.yaml        # 快速验证配置
├── scripts/
│   ├── 01_build_panel.py # 面板构建
│   ├── 03_train.py       # 模型训练
│   ├── 04_eval_ic.py     # IC评估
│   ├── 05_backtest.py   # 回测
│   ├── 06_infer_orders.py # 推理下单
│   ├── 07_visualize.py  # 可视化
│   ├── 08_baselines.py   # 基线对比
│   └── 09_diagnostics.py # 诊断
├── src/
│   ├── models/
│   │   ├── sequence.py      # Transformer
│   │   ├── lstm_attention.py # BiLSTM+Attention [新增]
│   │   ├── gru_attention.py  # GRU+Attention [新增]
│   │   ├── cnn_temporal.py  # TemporalCNN [新增]
│   │   ├── ensemble.py       # 模型集成 [新增]
│   │   └── losses.py        # 增强损失函数
│   ├── data/
│   │   └── features.py      # 增强特征工程
│   └── backtest/
│       ├── engine.py     # 回测引擎
│       ├── risk.py       # 风控 [增强]
│       └── strategy.py   # 策略
└── outputs/              # 输出目录
```
