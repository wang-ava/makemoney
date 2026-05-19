# 深度学习大作业 — 股票趋势预测与模拟交易

基于课程数据的端到端流水线：面板构建 → 多模型训练 → IC/回测/可视化 → **生成同花顺手动下单清单**。

当前版本已从原始 MLP 模板升级为包含多种前沿模型的完整方案：

## 核心改进

### 模型架构（创新性15%）
- **BiLSTM + Multi-Head Attention**: 双向LSTM + 自注意力
- **GRU + Self-Attention**: 轻量级时序模型
- **Temporal CNN + Inception**: 多尺度卷积特征提取
- **TemporalAttentionRegressor**: Transformer编码器（原有）
- **Ensemble**: 多模型集成（开发中）

### 特征工程（192个特征）
- 量价技术指标：RSI、MACD、布林带
- **新增指标**：KDJ、CCI、Williams%R、OBV、ATR、Momentum、ROC
- 资金流特征：净流入比、主动买入比率、smart money
- 横截面rank与z-score

### 损失函数优化
- Huber Loss（稳健）
- Top-Bottom排序损失
- 方向预测损失
- Focal Ranking损失
- Label Smoothing

### 训练策略
- 学习率预热（warmup）
- 梯度裁剪
- 验证IC早停

## 目录结构

```
stock_dl/
├── configs/default.yaml    # 正式训练配置
├── configs/quick.yaml      # 本地快速试跑
├── scripts/
│   ├── 01_build_panel.py       # 面板构建
│   ├── 03_train.py             # 模型训练
│   ├── 04_eval_ic.py           # IC评估
│   ├── 05_backtest.py         # 回测
│   ├── 06_infer_orders.py      # 推理下单
│   ├── 07_visualize.py        # 可视化
│   ├── 08_baselines.py         # 基线对比
│   └── 09_diagnostics.py       # 诊断
├── src/
│   └── models/
│       ├── sequence.py          # Transformer
│       ├── lstm_attention.py   # BiLSTM+Attention [新增]
│       ├── gru_attention.py    # GRU+Attention [新增]
│       ├── cnn_temporal.py     # TemporalCNN [新增]
│       └── losses.py           # 增强损失函数
└── outputs/                    # 输出目录
```

## 本地运行

```bash
cd stock_dl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 快速试跑
chmod +x run_all.sh
./run_all.sh configs/quick.yaml

# 正式训练（需GPU或107平台）
./run_all.sh configs/default.yaml
```

## 模型选择

在 `configs/default.yaml` 中选择模型：

```yaml
model:
  name: bilstm_attention  # 可选: temporal_attention, bilstm_attention, gru_attention, temporal_cnn, inception_time
  d_model: 128
  num_layers: 2
  dropout: 0.2
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/panel.parquet` | 合并后面板（192特征） |
| `outputs/model.pt` | 模型权重 |
| `outputs/val_predictions.csv` | 验证集预测 |
| `outputs/ic_summary.json` | IC / ICIR |
| `outputs/backtest_metrics.json` | 年化收益、夏普、回撤 |
| `outputs/figures/*.png` | 报告图表 |
| `outputs/orders_*.csv` | **同花顺下单参考** |

## 防泄露要点

- 标签 `label` = 下一日收益；特征只用当日及以前
- 标准化在 **滑动窗口内部** 完成
- `train_end` / `val_end` 按 `trade_date` 切分
- 回测使用盘后信号在下一交易日开盘执行

## 评分要点

| 模块 | 分数 | 改进 |
|------|------|------|
| 算法合理性 | 10% | 192特征、增强风控 |
| **创新性** | **15%** | **多模型架构** |
| 模型效果 | 20% | 验证IC、回测收益 |
| 报告 | 25% | 完整报告结构 |
| 规范依从性 | 10% | 防泄露机制 |
| 可复现性 | 5% | 代码清晰、README完整 |
