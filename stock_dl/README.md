# 深度学习大作业 — 股票趋势预测与模拟交易

基于课程数据的端到端流水线：面板构建 → 多模型训练 → IC/回测/可视化 → **生成同花顺手动下单清单**。

当前版本已按 `项目完整方案_v2.md` 升级为“三层架构”：

1. 多维量价/技术/资金流/截面排序特征
2. GRU-Transformer 深度时序通道 + LightGBM LambdaRank 截面通道
3. 融合分数驱动的动态调仓策略

## 核心改进

### 模型架构（创新性15%）
- **GRU-Transformer**: Bi-GRU 捕捉局部时序，Transformer 捕捉全局依赖，Temporal Attention Pooling 输出可解释权重
- **LightGBM LambdaRank**: 每个交易日作为 query，直接优化 Top-K 横截面排序质量
- **Hybrid Blend**: 深度模型分数和 LightGBM 分数按日 rank 后网格搜索融合权重
- **BiLSTM + Multi-Head Attention**: 双向LSTM + 自注意力
- **GRU + Self-Attention**: 轻量级时序模型
- **Temporal CNN + Inception**: 多尺度卷积特征提取
- **TemporalAttentionRegressor**: Transformer编码器（原有）

### 特征工程（200+个特征）
- 量价技术指标：RSI、MACD、布林带
- **新增指标**：KDJ、CCI、Williams%R、OBV、ATR、Momentum、ROC、EMA比、OBV斜率、VWAP偏离
- 资金流特征：净流入比、主动买入比率、smart money
- 横截面rank与z-score

### 损失函数优化
- Huber Loss（稳健）
- IC Loss（直接优化横截面相关性）
- Top-Bottom排序损失
- 方向预测损失
- Focal Ranking损失
- Label Smoothing

### 训练策略
- 学习率预热（warmup）
- 梯度裁剪
- 验证IC早停
- 交易日分组 batch，便于 IC Loss / 排序损失学习横截面关系

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
│   ├── 10_train_lgbm.py        # LightGBM LambdaRank
│   └── 11_blend_scores.py      # 深度模型 + LGBM 融合
├── src/
│   └── models/
│       ├── gru_transformer.py   # GRU-Transformer [新增]
│       ├── lgbm_ranker.py       # LambdaRank 工具 [新增]
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

# 首次使用 W&B 时登录一次
wandb login

# 快速试跑
chmod +x run_all.sh
./run_all.sh configs/quick.yaml

# 正式训练（需GPU或107平台）
./run_all.sh configs/default.yaml
```

## 服务器 8 小时默认跑法

如果只想跑一组固定参数，提交：

```bash
cd stock_dl
mkdir -p logs
sbatch jobs/train.sbatch
```

这会使用 `configs/server_8h.yaml`，输出到 `outputs_server8h/`。该配置保留全市场、完整特征、GRU-Transformer + LightGBM 融合，但把深度模型改成中等宽度、限制训练时间、开启 AMP，并关闭 W&B gradient watch 来省时间。训练脚本会在 `train.max_minutes` 或 sbatch 传入的 `TRAIN_MAX_MINUTES` 到达后优雅停止，保存当前验证集最好的 `model.pt`，再继续跑融合、IC、回测和图表。

`jobs/train.sbatch` 默认 `SKIP_PANEL=auto`：如果 `outputs_server8h/panel.parquet` 已存在，会自动跳过面板重建；如果改了日期、特征或股票池，强制重建：

```bash
SKIP_PANEL=0 sbatch jobs/train.sbatch
```

默认预算是：需要重建 panel 时深度模型最多训练 270 分钟；复用已有 panel 时最多训练 330 分钟。想临时加长或缩短：

```bash
TRAIN_MAX_MINUTES=360 sbatch jobs/train.sbatch
```

想临时跑更大的正式配置：

```bash
CONFIG=configs/default.yaml SKIP_PANEL=0 sbatch jobs/train.sbatch
```

## 自动调参

如果要让服务器自动试参数，以后反复提交这个：

```bash
cd stock_dl
mkdir -p logs
sbatch jobs/tune.sbatch
```

每次提交会自动选择下一组未跑过的参数，输出到 `outputs_tuning/trials/trial_XXXX/`。第一次会构建一次共享面板 `outputs_tuning/shared_panel/panel.parquet`，后续 trial 会复用它。

调参汇总文件：

| 文件 | 说明 |
|------|------|
| `outputs_tuning/tuning_results.csv` | 所有 trial 的参数和指标 |
| `outputs_tuning/best_params.json` | 当前最佳 trial、指标、完整参数 |
| `outputs_tuning/best_params.yaml` | 同上，方便复制进配置文件 |
| `outputs_tuning/best_config.yaml` | 当前最佳 trial 的完整可运行配置 |
| `outputs_tuning/best_trial` | 指向当前最佳输出目录的软链接 |

默认按验证集 `ic_mean` 选最佳，同时记录回测收益、Sharpe、最大回撤、融合 alpha、最佳 epoch 等。想一次排多个并行 trial：

```bash
sbatch --array=1-4 jobs/tune.sbatch
```

想一次提交连续跑 2 组：

```bash
TRIALS_PER_JOB=2 sbatch jobs/tune.sbatch
```

## W&B 实验跟踪

`configs/default.yaml` 已开启 W&B：

```yaml
wandb:
  enabled: true
  entity: avawang1031
  project: my-awesome-project
```

训练脚本会逐 epoch 上传 `train/loss`、`val/ic_mean`、`val/icir`、学习率等指标；后续 IC、回测、基线对比和图表脚本会复用同一个 `WANDB_RUN_ID`，把最终指标、曲线图和关键 CSV/JSON 作为 artifact 传到同一次实验里。

如果只想本地跑，把配置改成：

```yaml
wandb:
  enabled: false
```

## 模型选择

在 `configs/default.yaml` 中选择模型：

```yaml
model:
  name: gru_transformer  # 可选: gru_transformer, temporal_attention, bilstm_attention, gru_attention, temporal_cnn, inception_time
  d_model: 128
  num_layers: 2
  dropout: 0.2
```

`default.yaml` 默认启用 LightGBM；如果本机未安装 `lightgbm`，脚本会自动跳过该通道并使用深度模型分数。安装 `requirements.txt` 后会生成 `lgbm_val_predictions.csv`、`blend_alpha_search.csv` 和 `val_predictions_blend.csv`。

## 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/panel.parquet` | 合并后面板（192特征） |
| `outputs/model.pt` | 模型权重 |
| `outputs/val_predictions.csv` | 验证集预测 |
| `outputs/val_predictions_deep.csv` | 深度时序模型预测 |
| `outputs/lgbm_val_predictions.csv` | LightGBM LambdaRank预测 |
| `outputs/blend_alpha_search.csv` | 融合权重搜索结果 |
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
