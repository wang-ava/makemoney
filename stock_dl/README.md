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

### 特征工程（250+个特征）
- 量价技术指标：多周期收益、波动率、RSI、MACD、布林带、KDJ、CCI、Williams%R、OBV、ATR、Momentum、ROC
- 增强指标：Stochastic、ADX、Supertrend、PSY、量能突变、持续放量、价格区间位置
- 资金流/估值/规模：主力资金、smart money、active money、PE/PB/PS、EP/BP、市值与行业相对估值
- 横截面 rank、z-score、行业内 rank 与规模中性特征

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
│   ├── 09_diagnostics.py       # 诊断
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

# 正式训练（需GPU或107平台；默认对齐 best trial）
./run_all.sh configs/default.yaml
```

## 107 平台 8 小时默认跑法

首次在 107 平台跑之前，先确认环境装好依赖，尤其是 `lightgbm`：

```bash
cd stock_dl
conda activate dl-homework
pip install -r requirements.txt
python - <<'PY'
import lightgbm, torch
print("lightgbm ok")
print("cuda:", torch.cuda.is_available())
PY
```

之后建议直接用 sbatch 跑 `configs/server_8h.yaml`。这份配置已对齐当前 best trial 的深度模型和策略参数，并启用 LightGBM LambdaRank：GRU-Transformer 两层、`n_hold=30`、`k_trade=1`、`cash_reserve_ratio=0.2`，即按老师放宽口径保持约 80% 以上仓位。

```bash
cd stock_dl
mkdir -p logs
sbatch jobs/train.sbatch
```

这会输出到 `outputs_server8h/`。训练脚本会在 `train.max_minutes` 或 sbatch 传入的 `TRAIN_MAX_MINUTES` 到达后优雅停止，保存当前验证集最好的 `model.pt`，再继续跑融合、IC、回测、基线对比、图表和下单清单。

`jobs/train.sbatch` 默认 `SKIP_PANEL=auto`：如果 `outputs_server8h/panel.parquet` 已存在，会自动跳过面板重建；如果改了日期、特征或股票池，强制重建：

```bash
SKIP_PANEL=0 sbatch jobs/train.sbatch
```

默认预算是：需要重建 panel 时深度模型最多训练 270 分钟；复用已有 panel 时最多训练 330 分钟。想临时加长或缩短：

```bash
TRAIN_MAX_MINUTES=360 sbatch jobs/train.sbatch
```

想临时跑正式配置：

```bash
CONFIG=configs/default.yaml SKIP_PANEL=0 sbatch jobs/train.sbatch
```

查看运行日志：

```bash
tail -f logs/stock-dl-full_<jobid>.out
tail -f logs/stock-dl-full_<jobid>.err
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
  d_model: 96
  num_layers: 2
  dropout: 0.22
```

`default.yaml` 默认启用 LightGBM；如果本机未安装 `lightgbm`，脚本会自动跳过该通道并使用深度模型分数。安装 `requirements.txt` 后会生成 `lgbm_val_predictions.csv`、`blend_alpha_search.csv` 和 `val_predictions_blend.csv`。

## 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/panel.parquet` | 合并后面板（250+特征） |
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
| 算法合理性 | 10% | 全市场面板、250+特征、时间切分、防泄露标准化 |
| **创新性** | **15%** | **GRU-Transformer、LambdaRank、融合与动态调仓** |
| 模型效果 | 20% | 验证IC、回测收益、基准对比 |
| 报告 | 25% | 完整报告结构 |
| 规范依从性 | 10% | 防泄露机制、涨跌停/流动性过滤、约80%仓位合规 |
| 可复现性 | 5% | 代码清晰、README完整 |

## 当前 best trial 结果

`best-trails/` 保存了当前最佳实验记录：验证期为 2025-01-02 至 2026-05-15，回测交易日至 2026-05-18。

| 指标 | 结果 |
|------|------|
| 总收益 | 59.80% |
| 年化收益 | 43.35% |
| Sharpe | 1.605 |
| 最大回撤 | -18.41% |
| IC Mean | 10.83% |
| ICIR | 1.017 |

注意：这次 best-trails 记录里 `lgbm_status.json` 为 `missing_dependency`，所以 59.80% 是深度模型通道结果；107 平台装好 `lightgbm` 后会启用双通道融合。`cash_reserve_ratio=0.2` 的主要价值是满足 80% 仓位口径并保留实盘/模拟盘机动现金；它不保证在所有预测文件上提高绝对收益。当前同一组本地预测的复测显示，20%现金通常会改善 Sharpe/回撤，但绝对收益可能小幅下降。
