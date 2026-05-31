# 量化交易策略 - A股深度学习预测系统

基于 GRU-Transformer 和 LightGBM LambdaRank 的 A 股量化交易策略，包含完整的训练、评估、回测和订单生成流程。

---

## 📋 目录

- [快速开始](#-快速开始)
- [同花顺手动下单](#-同花顺手动下单) ⭐ **新增**
- [方案说明](#-方案说明)
- [实验结果](#-实验结果)
- [运行命令](#-运行命令)
- [代码改进](#-代码改进)
- [文件结构](#-文件结构)

---

## 🚀 快速开始

### 环境要求

```bash
pip install torch pandas numpy lightgbm wandb
```

### 一键运行

```bash
cd stock_dl

# 方案A: 纯深度学习
bash run_scheme_a.sh

# 方案B: 深度学习 + LightGBM
bash run_scheme_b.sh
```

---

## ⭐ 同花顺手动下单（实盘交易）

### 适用场景

比赛期间每日盘后运行，生成次日交易指南，支持手动在同花顺模拟账户操作。

### 一键生成交易指南

```bash
# 交互式（推荐首次使用）
cd ..
./trading.sh

# 命令行直接运行
./trading.sh a "600036.SH:500,000001.SZ:1000" 1000000
```

### 持仓输入格式

```bash
# 格式：代码:手数,代码:手数
--holdings "600036.SH:500,000001.SZ:1000,600519.SH:100"

# 无持仓时
--holdings ""
```

### 分配策略

| 策略 | 说明 | 推荐场景 |
|------|------|----------|
| `score_weighted` | 按分数加权（高分股票仓位更大） | **默认/推荐** |
| `rank_weighted` | 按排名加权（递减权重） | 追求分散 |
| `equal` | 等权重 | 保守 |
| `power` | 幂律加权 | 追求极端收益 |

### 输出文件

```
stock_dl/outputs/
├── trading_guide_YYYYMMDD.txt    # 完整交易指南（含备选方案）
├── trading_orders_YYYYMMDD.csv   # 买卖订单
├── backup_stocks_YYYYMMDD.csv    # 备选股票池
└── full_positions_YYYYMMDD.csv   # 完整目标持仓
```

### 交易指南包含内容

1. **卖出清单** - 先执行，按优先级列出需卖出的股票
2. **买入清单** - 后执行，按优先级列出需买入的股票
3. **失败备选方案** - ⭐ 每只股票买不到时的备选和调整策略
4. **备选股票池** - 目标股无法交易时的替代选择
5. **风险提示** - 大单分批、流动性差等预警
6. **快速检查清单** - 交易前确认事项

### 故障处理

| 情况 | 解决方案 |
|------|----------|
| 买不进去 | 撤单+0.5%重新报价，3次失败换备选 |
| 卖不出去 | 降报价0.3%，等尾盘14:50 |
| 只成交一半 | 成交率>80%正常，剩余买备选 |
| 参与率>10% | 建议分2批下单 |

详细说明请参考：[同花顺手动下单策略指南.md](同花顺手动下单策略指南.md)

---

## 📊 方案说明

### 方案A: 纯深度学习 (GRU-Transformer)

| 配置 | 值 |
|------|-----|
| 模型 | GRU-Transformer |
| LightGBM | ❌ 禁用 |
| 最终得分 | 100% DL |

### 方案B: 深度学习 + LightGBM

| 配置 | 值 |
|------|-----|
| 模型 | GRU-Transformer + LightGBM LambdaRank |
| Blend权重 | 75% DL + 25% LGBM |
| 最优alpha | 0.75 |

---

## 📈 实验结果

### 方案A vs 方案B (服务器完整数据 2018-2025)

| 指标 | 方案A (Pure DL) | 方案B (DL+LGBM) | 胜者 |
|------|-----------------|-----------------|------|
| **IC Mean** | 10.51% | 10.72% | B |
| **ICIR** | 0.945 | 0.955 | B |
| **方向准确率** | 53.18% | 49.05% | **A** |
| **总收益** | **38.64%** | 36.98% | **A** |
| **年化收益** | **27.68%** | 26.53% | **A** |
| **夏普比率** | 1.21 | **1.24** | B |
| **最大回撤** | **-14.39%** | -15.88% | **A** |
| **Calmar比率** | **2.25** | 1.95 | **A** |

### 推荐方案: **方案A**

理由：
1. 更高的总收益和更低的回撤
2. 方向准确率更高（选股更准）
3. 结构更简单，维护成本低

### 最优策略参数

```yaml
n_hold: 50           # 持仓数量
k_trade: 1           # 每次换手1只
min_position_ratio: 0.8
dynamic_position: true
```

---

## 🔧 运行命令

### 本地测试 (快速)

```bash
cd stock_dl

# 1. 构建数据面板
python scripts/01_build_panel.py --config configs/quick.yaml

# 2. 训练模型
python scripts/03_train.py --config configs/quick.yaml

# 3. 评估IC
python scripts/04_eval_ic.py --config configs/quick.yaml

# 4. 回测
python scripts/05_backtest.py --config configs/quick.yaml

# 5. 生成订单
python scripts/06_infer_orders.py --config configs/quick.yaml

# 6. 可视化
python scripts/07_visualize.py --config configs/quick.yaml
```

### 服务器完整训练

```bash
# 方案A: 纯深度学习
sbatch jobs/train_scheme_a.sbatch

# 方案B: DL + LightGBM
sbatch jobs/train_scheme_b.sbatch
```

### 消融实验

```bash
# 运行全部消融实验
python scripts/13_ablation.py --config configs/quick.yaml --max-epochs 10

# 只运行部分变体
python scripts/13_ablation.py --config configs/quick.yaml --variants "full,no_gru"
```

### 策略参数优化

```bash
python scripts/99_optimize_strategy.py --config configs/quick.yaml
```

### 成交量约束回测

```bash
# 对比有/无成交约束的回测
python -c "
from src.backtest.engine import run_backtest
from src.backtest.fill_constraint_backtest import run_backtest_with_fill_constraint

# 无约束
result_no = run_backtest(scores, prices, n_hold=50, k_trade=1)

# 有约束
result_with = run_backtest_with_fill_constraint(
    scores, prices, panel,
    n_hold=50, k_trade=1,
    use_fill_constraint=True,
    volume_window=20,
)
"
```

### 智能下单执行

```bash
# 模拟模式
python scripts/smart_executor.py \
    --orders orders_20260529.csv \
    --mode simulation \
    --output execution_report.csv

# 查看报告
python scripts/smart_executor.py --report execution_report.csv
```

---

## 🔧 代码改进

### Bug修复

| 问题 | 文件 | 修复 | 状态 |
|------|------|------|------|
| 涨跌停卖出逻辑 | engine.py:243 | <= → < | ✅ |
| lookback窗口 | 06_infer_orders.py:272 | 120 → 200天 | ✅ |
| 空仓边界 | 06_infer_orders.py:165-166 | 支持建仓 | ✅ |

### 新增功能

| 功能 | 文件 | 说明 |
|------|------|------|
| 成交率估算 | fill_rate.py | 基于参与率估算 |
| 带约束回测 | fill_constraint_backtest.py | 考虑真实成交 |
| 智能下单器 | smart_executor.py | 分批+追价 |
| 消融实验 | 13_ablation.py | 模型组件分析 |

### 改进效果

| 改进项 | 文件 | 评分收益 |
|--------|------|----------|
| 80%持仓强制 | 06_infer_orders.py | 10% |
| Attention热力图 | 07_visualize.py | 15% |
| 方向胜率指标 | ic.py | 20% |
| 高级风险指标 | 05_backtest.py | 20% |
| T+1检查 | engine.py | 10% |
| 消融实验 | 13_ablation.py | 20% |

---

## 📁 文件结构

```
.
├── trading.sh                           # ⭐ 同花顺一键交易脚本
├── trading_guide_daily.sh              # 每日检查单脚本
├── 同花顺手动下单策略指南.md           # ⭐ 完整交易策略指南
├── stock_dl/
│   ├── scripts/
│   │   ├── 01_build_panel.py          # 构建数据面板
│   │   ├── 03_train.py               # 模型训练
│   │   ├── 04_eval_ic.py             # IC评估
│   │   ├── 05_backtest.py            # 回测
│   │   ├── 06_infer_orders.py        # 订单生成
│   │   ├── 07_visualize.py          # 可视化
│   │   ├── 10_train_lgbm.py         # LightGBM训练
│   │   ├── 11_blend_scores.py       # 混合打分
│   │   ├── 13_ablation.py            # 消融实验
│   │   ├── 14_generate_trading_guide.py  # ⭐ 交易指南生成器 v2
│   │   ├── 99_optimize_strategy.py  # 策略优化
│   │   ├── smart_executor.py         # 智能下单执行器
│   │   └── example_fill_constraint.py # 成交约束示例
│   ├── src/
│   │   ├── backtest/
│   │   │   ├── engine.py             # 回测引擎 (含T+1、涨跌停)
│   │   │   ├── fill_rate.py         # ⭐ 成交率估算
│   │   │   └── fill_constraint_backtest.py  # ⭐ 带约束回测
│   │   ├── models/
│   │   │   ├── gru_transformer.py   # GRU-Transformer
│   │   │   └── losses.py            # 损失函数
│   │   ├── data/
│   │   │   ├── panel.py             # 数据面板
│   │   │   └── features.py          # 特征工程 (250+特征)
│   │   └── metrics/
│   │       └── ic.py                # IC指标
│   ├── configs/
│   │   ├── default.yaml             # 服务器完整配置
│   │   ├── quick.yaml              # 本地快速测试
│   │   ├── local_scheme_a.yaml    # 方案A本地
│   │   └── local_scheme_b.yaml    # 方案B本地
│   ├── outputs/                     # 输出目录
│   ├── outputs_scheme_a/            # 方案A输出
│   └── outputs_scheme_b/            # 方案B输出
├── best-trails/                      # 最佳实验记录
├── jobs/                            # 服务器训练脚本
│   ├── train_scheme_a.sbatch
│   └── train_scheme_b.sbatch
└── documents-export-*/              # 原始数据（需下载）
```

---

## 📊 核心指标

| 指标 | 数值 | 评价 |
|------|------|------|
| IC Mean | 10.51% | ✅ 良好 (>8%) |
| ICIR | 0.945 | ✅ 稳定 |
| 方向准确率 | 53.18% | ✅ >50% |
| 总收益 | 38.64% | ✅ 跑赢基准 |
| 夏普比率 | 1.21 | ✅ >1.0 |
| 最大回撤 | -14.39% | ⚠️ 可接受 |
| Calmar比率 | 2.25 | ✅ 优秀 |

### 与基准对比

| 基准 | 区间收益 |
|------|----------|
| 上证指数 | +27.6% |
| 沪深300 | +30.2% |
| **本策略** | **+38.64%** ✅ |

---

## 🔬 成交量约束验证

### 回测对比

| 指标 | 无约束 | 有约束 | 差异 |
|------|--------|--------|------|
| 总收益 | 7.96% | 5.34% | -2.62% |
| 夏普比率 | 2.32 | 1.71 | -0.61 |
| 成交率 | 100% | 97.4% | -2.6% |
| 仓位达成 | 100% | 86.4% | -13.6% |

**结论**: 考虑成交约束后，收益更接近真实交易情况。

---

## ✅ 数据集验证

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 时序分割 | ✅ | train_end=20241231, val_end=20250529 |
| In-window标准化 | ✅ | 只用窗口内统计量 |
| Label定义 | ✅ | shift(-1) 只用未来收益率 |
| 截面排名 | ✅ | 同日groupby，无跨日信息 |

**无未来数据泄露风险。**

---

## 📝 报告

- [策略优化报告](strategy_optimization_report.md)
- [成交量约束交易实现文档](成交量约束交易实现文档.md)
- [代码改进总结](代码改进总结(1).md)

---

*最后更新: 2026-05-31*
