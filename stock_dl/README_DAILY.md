# A股量化交易系统 - 每日使用指南

## 📋 概述

本系统包含两个交易方案：

| 方案 | 模型 | 输出目录 | 特点 |
|------|------|----------|------|
| **方案A (scheme_a)** | GRU + Transformer (纯深度学习) | `outputs_scheme_a/` | 纯神经网络，无LightGBM |
| **方案B (scheme_b)** | GRU + Transformer + LightGBM | `outputs_scheme_b/` | DL+LGBM集成，效果通常更好 |

---

## 🚀 每日使用流程

### 第一次运行（需要训练模型）

```bash
cd stock_dl

# 方案A: 纯深度学习
./run_scheme_a.sh

# 方案B: 深度学习+LightGBM（推荐）
./run_scheme_b.sh
```

训练完成后，会在对应目录生成：
- `model.pt` - 深度学习模型
- `lgbm_model.txt` - LightGBM模型（仅方案B）
- `orders_YYYYMMDD.csv` - 交易信号

### 每日运行（模型已训练好）

```bash
cd stock_dl

# 方案A每日推理
./daily_infer.sh scheme_a

# 方案B每日推理（推荐）
./daily_infer.sh scheme_b
```

---

## 📁 输出文件说明

### 交易信号文件
- **`orders_YYYYMMDD.csv`** - 主要交易指令
  - `side`: buy(买入) / sell(卖出) / hold(持有)
  - `ts_code`: 股票代码
  - `score`: 模型评分
  - `estimated_shares`: 建议买入股数

### 评分文件
- **`scores_YYYYMMDD.csv`** - 所有股票的评分
  - `ts_code`: 股票代码
  - `score`: 综合评分
  - `score_deep`: 深度学习评分
  - `score_lgbm`: LightGBM评分（仅方案B）
  - `rank_deep`/`rank_lgbm`: 百分比排名

---

## 🔧 服务器提交命令

### 训练阶段（一次性）

```bash
# 连接服务器
ssh user@107.ustc.edu.cn

# 进入项目目录
cd /home/scc/pb23061103/finalex/finalex
git pull --ff-only origin main
cd stock_dl

# 激活环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dl-homework

# 方案A训练
sbatch jobs/tune_pure_dl.sbatch

# 方案B训练
sbatch jobs/tune_with_lgbm.sbatch
```

### 每日推理阶段

```bash
cd stock_dl

# 方案A每日推理
./daily_infer.sh scheme_a

# 方案B每日推理
./daily_infer.sh scheme_b

# 查看最新交易信号
cat outputs_scheme_b/orders_*.csv | tail -50
```

---

## 📊 数据更新说明

1. **数据目录**: `../A股数据/`
2. **每日数据更新后**，直接运行推理脚本即可使用新数据
3. **不需要重新训练模型**，除非模型效果明显下降

---

## ⚙️ 参数调优

如果需要重新调参，修改以下文件中的参数：

- 方案A参数: `configs/a_share_pure_dl.yaml`
- 方案B参数: `configs/a_share_with_lgbm.yaml`
- 网格搜索候选: `scripts/12_tune_once.py` 中的 `CURATED_CANDIDATES`

调参后重新运行训练即可。

---

## 🆘 常见问题

### Q: 模型文件不存在怎么办？
```bash
# 检查输出目录
ls -la outputs_scheme_*/

# 如果没有模型，需要先训练
./run_scheme_b.sh
```

### Q: 推理报错 "No latest scores generated"
```bash
# 检查数据是否更新
ls -la ../A股数据/daily/*.csv | tail -5

# 重新构建面板
python scripts/01_build_panel.py --config configs/a_share_with_lgbm.yaml
```

### Q: 如何查看回测结果？
```bash
cat outputs_scheme_b/backtest_metrics.json
```

---

## 📞 技术支持

- 项目仓库: git@github.com:wang-ava/makemoney.git
- 配置文件: `configs/a_share_*.yaml`
- 训练脚本: `scripts/03_train.py`
- 推理脚本: `scripts/06_infer_orders.py`
