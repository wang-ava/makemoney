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
- 配置文件: `configs/local_scheme_*.yaml`
- 训练脚本: `scripts/03_train.py`
- 推理脚本: `scripts/06_infer_orders.py`
- 交易指南生成: `scripts/14_generate_trading_guide.py`
- k_trade 调优: `scripts/13b_ktrade_scan.py`
- 动态 k 单元测试: `scripts/test_dynamic_k_scenarios.py`

---

## 🆕 最近更新（2026-06-01）

### 1. 持仓质量驱动的动态换手（k_trade）

**问题**：固定 k_trade=2 太保守，不能应对持仓质量差异。
- 如果持仓都很烂，只换 2 只不合理
- 50 只持仓下 25 天才能换血

**改造**（[scripts/14_generate_trading_guide.py:127-191](scripts/14_generate_trading_guide.py#L127-L191) + [src/backtest/engine.py:107-160](src/backtest/engine.py#L107-L160)）：

```
k = base_k  # 起步
if 烂股数 >= bad_min:
    k = max(k, 烂股数 + bad_buffer)  # 烂股数决定下限
if 好候选数 >= bad_min and gap > 0.10:
    k += dynamic_k_step  # 候选好且差距大
if 烂股数 == 0 and gap < 0.02:
    k -= dynamic_k_step  # 持仓都好且差距小
k = min(k, max(烂股数, 好候选数))  # 理论换手上限
return max(1, k)
```

**关键定义**：
- 烂股 = 持仓中分数 < 候选池中位数的股
- 好候选 = 候选中分数 > 持仓均分的股
- gap = 候选最高分 - 持仓最低分
- max_k 默认 = max(烂股数, 好候选数)（理论换手上限，跟随持仓规模 0-50）

**配置项**（`configs/local_scheme_a.yaml:81-87`）：
```yaml
dynamic_k: true
dynamic_k_bad_min: 1        # 烂股阈值
dynamic_k_bad_buffer: 0     # 烂股基础上额外加几个
# dynamic_k_max: 跟随 max(烂股数, 好候选数)
debug_dynamic_k: true       # 调试输出（建议关掉）
```

**回测最优 base_k = 6**（[13b_ktrade_scan.py](scripts/13b_ktrade_scan.py) 1-20 扫描）：
- 年化 42.93%, Sharpe 1.604
- 比 k_trade=2 提升 +45.66% 年化、+1.620 Sharpe

**场景验证**（[test_dynamic_k_scenarios.py](scripts/test_dynamic_k_scenarios.py)）：

| 场景 | base_k | 烂股 | 决策 k |
|---|---|---|---|
| A 烂持仓 5 全烂，候选好，+2 | 6 | 5 | 8 ✅ |
| B 好持仓 5 全好，无好候选，-2 | 6 | 0 | 1 ✅ |
| C 混合 3 烂 + 2 好，max_k=4 | 6 | 3 | 4 ✅ |
| D 极端烂 10 全烂，max_k=10 | 6 | 10 | 10 ✅ |
| E base_k=2 烂持仓，max_k=10 | 2 | 5 | 7 ✅ |
| F 烂持仓 + 候选极好，+2 | 6 | 3 | 8 ✅ |

### 2. 同花顺手动下单价格优化（sell/buy slack）

**问题**：原版用昨日收盘价作为买卖建议价，同花顺满仓时按收盘价卖会挂很久才成交。

**改造**（[scripts/14_generate_trading_guide.py:315-373](scripts/14_generate_trading_guide.py#L315-L373)）：
- 卖价 = `prev_close * (1 - sell_slack)` → 让卖单快速成交
- 买价 = `prev_close * (1 + buy_slack)` → 让买单优先吃单
- 默认 sell_slack=0.002 (-0.2%), buy_slack=0.001 (+0.1%)
- 涨跌停限制（主板 ±10%）

**配置项**（`configs/local_scheme_a.yaml:88-91`）：
```yaml
sell_slack: 0.002
buy_slack: 0.001
limit_pct: 0.10
```

### 3. 指南输出格式升级

**新增段**（在卖出清单前）：
- 【执行流程 - 先卖后买】4 步操作（9:30-9:35 卖、9:35-9:45 等资金、9:35-10:00 买）
- 资金时序（卖出总额 / 买入总额 / 卖出后可用 / 资金缺口）

**清单表格新增列**：
- 卖出清单：优先级 / 代码 / 手数 / 收盘价 / **推荐挂单价** / **折让** / 风险
- 买入清单：优先级 / 代码 / 手数 / 收盘价 / **推荐挂单价** / **溢价** / 金额 / 风险

**示例**：
```
1    002998.SZ    3700     8.24     8.25        +0.10%  30,525  ⚠️流动性差
                                ^         ^
                            收盘价   推荐挂单价（含 0.1% 溢价）
```

### 4. k_trade 扫描脚本

**新增** [scripts/13b_ktrade_scan.py](scripts/13b_ktrade_scan.py)：
- 扫描 n_hold × k_trade 完整笛卡尔积
- 复用 14 脚本的 strategy_cfg（dynamic_k=True, dynamic_position=True, score_gap_trigger=True）
- 输出 CSV + Top 10 + 全局最优
- 默认用 `quick.yaml` 数据（29 天 val 期间），约 5 分钟跑完

**用法**：
```bash
cd stock_dl
python3 scripts/13b_ktrade_scan.py
# 输出：outputs_quick/ktrade_comparison.csv
# 也复制到 outputs_scheme_a/ktrade_comparison.csv
```

### 5. 配置文件更新

| 文件 | 变更 |
|---|---|
| `configs/local_scheme_a.yaml` | k_trade: 2 → 6；新增 sell/buy slack；新增 dynamic_k_bad_min/bad_buffer；新增 debug_dynamic_k |
| `configs/local_scheme_b.yaml` | 同上 |
| `scripts/14_generate_trading_guide.py` | get_prices 加 slack；generate_trading_guide 加执行流程/资金时序/推荐挂单价；_choose_dynamic_k 重写 |
| `src/backtest/engine.py` | _choose_dynamic_k 同步重写（保持回测一致） |

### 6. 验证结果

新指南（`outputs_scheme_a/trading_guides/20260529/trading_guide.txt`）包含：
- ✅ 策略参数区显示 `卖出折让 -0.20% | 买入溢价 +0.10%`
- ✅ 【执行流程 - 先卖后买】4 步操作说明
- ✅ 资金时序：卖出总额 / 买入总额 / 资金缺口 +712,376 元（✅ 充足）
- ✅ 买入清单 8 只，每只带"收盘价 / 推荐挂单价 / 溢价"三列
- ✅ 备选替换表用新价格

### 7. 使用方法

**日常生成交易指南**：
```bash
cd stock_dl
python3 scripts/14_generate_trading_guide.py \
    --config configs/local_scheme_a.yaml \
    --holdings "000001.SZ:1000,600000.SH:500" \
    --portfolio-value 1000000
# 输出：outputs_scheme_a/trading_guides/YYYYMMDD/trading_guide.txt
```

**调试动态 k 决策**：
```yaml
# configs/local_scheme_a.yaml
debug_dynamic_k: true   # 打印烂股数/好候选数/gap
```

**重新跑 k_trade 扫描**：
```bash
cd stock_dl
python3 scripts/13b_ktrade_scan.py
```

**场景单元测试**：
```bash
cd stock_dl
python3 scripts/test_dynamic_k_scenarios.py
```
