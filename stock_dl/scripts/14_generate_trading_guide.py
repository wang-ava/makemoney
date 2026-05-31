#!/usr/bin/env python3
"""
同花顺手动下单 - 动态持仓交易指南生成器 v2

核心改进：
1. 按分数动态分配权重（高分股票仓位更大）
2. 支持带手数的持仓输入
3. 生成完整的备选股票池
4. 包含买/卖失败时的调整方案
5. 分批下单策略

用法:
    # 完整命令
    python scripts/14_generate_trading_guide.py \
        --config configs/local_scheme_a.yaml \
        --holdings "000001.SZ:1000,600000.SH:500" \
        --portfolio-value 1000000 \
        --allocation score_weighted \
        --scheme-name "方案A(纯DL)"

    # 交互式
    python scripts/14_generate_trading_guide.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.backtest.engine import choose_target_position
from src.backtest.risk import attach_buyable_flag
from src.data.features import add_features, feature_columns
from src.data.panel import build_panel
from src.models.factory import build_model_from_checkpoint


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class HoldingStock:
    ts_code: str
    shares: int
    price: float = 0.0


@dataclass
class TargetStock:
    ts_code: str
    rank: int
    score: float
    weight: float
    target_shares: int
    target_value: float
    current_shares: int
    action: str  # buy, sell, hold
    action_shares: int
    buy_price: float
    sell_price: float
    liquidity_score: float  # 0-1, 流动性评分
    risk_flag: str = ""  # 风险提示


# ============================================================================
# 持仓解析
# ============================================================================

def parse_holdings(holdings_str: str) -> list[HoldingStock]:
    holdings = []
    if not holdings_str.strip():
        return holdings

    for item in holdings_str.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        code, shares_str = item.split(":")
        shares = int(shares_str)
        holdings.append(HoldingStock(ts_code=code.strip(), shares=shares))
    return holdings


# ============================================================================
# 分配策略
# ============================================================================

def calculate_weights(scores: pd.Series, n_hold: int, strategy: str = "score_weighted") -> pd.Series:
    """计算权重"""
    top_n = scores.nlargest(n_hold)

    if strategy == "equal":
        return pd.Series(1.0 / len(top_n), index=top_n.index)
    elif strategy == "score_weighted":
        total = top_n.sum()
        if total == 0:
            return pd.Series(1.0 / len(top_n), index=top_n.index)
        weights = top_n / total
        return weights / weights.sum()
    elif strategy == "rank_weighted":
        n = len(top_n)
        return pd.Series({code: 2 * (n - i) / (n * (n + 1)) for i, code in enumerate(top_n.index)})
    else:
        # power
        powered = top_n ** 2
        return (powered / powered.sum())


def calculate_liquidity_score(panel: pd.DataFrame, ts_code: str, window: int = 20) -> float:
    """计算流动性评分 (0-1)"""
    try:
        data = panel[panel["ts_code"] == ts_code].sort_values("trade_date").tail(window)
        if len(data) < 5:
            return 0.5

        # 用成交量变异系数和日均成交额评估流动性
        avg_amount = data["amount"].mean()
        std_amount = data["amount"].std()
        cv = std_amount / (avg_amount + 1)

        # 成交额越大、变异系数越小，流动性越好
        amount_score = min(1.0, avg_amount / 1e8)  # 以1亿为满分
        stability_score = max(0, 1 - cv)

        return (amount_score * 0.6 + stability_score * 0.4)
    except:
        return 0.5


# ============================================================================
# 核心函数
# ============================================================================

def score_latest(panel: pd.DataFrame, feat_cols: list, seq_len: int, ckpt: dict, device) -> pd.DataFrame:
    """对最新交易日数据进行预测打分"""
    import torch

    model, flatten = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    last_date = panel["trade_date"].max()
    rows = []
    for code, g in panel.groupby("ts_code"):
        g = g.sort_values("trade_date")
        if len(g) < seq_len or g["trade_date"].iloc[-1] != last_date:
            continue
        window = g[feat_cols].astype(np.float32).values[-seq_len:]
        if np.isnan(window).any():
            continue
        mu = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-6
        window = (window - mu) / std
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        x_np = window.reshape(-1) if flatten else window
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)
        with torch.no_grad():
            score = float(model(x).cpu().item())
        rows.append({"trade_date": last_date, "ts_code": code, "score": score})

    return pd.DataFrame(rows)


def get_prices(panel: pd.DataFrame, codes: list) -> dict:
    """获取价格信息"""
    last_date = panel["trade_date"].max()
    prev_date = (pd.to_datetime(last_date, format="%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")

    prices = {}
    for code in codes:
        prev_row = panel[(panel["ts_code"] == code) & (panel["trade_date"].astype(str) == str(prev_date))]
        if not prev_row.empty:
            prev_close = float(prev_row["close"].iloc[0])
        else:
            curr_row = panel[(panel["ts_code"] == code) & (panel["trade_date"].astype(str) == str(last_date))]
            prev_close = float(curr_row["close"].iloc[0]) if not curr_row.empty else 0

        prices[code] = {
            "prev_close": prev_close,
            "buy_price": round(prev_close * 1.002, 2),
            "sell_price": round(prev_close * 0.998, 2),
        }
    return prices


def generate_trading_guide(
    scores: pd.DataFrame,
    panel: pd.DataFrame,
    current_holdings: list[HoldingStock],
    n_hold: int,
    k_trade: int,
    target_position: float,
    portfolio_value: float,
    allocation_strategy: str,
    scheme_name: str,
    output_dir: Path,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """生成完整的交易指南（包含备选和调整方案）"""

    last_date = scores["trade_date"].max()
    trading_date = (pd.to_datetime(last_date, format="%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    # 可买股票
    if "buyable" in scores.columns:
        buyable = scores[scores["buyable"].astype(bool)].copy()
    else:
        buyable = scores.copy()

    buyable = buyable.sort_values("score", ascending=False)

    # 计算权重
    weights = calculate_weights(buyable.set_index("ts_code")["score"], n_hold, allocation_strategy)

    # 目标持仓
    target_codes = buyable.head(n_hold)["ts_code"].tolist()
    available_value = portfolio_value * target_position

    # 当前持仓映射
    current_map = {h.ts_code: h for h in current_holdings}

    # 获取价格
    all_codes = list(set(target_codes + [h.ts_code for h in current_holdings]))
    prices = get_prices(panel, all_codes)

    # 构建目标持仓
    targets = []
    for rank, code in enumerate(target_codes, 1):
        score = float(buyable[buyable["ts_code"] == code]["score"].iloc[0])
        weight = float(weights.get(code, 0))
        target_value = available_value * weight
        price = prices.get(code, {}).get("prev_close", 0)
        target_shares = int(target_value / price // 100 * 100) if price > 0 else 0

        current = current_map.get(code)
        current_shares = current.shares if current else 0

        if current_shares == 0:
            action, action_shares = "buy", target_shares
        elif target_shares > current_shares:
            action, action_shares = "buy", target_shares - current_shares
        elif target_shares < current_shares:
            action, action_shares = "sell", current_shares - target_shares
        else:
            action, action_shares = "hold", 0

        # 计算流动性
        liq_score = calculate_liquidity_score(panel, code)

        # 风险提示
        risk = ""
        if price > 0 and target_value / price > 500000:  # 单股金额超过50万
            risk = "⚠️ 大单，注意分批"
        if liq_score < 0.3:
            risk = "⚠️ 流动性差"
        if price > 0 and target_shares > 0:
            participation = (target_shares * price) / (panel[panel["ts_code"] == code]["amount"].mean() * 10000 + 1)
            if participation > 0.1:
                risk = "⚠️ 参与率高，分2批"

        targets.append(TargetStock(
            ts_code=code,
            rank=rank,
            score=score,
            weight=weight,
            target_shares=target_shares,
            target_value=target_value,
            current_shares=current_shares,
            action=action,
            action_shares=action_shares,
            buy_price=prices.get(code, {}).get("buy_price", 0),
            sell_price=prices.get(code, {}).get("sell_price", 0),
            liquidity_score=liq_score,
            risk_flag=risk,
        ))

    # 备选股票（目标之外的高分股票）
    backup_codes = buyable.iloc[n_hold:n_hold * 3]["ts_code"].tolist()
    backups = []
    for rank, code in enumerate(backup_codes, n_hold + 1):
        score = float(buyable[buyable["ts_code"] == code]["score"].iloc[0])
        liq_score = calculate_liquidity_score(panel, code)
        backups.append({
            "rank": rank,
            "ts_code": code,
            "score": score,
            "buy_price": prices.get(code, {}).get("buy_price", 0),
            "liquidity": "优" if liq_score > 0.6 else ("中" if liq_score > 0.3 else "差"),
            "替代": f"当{targets[rank - n_hold - 1].ts_code if rank - n_hold - 1 < len(targets) else '目标股'}买不到时使用",
        })

    # 分离买卖
    to_buy = [t for t in targets if t.action == "buy"]
    to_sell = [t for t in targets if t.action == "sell"]
    to_hold = [t for t in targets if t.action == "hold"]

    # 按权重排序
    to_buy.sort(key=lambda x: -x.weight)
    to_sell.sort(key=lambda x: x.weight)

    # 实际交易（最多k_trade只）
    actual_buy = to_buy[:k_trade]
    actual_sell = to_sell[:k_trade]

    # 构建文本
    guide = f"""
================================================================================
                     同花顺手动下单交易指南
================================================================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据日期: {last_date}
交易日期: {trading_date}
方案: {scheme_name}

--------------------------------------------------------------------------------
                           【策略参数】
--------------------------------------------------------------------------------
  持仓目标: {n_hold} 只
  目标仓位: {target_position:.0%}
  账户总额: {portfolio_value:,.0f} 元
  可用资金: {available_value:,.0f} 元
  分配策略: {allocation_strategy}

================================================================================
                         【卖出清单 - 优先执行】
================================================================================
操作顺序: 先卖出，后买入

"""

    if not actual_sell:
        guide += "  ✓ 无需卖出\n\n"
    else:
        guide += f"  {'优先级':<4} {'股票代码':<12} {'当前手数':<8} {'建议卖出价':<10} {'风险提示':<15}\n"
        guide += "  " + "-" * 60 + "\n"
        for i, t in enumerate(actual_sell, 1):
            guide += f"  {i:<4} {t.ts_code:<12} {t.current_shares:<8} {t.sell_price:<10.2f} {t.risk_flag:<15}\n"

        guide += f"""
  ↓ 卖出后资金可用，约 {(sum(t.current_shares * t.sell_price for t in actual_sell)):,.0f} 元

"""

    guide += f"""================================================================================
                         【买入清单 - 后执行】
================================================================================
"""

    if not actual_buy:
        guide += "  ✓ 无需买入\n\n"
    else:
        guide += f"  {'优先级':<4} {'股票代码':<12} {'目标手数':<8} {'建议买入价':<10} {'预估金额':<12} {'风险提示':<15}\n"
        guide += "  " + "-" * 80 + "\n"
        for i, t in enumerate(actual_buy, 1):
            est_amount = t.action_shares * t.buy_price
            guide += f"  {i:<4} {t.ts_code:<12} {t.target_shares:<8} {t.buy_price:<10.2f} {est_amount:<12,.0f} {t.risk_flag:<15}\n"

        guide += f"""
  ↓ 需买入资金约 {sum(t.action_shares * t.buy_price for t in actual_buy):,.0f} 元

"""

    # ========== 核心改进：失败时的备选方案 ==========
    guide += f"""================================================================================
                    【失败时的备选方案 - 务必阅读】
================================================================================

"""

    # 买失败的备选
    if actual_buy:
        guide += """【A. 买入失败时的调整方案】
"""

        for i, t in enumerate(actual_buy, 1):
            # 找对应的备选
            backup = backups[i - 1] if i - 1 < len(backups) else None
            guide += f"""
  目标股票 {t.ts_code} 买不进去？

  原因排查：
  1. 是否涨停？ → 涨停则跳过，换备选
  2. 报价太低？ → 提高0.5%重新报价
  3. 流动性太差？ → 减少买入量或换备选

  备选方案{i}：{backup['ts_code'] if backup else '无'}
  备选价格：{backup['buy_price'] if backup else 0:.2f}
  备选流动性：{backup['liquidity'] if backup else '未知'}
  备选分数：{backup['score']:.4f} (第{backup['rank']}名)

  调整策略：
  ① 第1次尝试：撤单，价格+0.5%
  ② 第2次尝试：撤单，价格+1.0%，减半买入
  ③ 第3次尝试：放弃，换备选股票
  ④ 备选也失败：选择下一个备选或放弃

"""

    # 卖失败的备选
    if actual_sell:
        guide += """【B. 卖出失败时的调整方案】
"""

        for i, t in enumerate(actual_sell, 1):
            guide += f"""
  持仓股票 {t.ts_code} 卖不出去？

  原因排查：
  1. 是否跌停？ → 跌停则等尾盘
  2. 报价太高？ → 降低0.3%重新报价
  3. 流动性太差？ → 降低报价0.5%

  调整策略：
  ① 第1次尝试：撤单，价格-0.3%
  ② 第2次尝试：等14:00后，价格-0.5%
  ③ 第3次尝试：14:50尾盘，价格-1.0%
  ④ 仍未卖出：接受现实，次日继续卖出

  注意：如果T+1限制（今日买的），则必须等到次日
"""

    # 部分成交的调整
    guide += """
【C. 部分成交时的处理】

  成交率判断：
  ┌──────────────────────────────────────────────────────┐
  │ 成交率    │ 处理方式                                 │
  ├──────────────────────────────────────────────────────┤
  │ > 80%    │ 正常，剩余资金留现金                    │
  │ 50-80%   │ 剩余资金买备选股或留现金                │
  │ < 50%    │ 分析原因，严重时次日重新操作              │
  └──────────────────────────────────────────────────────┘

"""

    # 分批下单建议
    guide += """
【D. 大单分批下单建议】

  当单股金额 > 50万 或 参与率 > 10% 时，建议分2批：

  第一批（60%）：开盘后30分钟内下单
  第二批（40%）：10:00后根据情况决定

  批次1报价：参考价 × 1.002
  批次2报价：参考价 × 1.005（如第一批未完全成交）

"""

    # ========== 备选股票池 ==========
    guide += f"""
================================================================================
                         【备选股票池】
================================================================================
当目标股票无法交易时，按顺序选择以下备选：

  {'序号':<4} {'股票代码':<12} {'分数':<10} {'买入价':<10} {'流动性':<8} {'替代说明':<20}
  """ + "-" * 70 + "\n"

    for i, b in enumerate(backups[:10], 1):
        guide += f"  {i:<4} {b['ts_code']:<12} {b['score']:<10.4f} {b['buy_price']:<10.2f} {b['liquidity']:<8} {b['替代']:<20}\n"

    # ========== 同花顺操作格式 ==========
    guide += f"""
================================================================================
                         【同花顺操作格式】
================================================================================
【卖出操作】（先执行）
  证券代码            卖出价格        卖出数量
"""

    if not actual_sell:
        guide += "  (无需卖出)\n"
    else:
        for t in actual_sell:
            guide += f"  {t.ts_code:<16} {t.sell_price:<14.2f} {t.current_shares:>10}\n"

    guide += """
【买入操作】（后执行）
  证券代码            买入价格        买入数量
"""

    if not actual_buy:
        guide += "  (无需买入)\n"
    else:
        for t in actual_buy:
            guide += f"  {t.ts_code:<16} {t.buy_price:<14.2f} {t.target_shares:>10}\n"

    # ========== 持仓核对 ==========
    guide += f"""
================================================================================
                         【持仓核对表】
================================================================================
"""

    guide += f"  {'操作':<6} {'股票代码':<12} {'当前手数':<10} {'目标手数':<10} {'需买卖':<10}\n"
    guide += "  " + "-" * 60 + "\n"

    for t in targets[:n_hold]:
        if t.action == "buy":
            need = f"+{t.action_shares}"
        elif t.action == "sell":
            need = f"-{t.action_shares}"
        else:
            need = "0"
        guide += f"  {t.action:<6} {t.ts_code:<12} {t.current_shares:<10} {t.target_shares:<10} {need:<10}\n"

    guide += f"""
================================================================================
                         【快速检查清单】
================================================================================
□ 确认卖出股票后资金可用
□ 确认买入股票有足够资金
□ 检查是否有涨停/跌停股需要特殊处理
□ 准备好备选股票代码
□ 大单确认分批下单策略

================================================================================
"""

    # 创建日期目录
    date_dir = output_dir / "trading_guides" / last_date
    date_dir.mkdir(parents=True, exist_ok=True)

    # 保存文件到日期目录
    guide_path = date_dir / "trading_guide.txt"
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide)

    # 保存CSV
    orders = []

    for t in actual_sell:
        orders.append({
            "操作": "卖出",
            "股票代码": t.ts_code,
            "当前手数": t.current_shares,
            "建议卖出价": t.sell_price,
            "预估金额": t.current_shares * t.sell_price,
            "风险提示": t.risk_flag,
        })

    for t in actual_buy:
        orders.append({
            "操作": "买入",
            "股票代码": t.ts_code,
            "目标手数": t.target_shares,
            "建议买入价": t.buy_price,
            "预估金额": t.target_shares * t.buy_price,
            "风险提示": t.risk_flag,
        })

    orders_df = pd.DataFrame(orders)
    orders_df.to_csv(date_dir / "trading_orders.csv", index=False, encoding="utf-8-sig")

    # 保存备选股票
    if backups:
        backups_df = pd.DataFrame(backups)
        backups_df.to_csv(date_dir / "backup_stocks.csv", index=False, encoding="utf-8-sig")

    # 保存完整持仓
    full = []
    for t in targets:
        full.append({
            "排名": t.rank,
            "股票代码": t.ts_code,
            "分数": t.score,
            "权重": f"{t.weight:.2%}",
            "目标手数": t.target_shares,
            "当前手数": t.current_shares,
            "操作": t.action,
            "需买卖手数": t.action_shares,
            "建议买入价": t.buy_price,
            "建议卖出价": t.sell_price,
            "流动性": "优" if t.liquidity_score > 0.6 else ("中" if t.liquidity_score > 0.3 else "差"),
            "风险提示": t.risk_flag,
        })

    full_df = pd.DataFrame(full)
    full_df.to_csv(date_dir / "full_positions.csv", index=False, encoding="utf-8-sig")

    # 保存配置信息
    config_info = {
        "data_date": last_date,
        "trading_date": trading_date,
        "scheme": scheme_name,
        "portfolio_value": portfolio_value,
        "target_position": target_position,
        "n_hold": len(targets),
        "n_buy": len(actual_buy),
        "n_sell": len(actual_sell),
        "allocation_strategy": allocation_strategy,
    }
    with open(date_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)

    return guide, orders_df, date_dir


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="同花顺手动下单交易指南生成器")
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--holdings", default="", help="格式: 代码:手数,代码:手数")
    parser.add_argument("--portfolio-value", type=float, default=1000000)
    parser.add_argument("--scheme-name", default="方案A(纯DL)")
    parser.add_argument("--allocation", default="score_weighted",
                       choices=["equal", "score_weighted", "rank_weighted", "power"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  同花顺手动下单交易指南生成器 v2")
    print(f"{'='*60}")
    print(f"配置: {args.config}")
    print(f"方案: {args.scheme_name}")
    print()

    # 加载模型
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = output_dir / "model.pt"

    if not ckpt_path.exists():
        print(f"❌ 错误: 模型文件不存在: {ckpt_path}")
        print("请先运行训练脚本")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    print(f"✓ 模型加载成功")

    # 构建面板
    end = cfg["end_date"]
    end_dt = datetime.strptime(end, "%Y%m%d")
    start = (end_dt - timedelta(days=200)).strftime("%Y%m%d")

    panel = build_panel(
        cfg["data_dir"],
        start_date=start,
        end_date=end,
        use_metric=cfg["features"]["use_metric"],
        use_moneyflow=cfg["features"]["use_moneyflow"],
        use_market=cfg["features"].get("use_market", True),
        use_news=cfg["features"].get("use_news", False),
        universe=cfg["universe"],
    )
    panel = add_features(
        panel,
        cross_section_rank=cfg["features"]["cross_section_rank"],
        label_horizon=cfg.get("label_horizon", 1),
        fill_missing=cfg["features"].get("fill_missing", True),
    )

    feat_cols = ckpt.get("feat_cols") or feature_columns(panel)
    for c in feat_cols:
        if c not in panel.columns:
            panel[c] = 0.0

    print(f"✓ 数据面板构建成功: {len(panel)} 行")

    # 预测
    scores = score_latest(panel, feat_cols, ckpt["seq_len"], ckpt, device)
    scores = attach_buyable_flag(scores, panel, cfg["strategy"])

    if scores.empty:
        print("❌ 错误: 无法生成预测分数")
        sys.exit(1)

    print(f"✓ 预测完成: {len(scores)} 只股票")

    # 解析持仓
    holdings = parse_holdings(args.holdings)
    print(f"✓ 当前持仓: {len(holdings)} 只股票")

    # 策略参数
    n_hold = cfg["strategy"].get("n_hold", 30)
    k_trade = cfg["strategy"].get("k_trade", 1)
    target_position, _ = choose_target_position(
        scores.set_index("ts_code")["score"], n_hold, cfg["strategy"], None, 0.0
    )

    # 生成指南
    guide, orders_df, date_dir = generate_trading_guide(
        scores=scores,
        panel=panel,
        current_holdings=holdings,
        n_hold=n_hold,
        k_trade=k_trade,
        target_position=target_position,
        portfolio_value=args.portfolio_value,
        allocation_strategy=args.allocation,
        scheme_name=f"{args.scheme_name} ({args.allocation})",
        output_dir=output_dir,
    )

    print(guide)

    print(f"\n✅ 文件已保存到目录:")
    print(f"   {date_dir}/")
    print(f"   ├── trading_guide.txt      # 完整交易指南")
    print(f"   ├── trading_orders.csv    # 买卖订单")
    print(f"   ├── backup_stocks.csv     # 备选股票")
    print(f"   ├── full_positions.csv    # 完整持仓")
    print(f"   └── config.json           # 配置信息")


if __name__ == "__main__":
    main()
