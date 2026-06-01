#!/usr/bin/env python3
"""
动态 k 场景验证：模拟 4 种持仓状态，验证 _choose_dynamic_k 决策

场景：
A. 烂持仓：5 只持仓都在候选池中位数以下（应该多换手）
B. 好持仓：5 只持仓都在候选池中位数以上（应该少换手）
C. 混合持仓：3 只烂 + 2 只好（应该换掉 3 只）
D. 极端：10 只持仓全都很烂（应该 max_k 限制）
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 直接 import 14 脚本里的函数（避免 src.engine 依赖）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "guide", ROOT / "scripts" / "14_generate_trading_guide.py"
)
# 但 14 脚本太长会 import torch 等，单独把函数抽出来测试更稳
# 这里直接复制逻辑到本地测试

def choose_dynamic_k(day_scores, buy_scores, holdings, base_k, strategy_cfg):
    """与 14 / engine 保持一致的动态 k 实现"""
    if not strategy_cfg.get("dynamic_k", False) or not holdings:
        return int(base_k)
    held = [c for c in holdings if c in day_scores.index]
    candidates = buy_scores.drop(index=[c for c in holdings if c in buy_scores.index], errors="ignore")
    if not held or candidates.empty:
        return int(base_k)

    held_scores = day_scores.loc[held]
    held_mean = float(held_scores.mean())
    held_min = float(held_scores.min())
    candidate_max = float(candidates.max())
    candidate_median = float(candidates.median())

    bad_stocks_count = int((held_scores < candidate_median).sum())
    bad_min = int(strategy_cfg.get("dynamic_k_bad_min", 1))
    bad_buffer = int(strategy_cfg.get("dynamic_k_bad_buffer", 0))

    k = int(base_k)

    if bad_stocks_count >= bad_min:
        k = max(k, bad_stocks_count + bad_buffer)

    high = float(strategy_cfg.get("score_gap_high", 0.10))
    low = float(strategy_cfg.get("score_gap_low", 0.02))
    gap = candidate_max - held_min
    good_candidates_count = int((candidates > held_mean).sum())
    if good_candidates_count >= bad_min and gap > high:
        k = k + int(strategy_cfg.get("dynamic_k_step", 2))

    if bad_stocks_count == 0 and gap < low:
        k = max(1, k - int(strategy_cfg.get("dynamic_k_step", 1)))

    max_k_default = max(bad_stocks_count, good_candidates_count)
    max_k_cfg = strategy_cfg.get("dynamic_k_max", None)
    max_k = int(max_k_cfg) if max_k_cfg is not None else max_k_default
    k = min(k, max_k)

    return max(1, k)


def setup_scenario(name, n_held, held_scores, n_candidates, cand_scores, base_k=6):
    """构造测试场景"""
    held_codes = [f"HELD_{i:02d}" for i in range(n_held)]
    cand_codes = [f"CAND_{i:02d}" for i in range(n_candidates)]

    all_codes = held_codes + cand_codes
    all_scores = list(held_scores) + list(cand_scores)
    day_scores = pd.Series(all_scores, index=all_codes)
    buy_scores = day_scores.copy()

    holdings = {c: 100 for c in held_codes}

    strategy_cfg = {
        "dynamic_k": True,
        "dynamic_k_bad_min": 1,
        "dynamic_k_bad_buffer": 0,
        # dynamic_k_max 不设置 → 默认 = len(held)（跟随持仓规模）
        "dynamic_k_step": 2,
        "score_gap_high": 0.10,
        "score_gap_low": 0.02,
    }

    k = choose_dynamic_k(day_scores, buy_scores, holdings, base_k, strategy_cfg)
    return k, {
        "scenario": name,
        "n_held": n_held,
        "n_candidates": n_candidates,
        "held_scores": held_scores,
        "cand_scores": cand_scores,
        "base_k": base_k,
        "decision_k": k,
    }


def main():
    print("=" * 90)
    print("动态 k 场景验证（持仓质量驱动）")
    print("=" * 90)

    # === 场景 A: 烂持仓（5 只都在候选中位数以下）===
    # 候选分 [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4] -> 中位 0.95
    # 持仓分 [0.1, 0.2, 0.3, 0.4, 0.5] -> 全部 < 0.95
    k_a, info_a = setup_scenario(
        "A. 烂持仓（5 只全 < 候选中位）",
        n_held=5, held_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        n_candidates=10, cand_scores=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    )
    print(f"\n{info_a['scenario']}")
    print(f"  持仓分数: {info_a['held_scores']}")
    print(f"  候选分数: {info_a['cand_scores']}")
    print(f"  候选中位: 0.95")
    print(f"  烂股数 (持仓分<0.95): 5")
    print(f"  期望 k = max(base_k=6, 5) = 6（取 base_k）")
    print(f"  实际 k = {info_a['decision_k']}  {'✅' if info_a['decision_k'] >= 5 else '❌'}")

    # === 场景 B: 好持仓（5 只全在候选中位数以上）===
    # 持仓分 [1.5, 1.6, 1.7, 1.8, 1.9] 候选中位 0.95，gap=1.4-1.5=-0.1
    k_b, info_b = setup_scenario(
        "B. 好持仓（5 只全 > 候选中位）",
        n_held=5, held_scores=[1.5, 1.6, 1.7, 1.8, 1.9],
        n_candidates=10, cand_scores=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    print(f"\n{info_b['scenario']}")
    print(f"  持仓分数: {info_b['held_scores']}")
    print(f"  候选分数: {info_b['cand_scores']}")
    print(f"  烂股数: 0")
    print(f"  gap = 候选max(1.0) - 持仓min(1.5) = -0.5（无新候选）")
    print(f"  实际 k = {info_b['decision_k']}（应=base_k=6 或 减少）")

    # === 场景 C: 混合持仓（3 只烂 + 2 只好）===
    # 候选中位 0.95，持仓 [0.5, 0.7, 0.8, 1.5, 1.7]
    # 烂股 (持仓分<0.95) = 3
    k_c, info_c = setup_scenario(
        "C. 混合持仓（3 只烂 + 2 只好）",
        n_held=5, held_scores=[0.5, 0.7, 0.8, 1.5, 1.7],
        n_candidates=10, cand_scores=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    )
    print(f"\n{info_c['scenario']}")
    print(f"  持仓分数: {info_c['held_scores']}")
    print(f"  候选中位: 0.95")
    print(f"  烂股数 (持仓分<0.95): 3")
    print(f"  期望 k = max(base_k=6, 3) = 6（取 base_k）")
    print(f"  实际 k = {info_c['decision_k']}  {'✅' if info_c['decision_k'] >= 3 else '❌'}")

    # === 场景 D: 极端烂持仓（10 只全烂）===
    # 持仓 10 只全在 0.1-0.5，候选中位 0.95
    k_d, info_d = setup_scenario(
        "D. 极端烂持仓（10 只全 < 候选中位）",
        n_held=10, held_scores=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8],
        n_candidates=10, cand_scores=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    )
    print(f"\n{info_d['scenario']}")
    print(f"  持仓分数: {info_d['held_scores']}")
    print(f"  烂股数: 9（0.8 < 0.95）")
    print(f"  期望 k = max(6, 9) = 9, 然后 max_k=10//2=5 限制 = 5")
    print(f"  实际 k = {info_d['decision_k']}  {'✅' if info_d['decision_k'] == 5 else '❌'}")

    # === 场景 E: base_k=2 烂持仓（验证 base_k 是下界）===
    k_e, info_e = setup_scenario(
        "E. base_k=2 烂持仓（base_k 是下界）",
        n_held=5, held_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        n_candidates=10, cand_scores=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        base_k=2,
    )
    print(f"\n{info_e['scenario']}")
    print(f"  base_k=2, 烂股数=5, 期望 k = max(2, 5) = 5")
    print(f"  实际 k = {info_e['decision_k']}  {'✅' if info_e['decision_k'] == 5 else '❌'}")

    # === 场景 F: 好持仓 + 候选也好（候选远高于持仓，触发 +2）===
    # 持仓 [0.5, 0.6, 0.7]，候选 [1.5, 1.6, ... 2.5]
    # 烂股 = 3 (候选中位约 2.0, 持仓全 < 2.0)
    k_f, info_f = setup_scenario(
        "F. 持仓烂 + 候选极好（gap 极大）",
        n_held=3, held_scores=[0.5, 0.6, 0.7],
        n_candidates=10, cand_scores=[1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5],
    )
    print(f"\n{info_f['scenario']}")
    print(f"  持仓分数: {info_f['held_scores']}")
    print(f"  候选中位: 1.95, 烂股数: 3")
    print(f"  gap = 2.5-0.5 = 2.0 (远 > 0.10)")
    print(f"  期望 k = max(6, 3) + 2 = 8")
    print(f"  实际 k = {info_f['decision_k']}  {'✅' if info_f['decision_k'] == 8 else '❌'}")

    print("\n" + "=" * 90)
    print("总结：动态 k 决策表（max_k = max(烂股数, 好候选数)）")
    print("=" * 90)
    print(f"{'场景':<35} {'base_k':>7} {'烂股':>5} {'gap':>7} {'决策k':>7} {'预期':>7}")
    print("-" * 90)
    summary = [
        # (场景, base_k, 烂股, gap, 实际, 预期)
        # 实际 k 来自上面的输出
        ("A 烂持仓（5 全烂，候选好，+2）", 6, 5, 0.9, info_a['decision_k'], 8),
        ("B 好持仓（5 全好，无好候选，-2）", 6, 0, -0.5, info_b['decision_k'], 1),
        ("C 混合持仓（3 烂 + 2 好，max_k=4）", 6, 3, 0.9, info_c['decision_k'], 4),
        ("D 极端烂（10 全烂，max_k=10）", 6, 10, 1.3, info_d['decision_k'], 10),
        ("E base_k=2 烂持仓（max_k=10）", 2, 5, 0.9, info_e['decision_k'], 7),
        ("F 烂持仓 + 候选极好（+2，max_k=10）", 6, 3, 2.0, info_f['decision_k'], 8),
    ]
    for name, bk, bad, gap, k, exp in summary:
        ok = "✅" if k == exp else "❌"
        print(f"  {name:<33} {bk:>7} {bad:>5} {str(gap):>7} {k:>7} {exp:>7} {ok}")


if __name__ == "__main__":
    main()
