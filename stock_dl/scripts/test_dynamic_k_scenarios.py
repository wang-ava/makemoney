#!/usr/bin/env python3
"""
动态 k 场景验证：模拟 6 种持仓状态，验证 _choose_dynamic_k 决策

_choose_dynamic_k 算法（持仓质量驱动）：
- 烂股 = 持仓中分数 < 候选池中位数的股
- k = max(base_k, 烂股数 + bad_buffer)
- 候选质量好（gap 大）→ k + dynamic_k_step
- 持仓都好（无烂股）且差距小 → k - dynamic_k_step
- 上限：max(烂股数, 好候选数)（cfg 中 dynamic_k_max 可覆盖）

本测试从 src.backtest.engine 导入 _choose_dynamic_k（与 14_generate_trading_guide.py
逻辑保持一致——本仓库以 14 脚本为正确版本，engine 已同步）。
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import _choose_dynamic_k


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
        "dynamic_k_use_rank_scores": False,
        "dynamic_k_bad_min": 1,
        "dynamic_k_bad_buffer": 0,
        # dynamic_k_max 不设置 → 默认 = max(烂股数, 好候选数)
        "dynamic_k_step": 2,
        "score_gap_high": 0.10,
        "score_gap_low": 0.02,
    }

    k = _choose_dynamic_k(day_scores, buy_scores, holdings, base_k, strategy_cfg)

    # 计算关键中间量（用于打印）
    held_set = [c for c in holdings if c in day_scores.index]
    candidates = buy_scores.drop(index=list(holdings.keys()), errors="ignore")
    held_scores_series = day_scores.loc[held_set]
    held_mean = float(held_scores_series.mean())
    held_min = float(held_scores_series.min())
    cand_median = float(candidates.median())
    cand_max = float(candidates.max())
    bad = int((held_scores_series < cand_median).sum())
    good = int((candidates > held_mean).sum())
    gap = cand_max - held_min

    return k, {
        "scenario": name,
        "n_held": n_held,
        "n_candidates": n_candidates,
        "held_scores": held_scores,
        "cand_scores": cand_scores,
        "base_k": base_k,
        "decision_k": k,
        "bad": bad,
        "good": good,
        "gap": gap,
        "held_mean": held_mean,
        "cand_median": cand_median,
    }


def main():
    print("=" * 90)
    print("动态 k 场景验证（持仓质量驱动）")
    print("=" * 90)
    print("k 决策流程：起步 base_k → 烂股>=1 则 k=max(k, 烂股+buffer) → 好候选>=1 且 gap>high 则 k+=step → ")
    print("            烂股=0 且 gap<low 则 k=max(1, k-step) → k=min(k, max(烂股, 好候选)) → max(1, k)")
    print()

    # === 场景 A: 烂持仓（5 只都在候选中位数以下）===
    k_a, info_a = setup_scenario(
        "A. 烂持仓（5 只全 < 候选中位）",
        n_held=5, held_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        n_candidates=10, cand_scores=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    )
    print(f"\n{info_a['scenario']}")
    print(f"  持仓 {info_a['n_held']} 只: {info_a['held_scores']}")
    print(f"  候选 {info_a['n_candidates']} 只: {info_a['cand_scores']}")
    print(f"  候选中位={info_a['cand_median']:.2f}, 持仓均分={info_a['held_mean']:.2f}, gap={info_a['gap']:.2f}")
    print(f"  烂股={info_a['bad']}, 好候选={info_a['good']}")
    print(f"  决策: 起步 k=6 → 烂股>=1 触发 k=max(6,5)=6 → 好候选>=1 且 gap>0.10 触发 k=6+2=8")
    print(f"        → 上限 max(烂股, 好候选)=max(5,10)=10 → min(8,10)=8 → max(1,8)=8")
    print(f"  实际 k = {info_a['decision_k']}")

    # === 场景 B: 好持仓（5 只全在候选中位数以上）===
    k_b, info_b = setup_scenario(
        "B. 好持仓（5 只全 > 候选中位）",
        n_held=5, held_scores=[1.5, 1.6, 1.7, 1.8, 1.9],
        n_candidates=10, cand_scores=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    print(f"\n{info_b['scenario']}")
    print(f"  持仓 {info_b['n_held']} 只: {info_b['held_scores']}")
    print(f"  候选 {info_b['n_candidates']} 只: {info_b['cand_scores']}")
    print(f"  候选中位={info_b['cand_median']:.2f}, 持仓均分={info_b['held_mean']:.2f}, gap={info_b['gap']:.2f}")
    print(f"  烂股={info_b['bad']}, 好候选={info_b['good']}")
    print(f"  决策: 起步 k=6 → 烂股=0 跳过强制 → 好候选=0 跳过 +step → 烂股=0 且 gap<0.02 触发 k=max(1,6-1)=5")
    print(f"        → 上限 max(0,0)=0 → min(5,0)=0 → max(1,0)=1")
    print(f"  实际 k = {info_b['decision_k']}（候选都比持仓差，强制保底 1 换手）")

    # === 场景 C: 混合持仓（3 只烂 + 2 只好）===
    k_c, info_c = setup_scenario(
        "C. 混合持仓（3 只烂 + 2 只好）",
        n_held=5, held_scores=[0.5, 0.7, 0.8, 1.5, 1.7],
        n_candidates=10, cand_scores=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    )
    print(f"\n{info_c['scenario']}")
    print(f"  持仓 {info_c['n_held']} 只: {info_c['held_scores']}")
    print(f"  候选 {info_c['n_candidates']} 只: {info_c['cand_scores']}")
    print(f"  候选中位={info_c['cand_median']:.2f}, 持仓均分={info_c['held_mean']:.2f}, gap={info_c['gap']:.2f}")
    print(f"  烂股={info_c['bad']}, 好候选={info_c['good']}")
    print(f"  决策: 起步 k=6 → 烂股>=1 触发 k=max(6,3)=6 → 好候选>=1 且 gap>0.10 触发 k=6+2=8")
    print(f"        → 上限 max(3,4)=4 → min(8,4)=4 → max(1,4)=4")
    print(f"  实际 k = {info_c['decision_k']}")

    # === 场景 D: 极端烂持仓（10 只全烂）===
    k_d, info_d = setup_scenario(
        "D. 极端烂持仓（10 只全 < 候选中位）",
        n_held=10, held_scores=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8],
        n_candidates=10, cand_scores=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    )
    print(f"\n{info_d['scenario']}")
    print(f"  持仓 {info_d['n_held']} 只: {info_d['held_scores']}")
    print(f"  候选 {info_d['n_candidates']} 只: {info_d['cand_scores']}")
    print(f"  候选中位={info_d['cand_median']:.2f}, 持仓均分={info_d['held_mean']:.2f}, gap={info_d['gap']:.2f}")
    print(f"  烂股={info_d['bad']}, 好候选={info_d['good']}")
    print(f"  决策: 起步 k=6 → 烂股>=1 触发 k=max(6,10)=10 → 好候选>=1 且 gap>0.10 触发 k=10+2=12")
    print(f"        → 上限 max(10,9)=10 → min(12,10)=10 → max(1,10)=10")
    print(f"  实际 k = {info_d['decision_k']}")

    # === 场景 E: base_k=2 烂持仓（验证 base_k 是下界）===
    k_e, info_e = setup_scenario(
        "E. base_k=2 烂持仓（base_k 是下界）",
        n_held=5, held_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        n_candidates=10, cand_scores=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        base_k=2,
    )
    print(f"\n{info_e['scenario']}")
    print(f"  base_k=2, 烂股={info_e['bad']}, 好候选={info_e['good']}, gap={info_e['gap']:.2f}")
    print(f"  决策: 起步 k=2 → 烂股>=1 触发 k=max(2,5)=5 → 好候选>=1 且 gap>0.10 触发 k=5+2=7")
    print(f"        → 上限 max(5,10)=10 → min(7,10)=7 → max(1,7)=7")
    print(f"  实际 k = {info_e['decision_k']}")

    # === 场景 F: 好持仓 + 候选也好（候选远高于持仓，触发 +2）===
    k_f, info_f = setup_scenario(
        "F. 持仓烂 + 候选极好（gap 极大）",
        n_held=3, held_scores=[0.5, 0.6, 0.7],
        n_candidates=10, cand_scores=[1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5],
    )
    print(f"\n{info_f['scenario']}")
    print(f"  持仓 {info_f['n_held']} 只: {info_f['held_scores']}")
    print(f"  候选 {info_f['n_candidates']} 只: {info_f['cand_scores']}")
    print(f"  候选中位={info_f['cand_median']:.2f}, 持仓均分={info_f['held_mean']:.2f}, gap={info_f['gap']:.2f}")
    print(f"  烂股={info_f['bad']}, 好候选={info_f['good']}")
    print(f"  决策: 起步 k=6 → 烂股>=1 触发 k=max(6,3)=6 → 好候选>=1 且 gap>0.10 触发 k=6+2=8")
    print(f"        → 上限 max(3,10)=10 → min(8,10)=8 → max(1,8)=8")
    print(f"  实际 k = {info_f['decision_k']}")

    # === 场景 G: rank 分数模式（A/B 分数尺度不同也能共用阈值）===
    rank_scores = pd.Series({
        "HELD_STRONG": 10.0,
        "HELD_WEAK": 9.9,
        "CAND_CLOSE": 9.8,
        "CAND_WEAK": 1.0,
    })
    k_g = _choose_dynamic_k(
        rank_scores,
        rank_scores,
        {"HELD_STRONG": 100, "HELD_WEAK": 100},
        2,
        {
            "dynamic_k": True,
            "dynamic_k_use_rank_scores": True,
            "score_gap_trigger": True,
            "score_gap_trigger_threshold": 0.30,
            "dynamic_k_bad_min": 1,
            "dynamic_k_bad_buffer": 0,
            "dynamic_k_step": 1,
            "score_gap_high": 0.10,
            "score_gap_low": 0.02,
            "dynamic_k_max": 6,
        },
    )
    print("\nG. rank 分数模式 + 分差不足")
    print("  原始分数虽然接近，但动态 K 使用横截面百分位；候选没有明显优于持仓时不换手")
    print(f"  实际 k = {k_g}（期望 0）")

    # === 总结表 ===
    print("\n" + "=" * 90)
    print("总结：动态 k 决策表")
    print("=" * 90)
    print(f"{'场景':<38} {'base_k':>7} {'烂股':>5} {'好候选':>7} {'gap':>6} {'决策k':>7}")
    print("-" * 90)
    summary = [
        ("A 烂持仓（5 全烂，+step，max=10）", 6, info_a['bad'], info_a['good'], info_a['gap'], info_a['decision_k']),
        ("B 好持仓（无好候选，max=0 保底 1）", 6, info_b['bad'], info_b['good'], info_b['gap'], info_b['decision_k']),
        ("C 混合持仓（3 烂+2 好，max=4）", 6, info_c['bad'], info_c['good'], info_c['gap'], info_c['decision_k']),
        ("D 极端烂（10 全烂，max=10）", 6, info_d['bad'], info_d['good'], info_d['gap'], info_d['decision_k']),
        ("E base_k=2 烂持仓（max=10）", 2, info_e['bad'], info_e['good'], info_e['gap'], info_e['decision_k']),
        ("F 烂持仓+候选极好（+step，max=10）", 6, info_f['bad'], info_f['good'], info_f['gap'], info_f['decision_k']),
        ("G rank模式分差不足（trigger=0）", 2, 0, 0, 0.0, k_g),
    ]
    for name, bk, bad, good, gap, k in summary:
        print(f"  {name:<36} {bk:>7} {bad:>5} {good:>7} {gap:>6.2f} {k:>7}")


if __name__ == "__main__":
    main()
