#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.utils.wandb_utils import finish_wandb, init_wandb, wandb_log_artifact, wandb_log_images


def save_training_curve(out: Path, fig_dir: Path) -> None:
    meta_fp = out / "train_meta.json"
    if not meta_fp.exists():
        return
    meta = json.loads(meta_fp.read_text(encoding="utf-8"))
    hist = pd.DataFrame(meta.get("history", []))
    if hist.empty:
        return
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(hist["epoch"], hist["train_loss"], label="train loss", color="#2b6cb0")
    ax1.plot(hist["epoch"], hist["val_loss"], label="val loss", color="#dd6b20")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    if "val_ic" in hist:
        ax2.plot(hist["epoch"], hist["val_ic"], label="val IC", color="#2f855a")
        ax2.set_ylabel("IC")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def save_ic_plot(out: Path, fig_dir: Path) -> None:
    fp = out / "daily_ic.csv"
    if not fp.exists():
        return
    ic = pd.read_csv(fp)
    if ic.empty:
        return
    ic["trade_date"] = pd.to_datetime(ic["trade_date"].astype(str))
    ic["cum_ic"] = ic["ic"].cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(ic["trade_date"], ic["ic"], width=1.0, color="#4a5568")
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_ylabel("daily IC")
    axes[0].grid(alpha=0.2)
    axes[1].plot(ic["trade_date"], ic["cum_ic"], color="#2b6cb0")
    axes[1].set_ylabel("cumulative IC")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "ic_timeseries.png", dpi=180)
    plt.close(fig)


def save_equity_plot(out: Path, fig_dir: Path) -> None:
    fp = out / "equity_curve.csv"
    if not fp.exists():
        return
    eq = pd.read_csv(fp)
    if eq.empty:
        return
    eq["trade_date"] = pd.to_datetime(eq["trade_date"].astype(str))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col, label, color in [
        ("equity", "strategy", "#c53030"),
        ("bench_hs300", "HS300", "#2b6cb0"),
        ("bench_sh", "SSE", "#2f855a"),
        ("bench_cyb", "ChiNext", "#805ad5"),
    ]:
        if col in eq:
            ax.plot(eq["trade_date"], eq[col] / eq[col].iloc[0], label=label, color=color)
    ax.set_ylabel("normalized equity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "equity_vs_benchmark.png", dpi=180)
    plt.close(fig)


def save_quantile_plot(out: Path, fig_dir: Path) -> None:
    fp = out / "val_predictions.csv"
    if not fp.exists():
        return
    pred = pd.read_csv(fp)
    if pred.empty:
        return
    pred = pred.dropna(subset=["score", "label"]).copy()
    pred["q"] = pred.groupby("trade_date")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    q = pred.groupby("q")["label"].agg(["mean", "count"]).reset_index()
    if q.empty:
        return
    q["bucket"] = q["q"].astype(int) + 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(q["bucket"], q["mean"] * 10000, color="#2b6cb0")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("score quantile, low to high")
    ax.set_ylabel("next-day return (bp)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "score_quantiles.png", dpi=180)
    plt.close(fig)


def save_baseline_plot(out: Path, fig_dir: Path) -> None:
    fp = out / "baseline_comparison.csv"
    if not fp.exists():
        return
    comp = pd.read_csv(fp)
    if comp.empty:
        return
    comp = comp.sort_values("total_return", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(comp["model"], comp["total_return"] * 100, color="#2b6cb0")
    axes[0].axvline(0, color="#222222", linewidth=0.8)
    axes[0].set_xlabel("total return (%)")
    axes[0].grid(axis="x", alpha=0.25)
    comp_ic = comp.sort_values("ic_mean", ascending=True)
    axes[1].barh(comp_ic["model"], comp_ic["ic_mean"], color="#2f855a")
    axes[1].axvline(0, color="#222222", linewidth=0.8)
    axes[1].set_xlabel("IC mean")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "baseline_comparison.png", dpi=180)
    plt.close(fig)


def save_feature_ic_plot(out: Path, fig_dir: Path) -> None:
    fp = out / "feature_ic.csv"
    if not fp.exists():
        return
    fi = pd.read_csv(fp)
    if fi.empty:
        return
    top = fi.sort_values("abs_ic_mean", ascending=False).head(20).sort_values("ic_mean")
    colors = ["#c53030" if x > 0 else "#2b6cb0" for x in top["ic_mean"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["ic_mean"], color=colors)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("single feature IC mean")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_ic_top20.png", dpi=180)
    plt.close(fig)


def save_lgbm_importance_plot(out: Path, fig_dir: Path) -> None:
    fp = out / "lgbm_feature_importance.csv"
    if not fp.exists():
        return
    imp = pd.read_csv(fp)
    if imp.empty or "gain" not in imp.columns:
        return
    top = imp.sort_values("gain", ascending=False).head(20).sort_values("gain")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["gain"], color="#805ad5")
    ax.set_xlabel("LightGBM feature importance (gain)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "lgbm_feature_importance.png", dpi=180)
    plt.close(fig)


def save_strategy_tuning_plot(out: Path, fig_dir: Path) -> None:
    fp = out / "strategy_tuning.csv"
    if not fp.exists():
        return
    tune = pd.read_csv(fp)
    if tune.empty:
        return
    pivot = tune.pivot(index="n_hold", columns="k_trade", values="total_return")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("k_trade")
    ax.set_ylabel("n_hold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val*100:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="total return (%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "strategy_tuning_heatmap.png", dpi=180)
    plt.close(fig)


def write_summary(out: Path, fig_dir: Path) -> None:
    metrics = {}
    ic = {}
    if (out / "backtest_metrics.json").exists():
        metrics = json.loads((out / "backtest_metrics.json").read_text(encoding="utf-8"))
    if (out / "ic_summary.json").exists():
        ic = json.loads((out / "ic_summary.json").read_text(encoding="utf-8"))

    lines = [
        "# 实验结果摘要",
        "",
        "## 核心指标",
        "",
        f"- IC mean: {ic.get('ic_mean', 0):.6f}",
        f"- ICIR: {ic.get('icir', 0):.6f}",
        f"- 策略总收益: {metrics.get('total_return', 0):.2%}",
        f"- 策略年化收益: {metrics.get('annual_return', 0):.2%}",
        f"- 夏普比率: {metrics.get('sharpe', 0):.3f}",
        f"- 最大回撤: {metrics.get('max_drawdown', 0):.2%}",
    ]
    benches = metrics.get("benchmarks", {})
    if benches:
        lines.extend(["", "## 市场基准总收益", ""])
        name_map = {"bench_sh": "上证指数", "bench_hs300": "沪深300", "bench_cyb": "创业板指数"}
        for key, vals in benches.items():
            lines.append(f"- {name_map.get(key, key)}: {vals.get('total_return', 0):.2%}")
    best_fp = out / "best_strategy.json"
    if best_fp.exists():
        best = json.loads(best_fp.read_text(encoding="utf-8"))
        lines.extend(["", "## 策略调优", ""])
        lines.append(f"- Best n_hold: {best.get('n_hold')}")
        lines.append(f"- Best k_trade: {best.get('k_trade')}")
    baseline_fp = out / "baseline_comparison.csv"
    if baseline_fp.exists():
        comp = pd.read_csv(baseline_fp)
        if not comp.empty:
            best = comp.sort_values("total_return", ascending=False).iloc[0]
            lines.extend(["", "## 基线对比", ""])
            lines.append(f"- 最优收益模型: {best['model']} ({best.get('total_return', 0):.2%})")
            if "final_model" in set(comp["model"]):
                row = comp[comp["model"] == "final_model"].iloc[0]
                lines.append(f"- 最终模型 IC mean: {row.get('ic_mean', 0):.6f}, 总收益: {row.get('total_return', 0):.2%}")
            if "lgbm_lambdarank" in set(comp["model"]):
                row = comp[comp["model"] == "lgbm_lambdarank"].iloc[0]
                lines.append(f"- LightGBM LambdaRank IC mean: {row.get('ic_mean', 0):.6f}, 总收益: {row.get('total_return', 0):.2%}")
    diag_fp = out / "prediction_diagnostics.csv"
    if diag_fp.exists():
        diag = pd.read_csv(diag_fp, index_col=0)["value"].to_dict()
        lines.extend(["", "## 预测诊断", ""])
        lines.append(f"- Top decile next return: {diag.get('top_decile_next_return', 0):.4%}")
        lines.append(f"- Bottom decile next return: {diag.get('bottom_decile_next_return', 0):.4%}")
        lines.append(f"- Top minus bottom: {diag.get('top_minus_bottom', 0):.4%}")
    lines.extend([
        "",
        "## 图表文件",
        "",
        f"- `{fig_dir / 'training_curve.png'}`",
        f"- `{fig_dir / 'ic_timeseries.png'}`",
        f"- `{fig_dir / 'equity_vs_benchmark.png'}`",
        f"- `{fig_dir / 'score_quantiles.png'}`",
        f"- `{fig_dir / 'baseline_comparison.png'}`",
        f"- `{fig_dir / 'feature_ic_top20.png'}`",
        f"- `{fig_dir / 'lgbm_feature_importance.png'}`",
        f"- `{fig_dir / 'strategy_tuning_heatmap.png'}`",
    ])
    (out / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_training_curve(out, fig_dir)
    save_ic_plot(out, fig_dir)
    save_equity_plot(out, fig_dir)
    save_quantile_plot(out, fig_dir)
    save_baseline_plot(out, fig_dir)
    save_feature_ic_plot(out, fig_dir)
    save_lgbm_importance_plot(out, fig_dir)
    save_strategy_tuning_plot(out, fig_dir)
    write_summary(out, fig_dir)
    wandb_run = init_wandb(cfg, job_type="visualize", extra_config={"script": "07_visualize.py"})
    wandb_log_images(wandb_run, fig_dir)
    if cfg.get("wandb", {}).get("log_artifacts", True):
        wandb_log_artifact(wandb_run, out / "experiment_summary.md", name="stock-dl-experiment-summary", artifact_type="report")
        wandb_log_artifact(wandb_run, fig_dir, name="stock-dl-figures", artifact_type="figures")
    finish_wandb(wandb_run)
    print(f"Saved figures and summary under {fig_dir}")


if __name__ == "__main__":
    main()
