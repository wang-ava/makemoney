#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_A="${CONFIG_A:-configs/server_8h_scheme_a_short.yaml}"
CONFIG_B="${CONFIG_B:-configs/server_8h_scheme_b_short.yaml}"
OUT_A="${OUT_A:-outputs_server8h_scheme_a_short}"
OUT_B="${OUT_B:-outputs_server8h_scheme_b_short}"

TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-470}"
A_TRAIN_MAX_MINUTES="${A_TRAIN_MAX_MINUTES:-150}"
B_TRAIN_MAX_MINUTES="${B_TRAIN_MAX_MINUTES:-150}"
B_LGBM_MAX_MINUTES="${B_LGBM_MAX_MINUTES:-65}"
AUDIT_TOP_K="${AUDIT_TOP_K:-30}"
DEPLOY_REQUIRE_GREEN="${DEPLOY_REQUIRE_GREEN:-0}"

export PYTHONPATH="$ROOT"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"

START_TS="$(date +%s)"

elapsed_minutes() {
  local now
  now="$(date +%s)"
  echo $(( (now - START_TS) / 60 ))
}

remaining_minutes() {
  local elapsed
  elapsed="$(elapsed_minutes)"
  echo $(( TOTAL_BUDGET_MINUTES - elapsed ))
}

check_budget() {
  local elapsed
  elapsed="$(elapsed_minutes)"
  if [ "$elapsed" -ge "$TOTAL_BUDGET_MINUTES" ]; then
    echo "Time budget exhausted: elapsed=${elapsed}m, budget=${TOTAL_BUDGET_MINUTES}m."
    exit 124
  fi
}

run_stage() {
  local name="$1"
  shift
  echo
  echo "[$(date '+%F %T')] == ${name} =="
  echo "elapsed=$(elapsed_minutes)m remaining=$(remaining_minutes)m"
  "$@"
  echo "[$(date '+%F %T')] done: ${name}; elapsed=$(elapsed_minutes)m remaining=$(remaining_minutes)m"
  check_budget
}

copy_panel_to_b() {
  mkdir -p "$OUT_B"
  cp "$OUT_A/panel.parquet" "$OUT_B/panel.parquet"
  if [ -f "$OUT_A/panel_schema.json" ]; then
    cp "$OUT_A/panel_schema.json" "$OUT_B/panel_schema.json"
  fi
}

summarize_audits() {
  python3 - "$OUT_A" "$OUT_B" "$AUDIT_TOP_K" "$DEPLOY_REQUIRE_GREEN" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


out_a, out_b, audit_top_k, require_green = sys.argv[1:5]
audit_top_k = int(audit_top_k)
require_green = str(require_green).lower() in {"1", "true", "yes", "y"}


def load_summary(name: str, out_dir: str) -> dict:
    path = Path(out_dir) / "model_audit_summary.json"
    if not path.exists():
        return {"scheme": name, "output_dir": out_dir, "status": "missing", "approved": False}
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["scheme"] = name
    return summary


def assess(summary: dict) -> dict:
    top_k = int(summary.get("audit_top_k", audit_top_k))
    tradable_ic = float(summary.get("tradable_ic_last20", 0.0))
    tradable_excess = float(summary.get("tradable_topk_excess_last20", summary.get("tradable_top70_excess_last20", 0.0)))
    t1_excess = float(summary.get("realizable_t1_topk_excess_last20", summary.get("realizable_t1_top70_excess_last20", 0.0)))
    equity20 = float(summary.get("equity_return_last20", 0.0))
    status = str(summary.get("status", "red"))
    approved = (
        status != "red"
        and tradable_ic >= 0.03
        and tradable_excess > 0.0
        and t1_excess > 0.0
        and equity20 > -0.05
    )
    return {
        "scheme": summary.get("scheme", "?"),
        "top_k": top_k,
        "status": status,
        "tradable_ic_last20": tradable_ic,
        "tradable_topk_excess_last20": tradable_excess,
        "realizable_t1_topk_excess_last20": t1_excess,
        "equity_return_last20": equity20,
        "approved": approved,
    }


rows = [assess(load_summary("A", out_a)), assess(load_summary("B", out_b))]
print("Deployment gate:")
for row in rows:
    print(
        "  {scheme}: status={status} top{top_k} "
        "ic20={tradable_ic_last20:.4f} "
        "excess20={tradable_topk_excess_last20:.4%} "
        "t1_excess20={realizable_t1_topk_excess_last20:.4%} "
        "equity20={equity_return_last20:.2%} "
        "approved={approved}".format(**row)
    )

approved = [row for row in rows if row["approved"]]
if approved:
    best = max(approved, key=lambda row: row["tradable_topk_excess_last20"])
    print(f"Deployment gate result: APPROVED scheme={best['scheme']} for guarded trading.")
else:
    print("Deployment gate result: NOT APPROVED. Keep outputs for review, but do not auto-buy.")
    if require_green:
        raise SystemExit(2)
PY
}

cat <<EOF
Server 8h A/B short retrain
Target: next_open_to_next_close
Meaning: signal before/near open -> buy next open -> score next close profit.
Budget: total=${TOTAL_BUDGET_MINUTES}m, A deep=${A_TRAIN_MAX_MINUTES}m, B deep=${B_TRAIN_MAX_MINUTES}m, B LGBM=${B_LGBM_MAX_MINUTES}m.
Audit: top_k=${AUDIT_TOP_K}, require_green=${DEPLOY_REQUIRE_GREEN}
Outputs: ${OUT_A}, ${OUT_B}
EOF

run_stage "A build shared short-term panel" python3 scripts/01_build_panel.py --config "$CONFIG_A"
run_stage "A train deep model" env TRAIN_MAX_MINUTES="$A_TRAIN_MAX_MINUTES" python3 scripts/03_train.py --config "$CONFIG_A"
run_stage "A eval IC" python3 scripts/04_eval_ic.py --config "$CONFIG_A"
run_stage "A backtest" python3 scripts/05_backtest.py --config "$CONFIG_A"
run_stage "A tradable audit" python3 scripts/16_model_audit.py --config "$CONFIG_A" --top-k "$AUDIT_TOP_K"

run_stage "copy shared panel to B" copy_panel_to_b
run_stage "B train deep model" env TRAIN_MAX_MINUTES="$B_TRAIN_MAX_MINUTES" python3 scripts/03_train.py --config "$CONFIG_B"
run_stage "B train LGBM channel" env LGBM_MAX_MINUTES="$B_LGBM_MAX_MINUTES" python3 scripts/10_train_lgbm.py --config "$CONFIG_B"
run_stage "B blend deep/LGBM" python3 scripts/11_blend_scores.py --config "$CONFIG_B"
run_stage "B eval IC" python3 scripts/04_eval_ic.py --config "$CONFIG_B"
run_stage "B backtest" python3 scripts/05_backtest.py --config "$CONFIG_B"
run_stage "B tradable audit" python3 scripts/16_model_audit.py --config "$CONFIG_B" --top-k "$AUDIT_TOP_K"
run_stage "audit deployment gate" summarize_audits

cat <<EOF

Finished A/B short retrain.
A audit: ${OUT_A}/model_audit_report.md
B audit: ${OUT_B}/model_audit_report.md
Elapsed: $(elapsed_minutes)m
EOF
