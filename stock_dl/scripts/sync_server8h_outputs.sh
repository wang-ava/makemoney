#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sync_server8h_outputs.sh <ssh-host> [a|b|both]

Examples:
  bash scripts/sync_server8h_outputs.sh pb23061103@gpu.example.edu b
  REMOTE_ROOT=/home/scc/pb23061103/projects/makemoney/stock_dl \
    bash scripts/sync_server8h_outputs.sh my-gpu both

Environment:
  REMOTE_ROOT  Remote stock_dl directory.
               Default: /home/scc/pb23061103/projects/makemoney/stock_dl
  WITH_PANEL   Set to 1 to also download panel.parquet. Default: 0.
  CLEAN_OLD    Set to 0 to keep old generated output folders. Default: 1.
EOF
}

REMOTE="${1:-}"
SCHEME="${2:-b}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/scc/pb23061103/projects/makemoney/stock_dl}"
WITH_PANEL="${WITH_PANEL:-0}"
CLEAN_OLD="${CLEAN_OLD:-1}"

if [[ -z "$REMOTE" || "$REMOTE" == "-h" || "$REMOTE" == "--help" ]]; then
  usage
  exit 0
fi

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$LOCAL_ROOT/.." && pwd)"

if [[ "$CLEAN_OLD" == "1" ]]; then
  rm -rf \
    "$PROJECT_ROOT/best-trails" \
    "$LOCAL_ROOT/outputs" \
    "$LOCAL_ROOT/outputs_quick" \
    "$LOCAL_ROOT/outputs_scheme_a" \
    "$LOCAL_ROOT/outputs_scheme_b" \
    "$LOCAL_ROOT/outputs_tuning" \
    "$LOCAL_ROOT/backtest_inputs" \
    "$LOCAL_ROOT/backtest_results"
fi

sync_one() {
  local scheme="$1"
  local out_dir="outputs_server8h_scheme_${scheme}_short"
  local dest="$LOCAL_ROOT/$out_dir"

  echo "== Sync $out_dir =="
  rm -rf "$dest"
  mkdir -p "$dest"

  local includes=(
    "*/"
    "model.pt"
    "train_meta.json"
    "ic_summary.json"
    "backtest_metrics.json"
    "best_strategy.json"
    "strategy_tuning.csv"
    "model_audit_summary.json"
    "model_audit_report.md"
    "model_audit_*.csv"
  )

  if [[ "$scheme" == "b" ]]; then
    includes+=(
      "lgbm_model.txt"
      "lgbm_meta.json"
      "lgbm_status.json"
      "blend_meta.json"
    )
  fi

  if [[ "$WITH_PANEL" == "1" ]]; then
    includes+=("panel.parquet" "panel_schema.json")
  fi

  local rsync_args=(-avP --prune-empty-dirs)
  for pattern in "${includes[@]}"; do
    rsync_args+=(--include="$pattern")
  done
  rsync_args+=(--exclude="*")

  rsync "${rsync_args[@]}" "$REMOTE:$REMOTE_ROOT/$out_dir/" "$dest/"
}

case "$SCHEME" in
  a)
    sync_one a
    ;;
  b)
    sync_one b
    ;;
  both)
    sync_one a
    sync_one b
    ;;
  *)
    echo "Unknown scheme: $SCHEME" >&2
    usage
    exit 2
    ;;
esac

echo
echo "Done. Local outputs are under:"
echo "  $LOCAL_ROOT/outputs_server8h_scheme_a_short"
echo "  $LOCAL_ROOT/outputs_server8h_scheme_b_short"
