#!/usr/bin/env bash
set -euo pipefail

CFG="${CFG:-logit_lens/configs/logit_lens.yaml}"
CATS="${CATS:-logit_lens/configs/categories.yaml}"
LAYERS="${LAYERS:-18 19 20 21 22 23}"
NUM_QUERIES="${NUM_QUERIES:-2000}"
FIRE_POLICY="${FIRE_POLICY:-topk}"   # topk | zscore
TOP_P="${TOP_P:-0.05}"
ZTHR="${ZTHR:-2.5}"
ACT_EPS="${ACT_EPS:-1e-6}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/exp2_full_${STAMP}.log"
echo "[INFO] Logging to $LOG"
echo "[INFO] Layers: $LAYERS ; num_queries=$NUM_QUERIES ; fire_policy=$FIRE_POLICY top_p=$TOP_P zthr=$ZTHR" | tee -a "$LOG"

for L in $LAYERS; do
  OUT="logit_lens/results/exp2/L${L}/neuron_specialization.csv"
  if [[ -s "$OUT" ]]; then
    echo "[L$L] exists, skipping (found $OUT)" | tee -a "$LOG"
    continue
  fi
  echo "[L$L] running..." | tee -a "$LOG"
  python -m logit_lens.experiments.exp2_specialists \
    --config "$CFG" --categories "$CATS" \
    --layer "$L" --num_queries "$NUM_QUERIES" \
    --fire_policy "$FIRE_POLICY" --top_p "$TOP_P" --zthr "$ZTHR" --act_eps "$ACT_EPS" \
    >> "$LOG" 2>&1
  echo "[L$L] done." | tee -a "$LOG"
done

echo "[ALL DONE] Exp2" | tee -a "$LOG"
