#!/usr/bin/env bash
set -euo pipefail

# ensure Python can import the repo package even under nohup
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CANDIDATES="${CANDIDATES:-artifacts/exp2/exp2_candidates.csv}"
LAYERS="${LAYERS:-18 19 20 21 22 23}"
NUM_QUERIES="${NUM_QUERIES:-4000}"
TOPK_DOCS="${TOPK_DOCS:-50}"
ACT_EPS="${ACT_EPS:-1e-6}"
LENS="${LENS:-simple}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/exp3_full_${STAMP}.log"
echo "[INFO] Logging to $LOG"
echo "[INFO] layers=$LAYERS nq=$NUM_QUERIES topk=$TOPK_DOCS lens=$LENS act_eps=$ACT_EPS" | tee -a "$LOG"

mkdir -p tmp

# split candidates by layer (so we can resume per-layer)
awk -F, 'NR==1{hdr=$0;next}{out="tmp/exp3_L"$1".csv"; if(!(out in seen)){print hdr > out; seen[out]=1} print > out}' "$CANDIDATES"

for L in $LAYERS; do
  LCSV="tmp/exp3_L${L}.csv"
  if [[ ! -s "$LCSV" ]]; then
    echo "[L$L] no candidates, skip" | tee -a "$LOG"
    continue
  fi
  echo "[L$L] running..." | tee -a "$LOG"
  python -u -m logit_lens.experiments.exp3_doc_promotions \
    --candidates "$LCSV" --layers "$L" \
    --num_queries "$NUM_QUERIES" --topk_docs "$TOPK_DOCS" \
    --act_eps "$ACT_EPS" --lens "$LENS" \
    >> "$LOG" 2>&1
  echo "[L$L] done." | tee -a "$LOG"
done

echo "[ALL DONE] Exp3" | tee -a "$LOG"
