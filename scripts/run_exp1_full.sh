#!/usr/bin/env bash
set -euo pipefail

# -------------------- Config (override with env) --------------------
CFG="${CFG:-logit_lens/configs/logit_lens.yaml}"
MODEL_DIR="${MODEL_DIR:-$PWD/model}"
DATA_DIR="${DATA_DIR:-$PWD/data/NQ10k}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/logit_lens/results/exp1}"

# Lens & activation
LENS="${LENS:-simple}"            # simple | lnaware
ACT_EPS="${ACT_EPS:-1e-5}"
NQ_VAL_N="${NQ_VAL_N:-32}"        # per-neuron query count for perquery CSV

# Concurrency (number of Python processes in parallel)
CONCURRENCY="${CONCURRENCY:-4}"

# Which layers? Auto-detected if LAYERS is empty
LAYERS="${LAYERS:-}"

# Export for child processes
export CFG MODEL_DIR DATA_DIR RESULTS_DIR LENS ACT_EPS NQ_VAL_N CONCURRENCY

# -------------------- Derived / setup --------------------
mkdir -p "$RESULTS_DIR" logs logit_lens/results/shared

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/exp1_full_${STAMP}.log"
MANIFEST="${RESULTS_DIR}/run_manifest_${STAMP}.json"

echo "[INFO] Logging to $LOG"
echo "[INFO] Results under $RESULTS_DIR"

# Record a manifest up front
python - <<PY | tee -a "$LOG" >/dev/null
import json, os, socket
m = {
  "started_at": "${STAMP}",
  "host": socket.gethostname(),
  "cfg": os.environ.get("CFG"),
  "model_dir": os.environ.get("MODEL_DIR"),
  "data_dir": os.environ.get("DATA_DIR"),
  "results_dir": os.environ.get("RESULTS_DIR"),
  "lens": os.environ.get("LENS"),
  "act_eps": os.environ.get("ACT_EPS"),
  "nq_val_n": os.environ.get("NQ_VAL_N"),
  "concurrency": os.environ.get("CONCURRENCY"),
  "layers_requested": os.environ.get("LAYERS",""),
}
os.makedirs(os.path.dirname("${MANIFEST}"), exist_ok=True)
json.dump(m, open("${MANIFEST}", "w"), indent=2)
print("[MANIFEST]", "${MANIFEST}")
PY

# Make sure model & data exist
[[ -e "$MODEL_DIR" ]] || { echo "[ERR] Missing MODEL_DIR=$MODEL_DIR" | tee -a "$LOG"; exit 2; }
[[ -e "$DATA_DIR"  ]] || { echo "[ERR] Missing DATA_DIR=$DATA_DIR"  | tee -a "$LOG"; exit 2; }

# -------------------- Ensure doc-id vocabulary exists --------------------
DOCJSON="logit_lens/results/shared/docid_vocab.json"
if [[ ! -s "$DOCJSON" ]]; then
  echo "[INFO] Building docid vocab JSON at $DOCJSON" | tee -a "$LOG"
  python - <<'PY' | tee -a "$LOG"
import os, re, json
from transformers import AutoTokenizer
MODEL_DIR=os.environ.get("MODEL_DIR","./model")
OUT="logit_lens/results/shared/docid_vocab.json"
tok=AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
v=tok.get_vocab()
pat=re.compile(r"^@DOC_ID_[\-0-9]+@$")
doc_ids=sorted({int(i) for s,i in v.items() if pat.match(s)})
print("[DOCS]", len(doc_ids))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"doc_token_ids": doc_ids}, open(OUT,"w"))
PY
fi

# -------------------- Auto-detect layers & d_ff --------------------
if [[ -z "$LAYERS" ]]; then
  LAYERS=$(python - <<'PY'
import os
from logit_lens.utils.config import load
from logit_lens.utils.model_io import load_model
cfg = load(os.environ.get("CFG","logit_lens/configs/logit_lens.yaml"))
mb = load_model(cfg, model_dir=os.environ.get("MODEL_DIR","./model"))
m = mb.model
layers = []
# seq2seq decoder?
dec = getattr(getattr(m,"model",m), "decoder", None)
if dec is not None:
    for att in ("block","layers"):
        if hasattr(dec, att):
            layers = list(range(len(getattr(dec, att))))
            break
# decoder-only?
if not layers and hasattr(getattr(m,"model",m), "layers"):
    layers = list(range(len(getattr(m, "model").layers)))
if not layers:
    layers = list(range(24))
print(*layers)
PY
)
  echo "[INFO] Auto-detected layers: $LAYERS" | tee -a "$LOG"
fi
export LAYERS

DFF=$(python - <<'PY'
import os
from logit_lens.utils.config import load
from logit_lens.utils.model_io import load_model, get_decoder_ff_out_weight
cfg = load(os.environ.get("CFG","logit_lens/configs/logit_lens.yaml"))
mb = load_model(cfg, model_dir=os.environ.get("MODEL_DIR","./model"))
# first layer from env LAYERS
L = int(os.environ["LAYERS"].split()[0])
W = get_decoder_ff_out_weight(mb.model, L)
print(W.shape[1])
PY
)
echo "[INFO] Detected d_ff=${DFF}" | tee -a "$LOG"

# -------------------- Helper: throttle background jobs --------------------
wait_for_slot () {
  while (( $(jobs -pr | wc -l) >= CONCURRENCY )); do sleep 1; done
}

# -------------------- Main loop: layers -> neurons --------------------
for L in $LAYERS; do
  LDIR="${RESULTS_DIR}/L${L}"
  mkdir -p "$LDIR"
  echo "[LAYER $L] starting..." | tee -a "$LOG"

  for ((N=0; N< DFF; N++)); do
    SUMCSV="${LDIR}/neuron_$(printf '%06d' "$N")_summary.csv"
    if [[ -s "$SUMCSV" ]]; then
      echo "[LAYER $L] skip neuron $N (summary exists)" | tee -a "$LOG"
      continue
    fi
    wait_for_slot
    (
      echo "[LAYER $L] neuron $N" | tee -a "$LOG"
      bash scripts/exp1_neuron_runner.sh \
        --layer "$L" --neuron "$N" --py_entry real -- \
        --use_nq_val "$NQ_VAL_N" --act_eps "$ACT_EPS" --lens "$LENS" \
        >> "$LOG" 2>&1
    ) &
  done
  wait
  echo "[LAYER $L] all neurons done, aggregating..." | tee -a "$LOG"
  python -m logit_lens.analysis.aggregate_exp1 \
    --layer "$L" --model_dir "$MODEL_DIR" --results_dir "$RESULTS_DIR" \
    >> "$LOG" 2>&1 || true
  echo "[LAYER $L] done." | tee -a "$LOG"
done

echo "[ALL DONE] See $RESULTS_DIR and $LOG"
