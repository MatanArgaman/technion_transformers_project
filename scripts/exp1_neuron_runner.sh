#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ---------- defaults ----------
CFG="${CFG:-logit_lens/configs/logit_lens.yaml}"
MODEL_DIR="${MODEL_DIR:-$PWD/model}"
DATA_DIR="${DATA_DIR:-$PWD/data/NQ10k}"
OUT_DIR="${OUT_DIR:-$PWD/logit_lens/results}"
PY_ENTRY="${PY_ENTRY:-placeholder}"   # set to "real" later
export CFG MODEL_DIR DATA_DIR OUT_DIR PY_ENTRY

# caches
export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TOKENIZERS_PARALLELISM=false

# ---------- arg parse ----------
LAYER=""; NEURON=""; NEURON_FIRST=""; NEURON_LAST=""
PY_EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --) shift; PY_EXTRA+=("$@"); break ;;
    --layer)        LAYER="$2"; shift 2 ;;
    --neuron)       NEURON="$2"; shift 2 ;;
    --neuron_first) NEURON_FIRST="$2"; shift 2 ;;
    --neuron_last)  NEURON_LAST="$2"; shift 2 ;;
    --config)       CFG="$2"; export CFG; shift 2 ;;
    --model_dir)    MODEL_DIR="$2"; export MODEL_DIR; shift 2 ;;
    --data_dir)     DATA_DIR="$2"; export DATA_DIR; shift 2 ;;
    --out_dir)      OUT_DIR="$2"; export OUT_DIR; shift 2 ;;
    --py_entry)     PY_ENTRY="$2"; export PY_ENTRY; shift 2 ;;
    *) echo "Unknown arg: $1 (use -- to pass through to Python)"; exit 2 ;;
  esac
done

[[ -n "$LAYER" ]] || { echo "ERROR: --layer is required" >&2; exit 2; }

mkdir -p "$OUT_DIR/exp1/L${LAYER}" logs

# ---------- pick neuron IDs ----------
NEURON_IDS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  NEURON_IDS=("${SLURM_ARRAY_TASK_ID}")
elif [[ -n "$NEURON" ]]; then
  NEURON_IDS=("$NEURON")
elif [[ -n "$NEURON_FIRST" && -n "$NEURON_LAST" ]]; then
  for i in $(seq "$NEURON_FIRST" "$NEURON_LAST"); do NEURON_IDS+=("$i"); done
else
  for i in $(seq 0 31); do NEURON_IDS+=("$i"); done
fi

echo "== RUN EXP1 =="
echo "CWD: $(pwd)"
echo "Layer: $LAYER"
echo "Neuron IDs: ${NEURON_IDS[*]}"
echo "CFG: $CFG"
echo "MODEL_DIR: $MODEL_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "OUT_DIR: $OUT_DIR"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

for p in "$MODEL_DIR" "$DATA_DIR" "$OUT_DIR"; do
  [[ -e "$p" ]] || { echo "Missing path: $p" >&2; exit 3; }
done

rc_all=0
for nid in "${NEURON_IDS[@]}"; do
  echo "--- L${LAYER} N${nid} ---"
  if [[ "$PY_ENTRY" == "real" ]]; then
    python -m logit_lens.experiments.exp1_individual_neuron \
      --config "$CFG" --layer "$LAYER" --neuron "$nid" \
      --model_dir "$MODEL_DIR" --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
      "${PY_EXTRA[@]}" || rc_all=$?
  else
    python - <<'PY' || rc_all=$?
import os, sys
need=["MODEL_DIR","DATA_DIR","OUT_DIR"]
vals={k:os.environ.get(k) for k in need}
print("env paths:", vals)
ok = all(v and os.path.exists(v) for v in vals.values())
print("placeholder OK:", ok)
sys.exit(0 if ok else 4)
PY
  fi
done

exit "$rc_all"
