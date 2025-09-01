#!/usr/bin/env bash
set -euo pipefail
LAYER="${1:-}"
if [[ -z "$LAYER" ]]; then
  echo "usage: scripts/sbatch/resubmit_missing_exp1.sh <LAYER_INDEX>" >&2
  exit 1
fi
OUT_DIR="${OUT_DIR:-$PWD/logit_lens/results}"
MODEL_DIR="${MODEL_DIR:-$PWD/model}"
CFG="${CFG:-logit_lens/configs/logit_lens.yaml}"

# compute D_MLP as in the submit wrapper
D_MLP=$(python - <<'PY'
import json, os
p=os.environ.get("MODEL_DIR","./model")+"/config.json"
try:
  j=json.load(open(p))
  for k in ("d_ff","ffn_dim","intermediate_size","feed_forward_proj_size","mlp_dim"):
    if k in j:
      print(int(j[k])); break
  else: print(1024)
except Exception: print(1024)
PY
)

layer_dir="$OUT_DIR/exp1/L${LAYER}"
mkdir -p "$layer_dir"
missing=()
for i in $(seq 0 $((D_MLP-1))); do
  fn=$(printf "%s/neuron_%06d_topk.csv" "$layer_dir" "$i")
  [[ -s "$fn" ]] || missing+=("$i")
done

if [[ ${#missing[@]} -eq 0 ]]; then
  echo "No missing neuron outputs for layer $LAYER"
  exit 0
fi

comma=$(IFS=,; echo "${missing[*]}")
echo "Resubmitting missing indices for layer $LAYER:"
echo "$comma"

sbatch --export=ALL,LAYER="$LAYER" \
  --array="$comma" \
  scripts/sbatch/exp1_neuron_array.sbatch
