#!/usr/bin/env bash
set -euo pipefail
LAYER="${1:-}"
if [[ -z "$LAYER" ]]; then
  echo "usage: scripts/sbatch/submit_exp1_layer.sh <LAYER_INDEX>" >&2
  exit 1
fi

CFG="${CFG:-logit_lens/configs/logit_lens.yaml}"
MODEL_DIR="${MODEL_DIR:-$PWD/model}"

# derive d_mlp
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
D_MLP="${D_MLP_OVERRIDE:-$D_MLP}"

# configurable partition and GPU flags
PARTITION="${PARTITION:-gpu}"         # override: PARTITION=a100
GPU_FLAGS="${GPU_FLAGS:---gres=gpu:1}" # override: GPU_FLAGS="--gres=gpu:tesla:1" or "--gpus=1" if your cluster supports it

echo "Submitting LAYER=$LAYER with array 0-$(($D_MLP-1)) on -p $PARTITION $GPU_FLAGS"
sbatch -p "$PARTITION" $GPU_FLAGS \
  --export=ALL,LAYER="$LAYER" \
  --array=0-$(($D_MLP-1)) \
  scripts/sbatch/exp1_neuron_array.sbatch
