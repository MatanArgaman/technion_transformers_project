#!/usr/bin/env bash
set -euo pipefail
LAYER="${1:-}"
if [[ -z "$LAYER" ]]; then
  echo "usage: scripts/sbatch/submit_exp1_layer_chunked.sh <LAYER_INDEX>" >&2
  exit 1
fi

CFG="${CFG:-logit_lens/configs/logit_lens.yaml}"
MODEL_DIR="${MODEL_DIR:-$PWD/model}"

# ---- derive d_mlp ----
D_MLP=$(python - <<'PY'
import json, os
p=os.environ.get("MODEL_DIR","./model")+"/config.json"
try:
  j=json.load(open(p))
  for k in ("d_ff","ffn_dim","intermediate_size","feed_forward_size","feed_forward_proj_size","mlp_dim"):
    if k in j:
      print(int(j[k])); break
  else: print(1024)
except Exception: print(1024)
PY
)
D_MLP="${D_MLP_OVERRIDE:-$D_MLP}"

# ---- chunk sizing ----
MAXARR=$({ scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2}'; } || true)
[[ -z "$MAXARR" ]] && MAXARR=1000
ARRAY_CHUNK="${ARRAY_CHUNK:-$MAXARR}"
(( ARRAY_CHUNK > MAXARR )) && ARRAY_CHUNK="$MAXARR"

# concurrency throttle
CONCURRENCY="${CONCURRENCY:-}"
[[ -n "$CONCURRENCY" ]] && CONC="%${CONCURRENCY}" || CONC=""

# partition & GPU flags (optional)
PARTITION="${PARTITION:-}"                 # e.g., PARTITION=gpu-a100 (leave empty to use site default)
GPU_FLAGS="${GPU_FLAGS:---gres=gpu:1}"     # e.g., "--gres=gpu:a100:1" or "--gpus=1" if supported

echo "D_MLP=$D_MLP  ARRAY_CHUNK=$ARRAY_CHUNK  CONCURRENCY=${CONC:-none}  PARTITION=${PARTITION:-default}  GPU_FLAGS=$GPU_FLAGS"

base_args=( --job-name "exp1_L${LAYER}" )
[[ -n "$PARTITION" ]] && base_args+=( -p "$PARTITION" )
IFS=' ' read -r -a gpu_args <<< "$GPU_FLAGS"

start=0
jobids=()
while (( start < D_MLP )); do
  end=$(( start + ARRAY_CHUNK - 1 ))
  (( end >= D_MLP )) && end=$(( D_MLP - 1 ))
  echo "Submitting LAYER=$LAYER chunk ${start}-${end}${CONC}"
  jid=$(sbatch "${base_args[@]}" "${gpu_args[@]}" \
      --export=ALL,LAYER="$LAYER" \
      --array=${start}-${end}${CONC} \
      scripts/sbatch/exp1_neuron_array.sbatch | awk '{print $4}')
  jobids+=("$jid")
  start=$(( end + 1 ))
done

# aggregator dependent on all chunks
if [[ "${SUBMIT_AGGREGATOR:-1}" == "1" ]]; then
  dep="afterok:$(IFS=:; echo "${jobids[*]}")"
  echo "Submitting aggregator for LAYER=$LAYER with dependency: $dep"
  sbatch --job-name "exp1_agg_L${LAYER}" \
         ${PARTITION:+-p "$PARTITION"} \
         --dependency="$dep" \
         --export=ALL,LAYER="$LAYER" \
         scripts/sbatch/exp1_aggregate_layer.sbatch
fi
