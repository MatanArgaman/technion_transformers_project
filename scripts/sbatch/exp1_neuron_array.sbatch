#!/usr/bin/env bash
#SBATCH -o logs/exp1_L%a.out
#SBATCH -e logs/exp1_L%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-7

set -euo pipefail

# Always run from the submit directory (repo root)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- env ---
source ~/.bashrc || true
conda activate genir-ll

# --- inputs ---
: "${LAYER?Set LAYER=18..23}"
CFG="${CFG:-logit_lens/configs/logit_lens.yaml}"
export MODEL_DIR="${MODEL_DIR:-$PWD/model}"
export DATA_DIR="${DATA_DIR:-$PWD/data/NQ10k}"
export OUT_DIR="${OUT_DIR:-$PWD/logit_lens/results}"
export NEURON_ID="${SLURM_ARRAY_TASK_ID}"

# Ensure output directory exists (layer subdir used later)
mkdir -p "$OUT_DIR" "$OUT_DIR/exp1" "$OUT_DIR/exp1/L${LAYER}"

# Caches (optional)
export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

echo "== EXP1 ARRAY JOB =="
echo "CWD: $(pwd)"
echo "JobID: ${SLURM_JOB_ID:-NA}  Task: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Layer: $LAYER  Neuron: $NEURON_ID"
echo "CFG: $CFG"
echo "MODEL_DIR: $MODEL_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "OUT_DIR: $OUT_DIR"
nvidia-smi || true

# -------- placeholder sanity until Step 3 code exists --------
python - <<'PY'
import os, sys, yaml
need=["MODEL_DIR","DATA_DIR","OUT_DIR","LAYER","NEURON_ID"]
for k in need:
    print(f"{k}={os.environ.get(k)}")
ok = all(os.path.exists(os.environ.get(k,"")) for k in ["MODEL_DIR","DATA_DIR","OUT_DIR"])
print("paths_ok:", ok)
sys.exit(0 if ok else 2)
PY

# -------- real entrypoint (Step 3 will enable) --------
# python -m logit_lens.experiments.exp1_individual_neuron \
#   --config $CFG --layer $LAYER --neuron $NEURON_ID \
#   --model_dir $MODEL_DIR --data_dir $DATA_DIR --out_dir $OUT_DIR
