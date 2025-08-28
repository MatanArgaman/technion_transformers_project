#!/usr/bin/env bash
# #SBATCH -A your_account
#SBATCH -J exp1_agg_L${LAYER}
#SBATCH -o logs/exp1_agg_L${LAYER}.out
#SBATCH -e logs/exp1_agg_L${LAYER}.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
source ~/.bashrc || true
conda activate genir-ll

: "${LAYER?Set LAYER to match the array layer}}"
CFG="${CFG:-logit_lens/configs/logit_lens.yaml}"
OUT_DIR="${OUT_DIR:-$PWD/logit_lens/results}"

echo "== EXP1 AGG LAYER =="
echo "Layer: $LAYER"
echo "CFG: $CFG"
echo "OUT_DIR: $OUT_DIR"

# placeholder aggregation
python - <<'PY'
import os, glob, json
out_dir=os.environ.get("OUT_DIR","./logit_lens/results")
layer=os.environ["LAYER"]
layer_dir=os.path.join(out_dir,"exp1",f"L{layer}")
print("Would aggregate files under:", layer_dir)
print("Sample files:", sorted(glob.glob(os.path.join(layer_dir,"*.csv")))[:5])
PY
