#!/usr/bin/env bash
set -euo pipefail
for L in 18 19 20 21 22 23; do
  scripts/sbatch/submit_exp1_layer.sh "$L"
done
