# Experiment 1 (Individual Neuron Logit Lens) — Artifacts

This folder keeps small, report-friendly outputs:
- `layer_L{L}_top_neurons.csv`: top 25 neurons per layer (ranked by top1_mode_frac, then active_rate)
- `layer_L{L}_summary_agg.csv`: full per-neuron summary table for the layer
- `run_manifest_*.json`: configuration used for this run (lens, layers, act_eps, etc.)

Heavy per-neuron CSVs (`*_topk.csv`, `*_stats.csv`, `*_perquery.csv`, `*_summary.csv`) remain in `logit_lens/results/exp1/L*/` on the server and are intentionally not tracked in Git.
