# Experiment 3 (Doc Promotions) — Curated Artifacts

- `exp3_summary_all_layers.csv`: one row per (layer, neuron); quick KPIs (active_rate, top_doc_frac, doc_entropy).
- `exp3_top_neurons.csv`: top 10 neurons per layer by top_doc_frac (for figures and case studies).
- `exp3_top_neurons_topdocs.csv`: for those top neurons, their top-10 promoted documents (freq, mean/max score, title if available).

Raw results live under `logit_lens/results/exp3/L*/` (not tracked by Git).
