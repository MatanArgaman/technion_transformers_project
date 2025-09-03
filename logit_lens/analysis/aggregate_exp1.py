from __future__ import annotations
import argparse, csv, glob, os
from typing import List, Dict, Any
from transformers import AutoTokenizer

def read_summary_csv(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    return r[0] if r else {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--results_dir", type=str, default="logit_lens/results/exp1")
    ap.add_argument("--model_dir", type=str, default="./model")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    layer_dir = os.path.join(args.results_dir, f"L{args.layer}")
    files = sorted(glob.glob(os.path.join(layer_dir, "neuron_*_summary.csv")))
    if not files:
        print(f"No summary CSVs found in {layer_dir}")
        return

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)

    rows: List[Dict[str, Any]] = []
    for p in files:
        row = read_summary_csv(p)
        # decode token id to string (if present)
        s = row.get("top1_mode_doc_token_id", "")
        try:
            sid = int(s) if str(s).strip() != "" else None
        except Exception:
            sid = None
        row["top1_mode_doc_token_str"] = tok.convert_ids_to_tokens([sid])[0] if sid is not None else ""
        # coerce numeric fields for sorting
        for k in ("active_rate","top1_mode_frac","a_j_mean_active","a_j_std_active"):
            try:
                row[k] = float(row.get(k, 0.0))
            except Exception:
                row[k] = 0.0
        rows.append(row)

    # sort: primary by top1_mode_frac desc, then active_rate desc
    rows.sort(key=lambda r: (r.get("top1_mode_frac",0.0), r.get("active_rate",0.0)), reverse=True)

    out_path = args.out or os.path.join(layer_dir, f"layer_L{args.layer}_summary_agg.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # also emit a short top list
    top_path = os.path.join(layer_dir, f"layer_L{args.layer}_top_neurons.csv")
    keep = ["neuron_id","active_rate","top1_mode_frac","top1_mode_doc_token_id","top1_mode_doc_token_str","a_j_mean_active","a_j_std_active","lens","n_queries"]
    with open(top_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keep)
        w.writeheader()
        for r in rows[:25]:
            w.writerow({k:r.get(k,"") for k in keep})

    print(f"Wrote {out_path}")
    print(f"Wrote {top_path}")

if __name__ == "__main__":
    main()
