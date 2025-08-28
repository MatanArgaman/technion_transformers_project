from __future__ import annotations
import argparse, csv, json, math, os
from typing import Dict, List, Tuple
import numpy as np
import torch

from logit_lens.utils.config import load as load_cfg
from logit_lens.utils.model_io import load_model
from logit_lens.utils.hooks import capture_ff_activations_for_query
from logit_lens.utils.nq10k import load_val_queries
from logit_lens.utils.categories import load_categories

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0: return 0.0
    return float(-(p * np.log(p)).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="logit_lens/configs/logit_lens.yaml")
    ap.add_argument("--categories", type=str, default="logit_lens/configs/categories.yaml")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--num_queries", type=int, default=2000)
    ap.add_argument("--fire_policy", type=str, default="topk", choices=["topk","zscore"])
    ap.add_argument("--top_p", type=float, default=0.05, help="top p fraction by |a| fired if topk")
    ap.add_argument("--zthr", type=float, default=2.5, help="|z| threshold if zscore")
    ap.add_argument("--act_eps", type=float, default=1e-6, help="clip tiny activations")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    mb = load_model(cfg)  # uses cfg['compute']['device'] and dtype
    device = mb.device

    # Load categories & queries
    matcher = load_categories(args.categories)
    rows = load_val_queries(cfg["paths"]["data_dir"], mb.tokenizer, doc_token_ids=None, max_examples=args.num_queries)
    queries = [(r["qid"], r["query"], matcher.label(r["query"])) for r in rows]
    if not queries:
        raise RuntimeError("No validation queries loaded")

    # per-layer setup
    # We only need d_ff; capture function returns per-neuron activations
    # We'll infer d_ff from the first query
    q0 = queries[0][1]
    a0 = capture_ff_activations_for_query(mb.model, mb.tokenizer, q0, args.layer, device)
    d_ff = int(a0.numel())

    # bookkeeping
    # category set & mapping
    cats_order = []
    seen = set()
    for _,_,c in queries:
        if c not in seen:
            seen.add(c); cats_order.append(c)
    C = len(cats_order)
    cat2idx = {c:i for i,c in enumerate(cats_order)}
    # counts
    cat_totals = np.zeros(C, dtype=np.int64)             # #queries per category
    fired_counts = np.zeros(d_ff, dtype=np.int64)        # total fired per neuron
    fired_by_cat = np.zeros((d_ff, C), dtype=np.int64)   # fired per neuron per category

    # include the first query we already computed
    def fire_mask_from(a: torch.Tensor) -> np.ndarray:
        x = a.detach().float().cpu().numpy()
        x = np.where(np.abs(x) < args.act_eps, 0.0, x)
        if args.fire_policy == "topk":
            k = max(1, int(math.ceil(args.top_p * x.size)))
            idx = np.argpartition(np.abs(x), -k)[-k:]
            m = np.zeros_like(x, dtype=bool); m[idx] = True
            return m
        else:
            mu = x.mean(); sd = x.std(ddof=0)
            if sd == 0: return np.zeros_like(x, dtype=bool)
            z = (x - mu) / sd
            return np.abs(z) >= args.zthr

    # process all queries
    for qi, (qid, q, cat) in enumerate(queries):
        if qi == 0:
            a = a0
        else:
            a = capture_ff_activations_for_query(mb.model, mb.tokenizer, q, args.layer, device)
        cat_idx = cat2idx[cat]
        cat_totals[cat_idx] += 1
        m = fire_mask_from(a)
        fired_counts += m.astype(np.int64)
        fired_by_cat[:, cat_idx] += m.astype(np.int64)

        if (qi+1) % 200 == 0:
            print(f"[L{args.layer}] processed {qi+1}/{len(queries)} queries...", flush=True)

    # compute metrics per neuron
    nQ = len(queries)
    base_rate = fired_counts / max(1, nQ)                      # activity_rate
    # rates per category
    rates_by_cat = np.zeros_like(fired_by_cat, dtype=np.float64)
    for c in range(C):
        if cat_totals[c] > 0:
            rates_by_cat[:, c] = fired_by_cat[:, c] / cat_totals[c]

    # specialization: entropy over category rates (normalized)
    H = np.zeros(d_ff, dtype=np.float64)
    S = np.zeros(d_ff, dtype=np.float64)  # spec_score = 1 - H/log(C)
    norm = math.log(C) if C > 1 else 1.0
    for j in range(d_ff):
        p = rates_by_cat[j, :]
        if p.sum() > 0:
            pj = p / p.sum()
            H[j] = entropy(pj)
            S[j] = 1.0 - (H[j] / norm)
        else:
            H[j] = 0.0; S[j] = 0.0

    # top category & lift
    top_cat_idx = rates_by_cat.argmax(axis=1)
    top_cat_rate = rates_by_cat[np.arange(d_ff), top_cat_idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        lift = np.where(base_rate > 0, top_cat_rate / base_rate, 0.0)

    # write outputs
    out_dir = os.path.join(cfg["paths"]["out_dir"], "exp2", f"L{args.layer}")
    ensure_dir(out_dir)

    # neuron_specialization.csv
    header = [
        "neuron_id","layer","n_queries",
        "fired_count","activity_rate","entropy","spec_score",
        "top_category","top_cat_rate","top_cat_lift"
    ] + [f"cat_{c}_rate" for c in cats_order] + [f"cat_{c}_count" for c in cats_order]

    out_path = os.path.join(out_dir, "neuron_specialization.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for j in range(d_ff):
            row = [
                j, args.layer, nQ,
                int(fired_counts[j]), float(base_rate[j]), float(H[j]), float(S[j]),
                cats_order[int(top_cat_idx[j])], float(top_cat_rate[j]), float(lift[j]),
            ]
            row += [float(rates_by_cat[j, cat2idx[c]]) for c in cats_order]
            row += [int(fired_by_cat[j, cat2idx[c]]) for c in cats_order]
            w.writerow(row)
    print(f"[L{args.layer}] wrote {out_path}")

    # top_specialists.csv (sorted)
    idx = np.argsort(S * top_cat_rate)[::-1]
    top_path = os.path.join(out_dir, "top_specialists.csv")
    with open(top_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["neuron_id","layer","spec_score","top_category","top_cat_rate","activity_rate","top_cat_lift"])
        for j in idx[:200]:  # top 200 for convenience
            w.writerow([int(j), args.layer, float(S[j]), cats_order[int(top_cat_idx[j])], float(top_cat_rate[j]), float(base_rate[j]), float(lift[j])])
    print(f"[L{args.layer}] wrote {top_path}")

    # save a small meta.json
    meta = {
        "layer": args.layer,
        "num_queries": nQ,
        "fire_policy": args.fire_policy,
        "top_p": args.top_p,
        "zthr": args.zthr,
        "categories": cats_order,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[L{args.layer}] wrote meta.json")

if __name__ == "__main__":
    main()
