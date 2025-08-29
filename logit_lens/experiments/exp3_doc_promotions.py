from __future__ import annotations
import argparse, csv, json, os, math
from typing import Dict, List
import numpy as np
import torch

from logit_lens.utils.config import load as load_cfg
from logit_lens.utils.model_io import (
    load_model, get_lm_head_weight, get_decoder_ff_out_weight, get_output_layernorm
)
from logit_lens.utils.docids import get_doc_token_ids
from logit_lens.utils.doc_lookup import build_doc_index
from logit_lens.utils.hooks import capture_ff_activations_for_query
from logit_lens.utils.nq10k import load_val_queries
from logit_lens.utils.lens import (
    logits_from_unit_delta, logits_from_delta_lnaware, restrict_to_docids
)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def project(delta_vec: torch.Tensor, lmW: torch.Tensor, lens: str, ln_mod):
    if lens == "lnaware":
        return logits_from_delta_lnaware(delta_vec, lmW, ln_mod)
    return logits_from_unit_delta(delta_vec, lmW)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="logit_lens/configs/logit_lens.yaml")
    ap.add_argument("--candidates", type=str, default="artifacts/exp2/exp2_candidates.csv")
    ap.add_argument("--layers", type=str, default="18 19 20 21 22 23")
    ap.add_argument("--num_queries", type=int, default=4000)
    ap.add_argument("--topk_docs", type=int, default=50)
    ap.add_argument("--act_eps", type=float, default=1e-5)
    ap.add_argument("--lens", type=str, default="simple", choices=["simple","lnaware"])
    ap.add_argument("--save_perquery_topk", type=int, default=0, help="if >0, save per-query topK (K=this value)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    mb = load_model(cfg)
    device = mb.device

    # doc ids & (optional) index for titles
    doc_ids = get_doc_token_ids(cfg["paths"]["model_dir"], cfg["retrieval"]["docid_vocab_path"], allow_autodiscover=True)
    try:
        doc_index = build_doc_index(cfg["paths"]["model_dir"], cfg["paths"]["data_dir"])
    except Exception as e:
        print("[WARN] doc_index build failed:", e)
        doc_index = {}

    # queries
    qrows = load_val_queries(cfg["paths"]["data_dir"], mb.tokenizer, doc_token_ids=doc_ids, max_examples=args.num_queries)
    queries = [(r.get("qid", i), r["query"]) for i, r in enumerate(qrows)]
    if not queries:
        raise RuntimeError("No queries loaded for exp3")

    # candidates by layer
    if not os.path.exists(args.candidates):
        raise FileNotFoundError(f"Missing candidates CSV: {args.candidates}")
    layer_allow = set(int(x) for x in args.layers.split())
    cand_by_layer: Dict[int, List[int]] = {}
    with open(args.candidates, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            L = int(row["layer"]); j = int(row["neuron_id"])
            if L in layer_allow:
                cand_by_layer.setdefault(L, []).append(j)
    if not cand_by_layer:
        raise RuntimeError("No candidates for requested layers")

    lmW = get_lm_head_weight(mb.model)
    ln_mod = get_output_layernorm(mb.model)

    for L, neurons in cand_by_layer.items():
        out_dir = os.path.join(cfg["paths"]["out_dir"], "exp3", f"L{L}")
        ensure_dir(out_dir)
        Wout = get_decoder_ff_out_weight(mb.model, L)  # [d_model, d_ff]

        for j in neurons:
            prom_path = os.path.join(out_dir, f"neuron_{j:06d}_doc_promotions.csv")
            sum_path  = os.path.join(out_dir, f"neuron_{j:06d}_exp3_summary.csv")
            if os.path.exists(sum_path) and os.path.exists(prom_path):
                print(f"[L{L} N{j}] skip (exists)")
                continue

            # per-query optional logging (top-1 only to keep file small)
            perq_fh = None
            perq_csv = None
            if args.save_perquery_topk > 0:
                perq_path = os.path.join(out_dir, f"neuron_{j:06d}_perquery_topk.csv")
                perq_fh = open(perq_path, "w", newline="", encoding="utf-8")
                perq_csv = csv.writer(perq_fh)
                perq_csv.writerow(["q_idx","qid","top_doc_token_id","top_doc_score"])

            doc_freq: Dict[int,int] = {}
            doc_sum:  Dict[int,float] = {}
            doc_max:  Dict[int,float] = {}
            active_count = 0

            w_col = Wout[:, j]  # [d_model]

            for qi, (qid, qtext) in enumerate(queries):
                a = capture_ff_activations_for_query(mb.model, mb.tokenizer, qtext, L, device)
                a_j = float(a[j])
                if abs(a_j) < args.act_eps:
                    continue
                active_count += 1

                delta = w_col * a[j].to(w_col.device)  # [d_model]
                logits_vocab = project(delta, lmW, args.lens, ln_mod)
                logits_docs = restrict_to_docids(logits_vocab, doc_ids)  # np.ndarray [n_docs]
                if logits_docs.size == 0:
                    continue

                idx = np.argsort(logits_docs)[::-1][:args.topk_docs]
                vals = logits_docs[idx]

                if perq_csv is not None and idx.size > 0:
                    top_tid = int(doc_ids[int(idx[0])])
                    perq_csv.writerow([qi, qid, top_tid, float(vals[0])])

                for di, v in zip(idx.tolist(), vals.tolist()):
                    tid = int(doc_ids[int(di)])
                    doc_freq[tid] = doc_freq.get(tid, 0) + 1
                    doc_sum[tid]  = doc_sum.get(tid, 0.0) + float(v)
                    doc_max[tid]  = max(doc_max.get(tid, float("-inf")), float(v))

                if (qi + 1) % 200 == 0:
                    print(f"[L{L} N{j}] processed {qi+1}/{len(queries)}", flush=True)

            if perq_fh is not None:
                perq_fh.close()

            # write promotions
            rows = []
            total_hits = sum(doc_freq.values()) if doc_freq else 0
            for tid, cnt in doc_freq.items():
                s = doc_sum.get(tid, 0.0)
                mx = doc_max.get(tid, float("-inf"))
                meta = doc_index.get(tid, {"token_str":"", "title":"", "text":""})
                rows.append({
                    "doc_token_id": tid,
                    "doc_token_str": meta.get("token_str",""),
                    "title": meta.get("title",""),
                    "freq": cnt,
                    "freq_frac": (cnt / total_hits) if total_hits>0 else 0.0,
                    "score_mean": (s / cnt) if cnt>0 else 0.0,
                    "score_max": mx if mx!=-float("inf") else 0.0
                })
            rows.sort(key=lambda r: (r["freq"], r["score_mean"], r["score_max"]), reverse=True)

            with open(prom_path, "w", newline="", encoding="utf-8") as fh:
                dw = csv.DictWriter(fh, fieldnames=["doc_token_id","doc_token_str","title","freq","freq_frac","score_mean","score_max"])
                dw.writeheader()
                dw.writerows(rows)
            print(f"[L{L} N{j}] wrote {prom_path} ({len(rows)} rows)")

            # summary
            H = 0.0
            if total_hits := sum(doc_freq.values()):
                ps = [cnt/total_hits for cnt in doc_freq.values()]
                H = -sum(p*math.log(p) for p in ps if p>0)
            if rows:
                top_tid = rows[0]["doc_token_id"]; top_freq = rows[0]["freq"]; top_frac = rows[0]["freq_frac"]
            else:
                top_tid, top_freq, top_frac = "", 0, 0.0
            with open(sum_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["layer","neuron_id","n_queries","active_count","active_rate","unique_docs","top_doc_token_id","top_doc_freq","top_doc_frac","doc_entropy"])
                w.writerow([L, j, len(queries), active_count, (active_count/len(queries)) if queries else 0.0,
                            len(rows), top_tid, top_freq, top_frac, H])
            print(f"[L{L} N{j}] wrote {sum_path}")

if __name__ == "__main__":
    main()
