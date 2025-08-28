from __future__ import annotations
import argparse, csv, os
from typing import List, Dict, Any, Tuple

import numpy as np
import torch

from logit_lens.utils.config import load as load_cfg
from logit_lens.utils.model_io import (
    load_model,
    get_lm_head_weight,
    get_decoder_ff_out_weight,
    get_output_layernorm,
)
from logit_lens.utils.docids import get_doc_token_ids
from logit_lens.utils.lens import (
    logits_from_unit_delta,
    restrict_to_docids,
    analyze_doc_logits,
    logits_from_delta_lnaware,
)
from logit_lens.utils.hooks import capture_ff_activations_for_query
from logit_lens.utils.nq10k import load_val_queries


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_topk_csv(path: str, doc_token_ids: List[int], top_indices: np.ndarray, top_values: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_token_id", "rank", "score"])
        for r, (i, v) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
            w.writerow([doc_token_ids[int(i)], r, float(v)])


def save_stats_csv(path: str, neuron_id: int, layer: int, stats: Dict[str, float], thr: Dict[str, int]) -> None:
    hdr = ["neuron_id", "layer"] + list(stats.keys()) + list(thr.keys())
    vals = [neuron_id, layer] + [stats[k] for k in stats] + [thr[k] for k in thr]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerow(vals)


def _project_vocab(delta_vec: torch.Tensor, lmW: torch.Tensor, lens_kind: str, ln_mod: Any) -> torch.Tensor:
    if lens_kind == "lnaware":
        return logits_from_delta_lnaware(delta_vec, lmW, ln_mod)
    return logits_from_unit_delta(delta_vec, lmW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="logit_lens/configs/logit_lens.yaml")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--neuron", type=int, required=True)
    ap.add_argument("--model_dir", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--lens", type=str, default="simple", choices=["simple", "lnaware"], help="projection lens for logits")
    ap.add_argument("--use_nq_val", type=int, default=0, help="if >0, use first N queries from NQ10k val split and write perquery CSV with GT metrics when available")
    ap.add_argument("--act_eps", type=float, default=1e-6, help="activation threshold to consider neuron active")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.model_dir: cfg.setdefault("paths", {})["model_dir"] = args.model_dir
    if args.data_dir:  cfg.setdefault("paths", {})["data_dir"]  = args.data_dir
    if args.out_dir:   cfg.setdefault("paths", {})["out_dir"]   = args.out_dir

    out_base = os.path.join(cfg["paths"]["out_dir"], "exp1", f"L{args.layer}")
    ensure_dir(out_base)

    # Load model + weights
    mb = load_model(cfg, model_dir=cfg["paths"]["model_dir"])
    lmW = get_lm_head_weight(mb.model)              # [vocab, d_model]
    Wout = get_decoder_ff_out_weight(mb.model, args.layer)  # [d_model, d_ff]
    ln_mod = get_output_layernorm(mb.model)

    d_model, d_ff = Wout.shape
    if not (0 <= args.neuron < d_ff):
        raise ValueError(f"neuron id {args.neuron} out of range d_ff={d_ff}")

    # Doc token ids
    doc_ids = get_doc_token_ids(
        model_dir=cfg["paths"]["model_dir"],
        docid_vocab_path=cfg["retrieval"]["docid_vocab_path"],
        allow_autodiscover=False,
    )

    # Unit (query-independent) logit lens
    w_col = Wout[:, args.neuron]  # [d_model]
    logits_vocab = _project_vocab(w_col, lmW, args.lens, ln_mod)  # [vocab]
    logits_docs = restrict_to_docids(logits_vocab, doc_ids)

    eval_topk = cfg.get("retrieval", {}).get("eval_topk", [1, 5, 10, 20])
    thresholds = cfg.get("retrieval", {}).get("thresholds", {})
    abs_thr = thresholds.get("absolute", [0.0])
    q_thr = thresholds.get("quantiles", [0.99, 0.95])

    analysis = analyze_doc_logits(logits_docs, eval_topk, abs_thr, q_thr)

    topk_k = max(eval_topk) if eval_topk else min(20, len(doc_ids))
    topv, topi = analysis["topk_values"][:topk_k], analysis["topk_indices"][:topk_k]

    topk_path = os.path.join(out_base, f"neuron_{args.neuron:06d}_topk.csv")
    save_topk_csv(topk_path, doc_ids, topi, topv)

    stats_path = os.path.join(out_base, f"neuron_{args.neuron:06d}_stats.csv")
    save_stats_csv(stats_path, args.neuron, args.layer, analysis["stats"], analysis["thr"])

    print(f"[OK] unit lens — neuron {args.neuron} layer {args.layer} ({args.lens})")
    print(f" wrote {topk_path}")
    print(f" wrote {stats_path}")

    # Optional: per-query with NQ10k validation queries
    if args.use_nq_val and args.use_nq_val > 0:
        nq_rows = load_val_queries(cfg["paths"]["data_dir"], mb.tokenizer, doc_token_ids=doc_ids, max_examples=args.use_nq_val)

        perq_path = os.path.join(out_base, f"neuron_{args.neuron:06d}_perquery.csv")
        with open(perq_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            hdr = [
                "q_idx", "qid", "query", "a_j", "active",
                "top1_doc_token_id", "top1_score",
                "gt_doc_token_id", "gt_rank", "hit@1", "hit@5", "hit@10", "hit@20",
            ]
            w.writerow(hdr)

            # accumulators for summary
            sum_active = 0
            a_active: List[float] = []
            top1_active: List[int] = []
            gt_with_label = 0
            hits = {"@1": 0, "@5": 0, "@10": 0, "@20": 0}

            for qi, row in enumerate(nq_rows):
                q = row["query"]
                a_vec = capture_ff_activations_for_query(mb.model, mb.tokenizer, q, args.layer, mb.device)  # [d_ff]
                a_j = float(a_vec[args.neuron])

                if abs(a_j) < args.act_eps:
                    w.writerow([qi, row.get("qid", qi), q, a_j, "inactive", "", "", row.get("gt_doc_token_id", ""), "", 0, 0, 0, 0])
                    continue

                sum_active += 1
                a_active.append(a_j)

                delta = Wout[:, args.neuron] * (a_vec[args.neuron].to(Wout.device))
                logits_vocab_q = _project_vocab(delta, lmW, args.lens, ln_mod)
                logits_docs_q = restrict_to_docids(logits_vocab_q, doc_ids)

                # top-1
                top1_docid, top1_score = "", ""
                if logits_docs_q.size > 0:
                    idx1 = int(logits_docs_q.argmax())
                    top1_docid = int(doc_ids[idx1])
                    top1_score = float(logits_docs_q[idx1])
                    top1_active.append(top1_docid)

                # ground-truth rank + hits
                gt_tok = row.get("gt_doc_token_id", None)
                gt_rank = ""
                h1 = h5 = h10 = h20 = 0
                if gt_tok is not None and logits_docs_q.size > 0:
                    idx_sorted = logits_docs_q.argsort()[::-1]
                    gt_rank_pos = None
                    for r_pos, i_doc in enumerate(idx_sorted, start=1):
                        if doc_ids[int(i_doc)] == gt_tok:
                            gt_rank_pos = r_pos
                            break
                    if gt_rank_pos is not None:
                        gt_rank = int(gt_rank_pos)
                        gt_with_label += 1
                        h1 = 1 if gt_rank <= 1 else 0
                        h5 = 1 if gt_rank <= 5 else 0
                        h10 = 1 if gt_rank <= 10 else 0
                        h20 = 1 if gt_rank <= 20 else 0
                        hits["@1"] += h1
                        hits["@5"] += h5
                        hits["@10"] += h10
                        hits["@20"] += h20

                w.writerow([
                    qi, row.get("qid", qi), q, a_j, "active",
                    top1_docid, top1_score,
                    row.get("gt_doc_token_id", ""), gt_rank, h1, h5, h10, h20
                ])

        print(f" wrote {perq_path}")

        # summary
        sum_path = os.path.join(out_base, f"neuron_{args.neuron:06d}_summary.csv")
        total = len(nq_rows)
        active_rate = (sum_active / total) if total > 0 else 0.0
        mean_a = (sum(a_active) / sum_active) if sum_active > 0 else 0.0
        std_a = ((sum((x - mean_a) ** 2 for x in a_active) / (sum_active - 1)) ** 0.5) if sum_active > 1 else 0.0

        top1_mode_id, top1_mode_frac = "", 0.0
        if top1_active:
            from collections import Counter
            c = Counter(top1_active)
            top1_mode_id, cnt = c.most_common(1)[0]
            top1_mode_frac = cnt / len(top1_active)

        hdr = [
            "neuron_id", "layer", "lens", "n_queries",
            "active_count", "active_rate",
            "a_j_mean_active", "a_j_std_active",
            "top1_mode_doc_token_id", "top1_mode_frac",
            "gt_n_with_label", "hits@1", "hits@5", "hits@10", "hits@20",
        ]
        row = [
            args.neuron, args.layer, args.lens, total,
            sum_active, active_rate,
            mean_a, std_a,
            top1_mode_id, top1_mode_frac,
            gt_with_label, hits["@1"], hits["@5"], hits["@10"], hits["@20"],
        ]
        with open(sum_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerow(row)

        print(f" wrote {sum_path}")


if __name__ == "__main__":
    main()
