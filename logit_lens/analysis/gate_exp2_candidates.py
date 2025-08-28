import argparse, csv, os, math
from collections import defaultdict

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def to_float(d, k, default=0.0):
    try:
        return float(d.get(k, default))
    except Exception:
        return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=str, default="18 19 20 21 22 23")
    ap.add_argument("--results_dir", type=str, default="logit_lens/results/exp2")
    ap.add_argument("--out", type=str, default="artifacts/exp2/exp2_candidates.csv")
    ap.add_argument("--spec_min", type=float, default=0.60)
    ap.add_argument("--lift_min", type=float, default=2.0)
    ap.add_argument("--act_min", type=float, default=0.01)
    ap.add_argument("--act_max", type=float, default=0.30)
    ap.add_argument("--top_per_cat", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    layers = [int(x) for x in args.layers.split()]
    rows_out = []
    per_group = defaultdict(list)  # (layer, top_category) -> list

    for L in layers:
        spec_path = os.path.join(args.results_dir, f"L{L}", "neuron_specialization.csv")
        if not os.path.exists(spec_path):
            print(f"[WARN] missing {spec_path}, skipping")
            continue
        for r in read_csv(spec_path):
            spec = to_float(r, "spec_score")
            lift = to_float(r, "top_cat_lift")
            act  = to_float(r, "activity_rate")
            if not (spec >= args.spec_min and lift >= args.lift_min and args.act_min <= act <= args.act_max):
                continue
            top_cat = r.get("top_category","other")
            top_rate = to_float(r, "top_cat_rate")
            score = spec * top_rate * min(lift, 5.0)  # bounded lift
            rows_out.append({
                "layer": L,
                "neuron_id": int(r["neuron_id"]),
                "top_category": top_cat,
                "spec_score": spec,
                "top_cat_rate": top_rate,
                "activity_rate": act,
                "top_cat_lift": lift,
                "rank_score": score,
            })
            per_group[(L, top_cat)].append(rows_out[-1])

    # rank within each (layer, category), keep top_k
    selected = []
    for key, lst in per_group.items():
        lst.sort(key=lambda x: (x["rank_score"], x["spec_score"], x["top_cat_rate"], x["top_cat_lift"]), reverse=True)
        selected.extend(lst[:args.top_per_cat])

    # global sort for readability
    selected.sort(key=lambda x: (x["layer"], x["top_category"], x["rank_score"]), reverse=True)

    # write
    header = ["layer","neuron_id","top_category","spec_score","top_cat_rate","activity_rate","top_cat_lift","rank_score"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in selected:
            w.writerow(r)
    print(f"[OK] wrote {args.out} with {len(selected)} candidates.")

if __name__ == "__main__":
    main()
