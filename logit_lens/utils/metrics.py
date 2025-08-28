from __future__ import annotations
import math
from typing import Dict, Iterable, List, Tuple
import numpy as np

EPS = 1e-12

def topk_indices(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (topk_values, topk_indices) sorted by value desc."""
    if k <= 0:
        return np.array([]), np.array([], dtype=int)
    k = min(k, x.shape[-1])
    idx = np.argpartition(-x, k-1)[:k]
    # sort those k by value desc
    idx = idx[np.argsort(-x[idx])]
    return x[idx], idx

def margin_top1_top2(x: np.ndarray) -> float:
    """Top-1 minus Top-2 margin. If len(x)<2, margin is nan."""
    if x.size < 2:
        return float("nan")
    # no need to sort all: use partition twice
    a = np.partition(-x, 0)[0]      # largest (negated)
    b = np.partition(-x, 1)[1]      # second largest (negated)
    return (-a) - (-b)

def entropy_perplexity(x: np.ndarray) -> Tuple[float, float]:
    """Shannon entropy (nats) & perplexity over softmax-normalized x."""
    # use log-sum-exp trick for stability
    m = x.max()
    ex = np.exp(x - m)
    p = ex / (ex.sum() + EPS)
    ent = -np.sum(p * (np.log(p + EPS)))
    ppl = float(np.exp(ent))
    return float(ent), ppl

def gini(x: np.ndarray) -> float:
    """Gini coefficient of nonnegative vector (use softmax to ensure nonnegativity)."""
    # Use softmax probs for a consistent distributional measure
    m = x.max()
    ex = np.exp(x - m)
    p = ex / (ex.sum() + EPS)
    # Gini for probabilities
    n = p.size
    # sort ascending
    ps = np.sort(p)
    cum = np.cumsum(ps)
    g = (n + 1 - 2 * np.sum(cum) / (cum[-1] + EPS)) / n
    return float(max(0.0, min(1.0, g)))

def hhi(x: np.ndarray) -> float:
    """Herfindahl–Hirschman Index on softmax probs."""
    m = x.max()
    ex = np.exp(x - m)
    p = ex / (ex.sum() + EPS)
    return float(np.sum(p**2))

def distribution_stats(x: np.ndarray) -> Dict[str, float]:
    ent, ppl = entropy_perplexity(x)
    return {
        "entropy": ent,
        "perplexity": ppl,
        "gini": gini(x),
        "hhi": hhi(x),
        "margin_top1_top2": margin_top1_top2(x),
    }

def thresholds_apply(x: np.ndarray, absolute: Iterable[float] = (), quantiles: Iterable[float] = ()) -> Dict[str, int]:
    """
    Count how many entries exceed absolute thresholds or quantile cutoffs (on raw x).
    quantiles are in (0,1), e.g., 0.95.
    """
    out: Dict[str, int] = {}
    x_sorted = np.sort(x)
    n = x_sorted.size
    for a in absolute:
        out[f"thr_abs_{a:g}"] = int((x > a).sum())
    for q in quantiles:
        q = float(q)
        q = min(max(q, 0.0), 1.0)
        idx = int(math.floor(q * max(n-1, 0)))
        cutoff = x_sorted[idx] if n > 0 else float("inf")
        out[f"thr_q_{q:.2f}"] = int((x >= cutoff).sum())
    return out

def recall_at_k(rank: int, k: int) -> int:
    return 1 if (rank > 0 and rank <= k) else 0
