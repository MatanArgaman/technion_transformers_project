from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch

from .metrics import topk_indices, distribution_stats, thresholds_apply

@torch.no_grad()
def logits_from_unit_delta(
    w_out_col: torch.Tensor,          # [d_model] column = W[:, j]
    lm_head_weight: torch.Tensor,     # [vocab, d_model]
) -> torch.Tensor:
    """
    Compute vocabulary logits produced by a unit delta-residual equal to W_out[:, j].
    logits_vocab[v] = lm_head_weight[v] dot w_out_col
    Returns torch vector [vocab].
    """
    if w_out_col.dim() != 1:
        w = w_out_col.view(-1)
    else:
        w = w_out_col
    V, D = lm_head_weight.shape
    assert w.shape[0] == D, f"Dim mismatch: w_out_col={tuple(w.shape)} vs lm_head_weight={tuple(lm_head_weight.shape)}"
    logits_vocab = lm_head_weight @ w   # [vocab]
    return logits_vocab

def restrict_to_docids(
    logits_vocab: torch.Tensor, doc_token_ids: List[int]
) -> np.ndarray:
    """
    Select only doc-token ids and return numpy array [#docids] of logits.
    """
    idx = torch.tensor(doc_token_ids, dtype=torch.long, device=logits_vocab.device)
    sel = logits_vocab.index_select(0, idx)  # [#docids]
    return sel.detach().cpu().numpy()

def analyze_doc_logits(
    logits_docs: np.ndarray,
    eval_topk: List[int],
    thresholds_abs: List[float],
    thresholds_q: List[float],
) -> Dict:
    """
    Compute top-k list and distribution metrics/threshold counts.
    Returns a dict with:
      - 'topk_values', 'topk_indices' (indices within doc_token_ids array)
      - 'stats' dict with entropy/perplexity/gini/hhi/margin
      - 'thr' dict with threshold counts
    """
    k = max(eval_topk) if eval_topk else min(20, logits_docs.size)
    topv, topi = topk_indices(logits_docs, k)
    stats = distribution_stats(logits_docs)
    thr = thresholds_apply(logits_docs, absolute=thresholds_abs, quantiles=thresholds_q)
    return {"topk_values": topv, "topk_indices": topi, "stats": stats, "thr": thr}


def _apply_layer_norm_vector(x: torch.Tensor, ln_module) -> torch.Tensor:
    """
    Apply the given nn.LayerNorm to a single hidden vector x [d_model].
    """
    if ln_module is None:
        return x
    # manual LN so it works even if ln_module expects [*, d_model]
    eps = getattr(ln_module, "eps", 1e-5)
    w = getattr(ln_module, "weight", None)
    b = getattr(ln_module, "bias", None)
    mu = x.mean()
    var = (x - mu).pow(2).mean()
    y = (x - mu) / torch.sqrt(var + eps)
    if w is not None:
        y = y * w.to(x.device, x.dtype)
    if b is not None:
        y = y + b.to(x.device, x.dtype)
    return y

@torch.no_grad()
def logits_from_delta_lnaware(
    delta: torch.Tensor,              # [d_model]
    lm_head_weight: torch.Tensor,     # [vocab, d_model]
    ln_module,                        # nn.LayerNorm or None
) -> torch.Tensor:
    """
    LN-aware logits: apply output LayerNorm to delta before unembedding.
    """
    y = _apply_layer_norm_vector(delta.view(-1), ln_module)
    return lm_head_weight @ y
