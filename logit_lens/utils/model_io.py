from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)


@dataclass
class ModelBundle:
    cfg: dict
    device: torch.device
    dtype: torch.dtype
    tokenizer: Any
    model: Any


def _pick_device(cfg: dict) -> torch.device:
    dev = cfg.get("compute", {}).get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _pick_dtype(cfg: dict) -> torch.dtype:
    dt = cfg.get("compute", {}).get("dtype", "float16")
    if dt in ("float16", "fp16", "half"):
        return torch.float16
    if dt in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def load_model(cfg: dict, model_dir: Optional[str] = None) -> ModelBundle:
    paths = cfg.get("paths", {})
    model_dir = model_dir or paths.get("model_dir", "./model")

    device = _pick_device(cfg)
    dtype = _pick_dtype(cfg)

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    hf_cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)

    # Prefer seq2seq if encoder-decoder; otherwise causal
    model = None
    try:
        if getattr(hf_cfg, "is_encoder_decoder", False):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=dtype
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=dtype
            )
    except Exception:
        # Try the other family if the first guess failed
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=dtype
            )
        except Exception:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=dtype
            )

    model.to(device)
    model.eval()

    return ModelBundle(cfg=cfg, device=device, dtype=dtype, tokenizer=tok, model=model)


def get_lm_head_weight(m: Any) -> torch.Tensor:
    """
    Return lm_head weight as [vocab_size, d_model].
    Falls back to tied output embeddings if needed.
    """
    head = getattr(m, "lm_head", None)
    if head is None:
        if hasattr(m, "get_output_embeddings") and m.get_output_embeddings() is not None:
            head = m.get_output_embeddings()
        else:
            raise AttributeError("Model has no lm_head or output embeddings.")
    W = head.weight.data
    if W.dim() != 2:
        raise ValueError(f"Unexpected lm_head weight dim: {tuple(W.shape)}")
    return W


def _get_decoder_stack(m: Any):
    """
    Return a tuple (decoder_stack, attr_name_for_layers)
    attr_name_for_layers is either 'block' or 'layers' or None if direct indexing works.
    """
    # Common HF layouts
    if hasattr(m, "model"):
        mm = m.model
        if hasattr(mm, "decoder"):
            dec = mm.decoder
            if hasattr(dec, "block"):
                return dec, "block"
            if hasattr(dec, "layers"):
                return dec, "layers"
            return dec, None
        # decoder-only stacks sometimes expose layers directly on m.model
        if hasattr(mm, "layers"):
            return mm, "layers"
        return mm, None
    # Some models may expose decoder directly on the top-level module
    if hasattr(m, "decoder"):
        dec = m.decoder
        if hasattr(dec, "block"):
            return dec, "block"
        if hasattr(dec, "layers"):
            return dec, "layers"
        return dec, None
    return None, None


def _get_decoder_layer(m: Any, layer_idx: int):
    dec, attr = _get_decoder_stack(m)
    if dec is None:
        raise AttributeError("Could not locate decoder stack on the model")
    if attr and hasattr(dec, attr):
        seq = getattr(dec, attr)
        return seq[layer_idx]
    # Try direct indexing
    try:
        return dec[layer_idx]
    except Exception as e:
        raise AttributeError(f"Could not index decoder layer {layer_idx}: {e}")


def get_decoder_ff_out_weight(m: Any, layer_idx: int) -> torch.Tensor:
    """
    Return decoder FF out-projection weight for a given layer, normalized to [d_model, d_ff]
    (so column j = W[:, j]).

    Strategy:
      1) Try known layouts (T5, BART/MBART, LLaMA-like).
      2) Generic fallback: find an nn.Linear in the decoder layer whose weight has shape
         [d_model, d_ff] with d_ff > d_model.
    """
    layer = _get_decoder_layer(m, layer_idx)

    # 1) Known layouts

    # T5-style: model.model.decoder.block[i].layer[2].DenseReluDense.wo
    try:
        if hasattr(layer, "layer"):
            ff = getattr(layer, "layer")[-1]  # last sublayer is FF
            for name in ("DenseReluDense", "DenseActDense"):
                ffm = getattr(ff, name, None)
                if ffm is not None and hasattr(ffm, "wo"):
                    W = ffm.wo.weight.data  # [d_model, d_ff]
                    return W
            if hasattr(ff, "wo"):
                return ff.wo.weight.data
    except Exception:
        pass

    # BART/MBART-style: decoder.layers[i].fc2.weight
    try:
        if hasattr(layer, "fc2"):
            return layer.fc2.weight.data  # [d_model, d_ff]
    except Exception:
        pass

    # LLaMA/decoder-only: down_proj / out_proj / fc2 under mlp
    try:
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if mlp is not None:
            if hasattr(mlp, "down_proj"):
                return mlp.down_proj.weight.data  # [d_model, d_ff]
            if hasattr(mlp, "out_proj"):
                return mlp.out_proj.weight.data
            if hasattr(mlp, "fc2"):
                return mlp.fc2.weight.data
    except Exception:
        pass

    # 2) Generic fallback by shape
    # Determine d_model from config (or from lm_head if missing)
    d_model = getattr(getattr(m, "config", None), "d_model", None)
    if d_model is None:
        d_model = getattr(getattr(m, "config", None), "hidden_size", None)
    if d_model is None:
        try:
            Wv = get_lm_head_weight(m)
            d_model = Wv.shape[1]
        except Exception:
            raise AttributeError("Cannot infer d_model to locate FF out-proj.")

    import torch.nn as nn
    candidates = []
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and hasattr(mod, "weight"):
            W = mod.weight.data
            if W.dim() == 2 and W.shape[0] == d_model and W.shape[1] > d_model:
                candidates.append((name, W))
    if candidates:
        # Choose the largest second dim as the FF out-projection
        name, W = max(candidates, key=lambda x: x[1].shape[1])
        return W

    raise AttributeError(f"Could not find decoder FF out-projection for layer {layer_idx}")


def get_output_layernorm(m: Any):
    """
    Try to fetch the LayerNorm applied before lm_head.
    Returns a torch.nn.LayerNorm or None.
    """
    # Common seq2seq (T5/BART-like)
    if hasattr(m, "model"):
        mm = m.model
        # T5 decoder final LN
        if hasattr(mm, "decoder") and hasattr(mm.decoder, "final_layer_norm"):
            return mm.decoder.final_layer_norm
        # BART/MBART decoder LN
        if hasattr(mm, "decoder") and hasattr(mm.decoder, "layer_norm"):
            return mm.decoder.layer_norm
        # Some models keep a top-level norm
        if hasattr(mm, "norm"):
            return mm.norm
    # GPT-like
    for cand in ("ln_f", "norm", "layer_norm"):
        if hasattr(m, cand):
            return getattr(m, cand)
    return None
