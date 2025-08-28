from __future__ import annotations
from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# --- locate the FF down-projection module (maps d_ff -> d_model) for a given decoder layer ---
def get_ff_down_proj_module(m: PreTrainedModel, layer_idx: int):
    # Try common layouts first
    dec = None
    if hasattr(m, "model"):
        mm = m.model
        if hasattr(mm, "decoder"):
            dec = mm.decoder
        elif hasattr(mm, "layers"):
            dec = mm
        else:
            dec = getattr(m, "decoder", None)
    else:
        dec = getattr(m, "decoder", None)
    if dec is None:
        raise AttributeError("Could not locate decoder stack on the model")

    layer = None
    for attr in ("block", "layers"):
        if hasattr(dec, attr):
            layer = getattr(dec, attr)[layer_idx]
            break
    if layer is None:
        try:
            layer = dec[layer_idx]
        except Exception as e:
            raise AttributeError(f"Could not index decoder layer {layer_idx}: {e}")

    # T5-style: last sublayer is FF, with DenseReluDense/DenseActDense.wo
    if hasattr(layer, "layer"):
        ff = layer.layer[-1]
        for name in ("DenseReluDense", "DenseActDense"):
            ffm = getattr(ff, name, None)
            if ffm is not None and hasattr(ffm, "wo"):
                return ffm.wo
        if hasattr(ff, "wo"):
            return ff.wo

    # BART/MBART-style: fc2
    if hasattr(layer, "fc2"):
        return layer.fc2

    # LLaMA-like: mlp.down_proj/out_proj/fc2
    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
    if mlp is not None:
        for cand in ("down_proj", "out_proj", "fc2"):
            if hasattr(mlp, cand):
                return getattr(mlp, cand)

    # Fallback: find a Linear whose weight matches [d_model, d_ff] with d_ff > d_model
    import torch.nn as nn
    d_model = getattr(getattr(m, "config", None), "d_model", None) or getattr(getattr(m, "config", None), "hidden_size", None)
    candidates = []
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and hasattr(mod, "weight"):
            W = mod.weight
            if W.dim()==2 and d_model is not None and W.shape[0]==d_model and W.shape[1]>d_model:
                candidates.append((name, mod))
    if candidates:
        # pick the largest second dim
        candidates.sort(key=lambda x: x[1].weight.shape[1], reverse=True)
        return candidates[0][1]

    raise AttributeError("Could not find FF down-projection module for this layer")


@torch.no_grad()
def capture_ff_activations_for_query(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    layer_idx: int,
    device: torch.device,
    max_input_len: int = 256,
) -> torch.Tensor:
    """
    Run a single decoder step on `text` and return the FF activations vector a ∈ R^{d_ff}
    seen as INPUT to the down-projection (wo/fc2/down_proj) at decoder layer `layer_idx`.

    Works on encoder-decoder AND decoder-only. Always returns a CPU tensor [d_ff].
    """
    model.eval()
    # Prepare inputs
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_input_len
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Single-token start for decoder
    dstart = getattr(model.config, "decoder_start_token_id", None)
    if dstart is None:
        # best-effort fallbacks
        dstart = getattr(tokenizer, "pad_token_id", None)
    if dstart is None:
        dstart = getattr(tokenizer, "eos_token_id", None)
    if dstart is None:
        # last resort: use BOS if exists
        dstart = getattr(tokenizer, "bos_token_id", None)
    if dstart is None:
        raise RuntimeError("Could not determine a decoder start token id.")

    decoder_input_ids = torch.tensor([[int(dstart)]], device=device)

    # Hook the input to the down-projection module
    ff_down = get_ff_down_proj_module(model, layer_idx)
    captured = {"a": None}

    def hook(mod, inp, out):
        # inp is a tuple with a single tensor of shape [batch, seq, d_ff]
        x = inp[0]
        # take the last time step (seq pos 0 for single token)
        a = x[:, -1, :].detach().to("cpu")  # [1, d_ff]
        captured["a"] = a

    h = ff_down.register_forward_hook(hook)

    try:
        if getattr(model.config, "is_encoder_decoder", False):
            model(**enc, decoder_input_ids=decoder_input_ids, use_cache=False)
        else:
            # decoder-only: just feed the tokens (query becomes the prompt)
            model(enc["input_ids"], attention_mask=enc.get("attention_mask"), use_cache=False)
    finally:
        h.remove()

    if captured["a"] is None:
        raise RuntimeError("Failed to capture activations (hook not triggered).")
    return captured["a"][0]  # [d_ff] on CPU
