from __future__ import annotations
import json, os, re
from typing import List, Optional, Sequence
from transformers import AutoTokenizer

_DOC_PATTERNS = [
    r"^<doc[_-]?\d+>$",
    r"^<d[_-]?\d+>$",
    r"^doc[_-]?\d+$",
    r"^<\|docid:\d+\|>$",
    r"^\[DOCID:\d+\]$",
]

def _compile_patterns() -> List[re.Pattern]:
    return [re.compile(p) for p in _DOC_PATTERNS]

def load_tokenizer(model_dir: str):
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)

def _discover_doc_tokens(tokenizer) -> List[int]:
    vocab = tokenizer.get_vocab()  # token->id
    pats = _compile_patterns()
    ids: List[int] = []
    for tok, tid in vocab.items():
        t = tok
        # many tokenizers escape special tokens differently; normalize trivial cases
        if isinstance(t, bytes):
            try:
                t = t.decode("utf-8", "ignore")
            except Exception:
                pass
        for pat in pats:
            if pat.match(t):
                ids.append(int(tid))
                break
    # de-duplicate & sort
    ids = sorted(set(ids))
    return ids

def get_doc_token_ids(
    model_dir: str = "./model",
    docid_vocab_path: str = "./logit_lens/results/shared/docid_vocab.json",
    allow_autodiscover: bool = True,
    min_expected: Optional[int] = None,
) -> List[int]:
    """
    Return list of token IDs representing document identifiers.

    Priority:
      1) If docid_vocab_path exists -> load list[int]
      2) Else, if allow_autodiscover -> scan tokenizer vocab with regex patterns
      3) Else, raise with helpful message
    """
    # 1) user-provided file
    if docid_vocab_path and os.path.exists(docid_vocab_path):
        with open(docid_vocab_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "doc_token_ids" in obj:
            ids = list(map(int, obj["doc_token_ids"]))
        elif isinstance(obj, list):
            ids = list(map(int, obj))
        else:
            raise ValueError(f"Unrecognized docid mapping format in {docid_vocab_path}")
        if not ids:
            raise ValueError(f"{docid_vocab_path} found but empty.")
        return sorted(set(ids))

    # 2) autodiscover
    if allow_autodiscover:
        tok = load_tokenizer(model_dir)
        ids = _discover_doc_tokens(tok)
        if not ids:
            raise RuntimeError(
                "Could not autodiscover DocID tokens from tokenizer vocab. "
                "Please create a JSON file at retrieval.docid_vocab_path containing a list of token IDs "
                "or {'doc_token_ids': [...]}."
            )
        if min_expected is not None and len(ids) < min_expected:
            # Not fatal, but warn loudly (caller can decide)
            print(f"[WARN] Only discovered {len(ids)} DocID tokens (< {min_expected}). Using anyway.")
        return ids

    # 3) give up
    raise FileNotFoundError(
        f"No docid mapping file at {docid_vocab_path} and autodiscovery disabled."
    )
