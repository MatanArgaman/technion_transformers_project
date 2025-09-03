from __future__ import annotations
import glob, json, os, re
from typing import Iterable, List, Optional, Tuple, Any
from transformers import PreTrainedTokenizerBase

DOC_TOKEN_RE = re.compile(r"^@DOC_ID_[\-0-9]+@$")

def _find_file(patterns: Iterable[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

def locate_val_and_docs(data_dir: str) -> Tuple[Optional[str], Optional[str]]:
    valp = _find_file([os.path.join(data_dir, "val_queries-*.json"),
                       os.path.join(data_dir, "val*.json")])
    docp = _find_file([os.path.join(data_dir, "documents-*.json"),
                       os.path.join(data_dir, "docs*.json")])
    return valp, docp

def _token_str_to_id(tok: PreTrainedTokenizerBase, s: str) -> Optional[int]:
    if not isinstance(s, str):
        return None
    tid = tok.convert_tokens_to_ids(s)
    return int(tid) if isinstance(tid, int) and tid != tok.unk_token_id else None

def _deep_find_doc_token(obj: Any, depth: int = 0) -> Optional[str]:
    """Only accept explicit @DOC_ID_...@ strings anywhere in the record."""
    if depth > 4:
        return None
    if isinstance(obj, str) and DOC_TOKEN_RE.match(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _deep_find_doc_token(v, depth+1)
            if t is not None:
                return t
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            t = _deep_find_doc_token(v, depth+1)
            if t is not None:
                return t
    return None

def _extract_query_and_gt(raw: dict) -> Tuple[Optional[str], Optional[str]]:
    q = None
    for k in ("query", "question", "text", "input", "prompt"):
        if k in raw and isinstance(raw[k], str) and raw[k].strip():
            q = raw[k].strip()
            break
    gt_tok = _deep_find_doc_token(raw, 0)
    return q, gt_tok

def load_val_queries(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    doc_token_ids: Optional[List[int]] = None,
    max_examples: Optional[int] = None,
) -> List[dict]:
    valp, _ = locate_val_and_docs(data_dir)
    if valp is None:
        raise FileNotFoundError(f"Could not find val queries in {data_dir}")

    with open(valp, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        for key in ("data", "examples", "items", "val", "records"):
            if key in payload and isinstance(payload[key], list):
                items = payload[key]; break
        else:
            items = list(payload.values())
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unrecognized val queries shape: {type(payload)}")

    out: List[dict] = []
    doc_set = set(doc_token_ids) if doc_token_ids is not None else None
    for i, raw in enumerate(items):
        base = raw if isinstance(raw, dict) else {}
        q, doc_tok_str = _extract_query_and_gt(base)
        if not q:
            continue
        rec = {"qid": i, "query": q}
        if doc_tok_str is not None:
            tid = _token_str_to_id(tokenizer, doc_tok_str)
            if tid is not None and (doc_set is None or tid in doc_set):
                rec["gt_doc_token_id"] = tid
        out.append(rec)
        if max_examples is not None and len(out) >= max_examples:
            break
    return out
