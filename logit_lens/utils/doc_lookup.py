import glob, json, os, re
from typing import Any, Dict, Optional
from transformers import AutoTokenizer

DOC_TOKEN_RE = re.compile(r"^@DOC_ID_[\-0-9]+@$")
NUM_RE = re.compile(r"-?\d+")

def _find_documents_path(data_dir: str) -> Optional[str]:
    for pat in (
        os.path.join(data_dir, "documents-*.json"),
        os.path.join(data_dir, "docs*.json"),
        os.path.join(data_dir, "documents.json"),
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

def _deep_find_doc_token(obj: Any, depth: int = 0) -> Optional[str]:
    if depth > 5:
        return None
    if isinstance(obj, str) and DOC_TOKEN_RE.match(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _deep_find_doc_token(v, depth + 1)
            if t is not None:
                return t
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            t = _deep_find_doc_token(v, depth + 1)
            if t is not None:
                return t
    return None

def _deep_find_numeric_docid(rec: Dict[str, Any], depth: int = 0) -> Optional[int]:
    """
    Try common key names first; fall back to scanning values for an integer-looking doc id.
    """
    # 1) common key names
    priority_keys = [
        "docid", "doc_id", "document_id", "id", "docId", "docNo", "doc_no", "docnum",
    ]
    for k in priority_keys:
        if k in rec:
            v = rec[k]
            # direct int
            if isinstance(v, int):
                return v
            # string that contains an int
            if isinstance(v, str):
                m = NUM_RE.search(v)
                if m:
                    try:
                        return int(m.group(0))
                    except Exception:
                        pass

    # 2) loose scan limited depth
    if depth > 3:
        return None
    for v in rec.values():
        if isinstance(v, dict):
            x = _deep_find_numeric_docid(v, depth + 1)
            if x is not None:
                return x
        elif isinstance(v, (list, tuple)):
            for vv in v:
                if isinstance(vv, dict):
                    x = _deep_find_numeric_docid(vv, depth + 1)
                    if x is not None:
                        return x
        elif isinstance(v, str):
            m = NUM_RE.search(v)
            if m:
                try:
                    return int(m.group(0))
                except Exception:
                    pass
    return None

def _extract_title_text(rec: Dict[str, Any]) -> (str, str):
    title = ""
    text = ""
    # best-effort title
    for k in ("title", "doc_title", "page_title", "name", "heading"):
        if isinstance(rec.get(k), str):
            title = rec[k].strip()
            break
    # best-effort body/text
    for k in ("text", "body", "content", "passage", "abstract"):
        if isinstance(rec.get(k), str):
            text = rec[k].strip()
            break
    return title, text

def build_doc_index(model_dir: str, data_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Returns: { doc_token_id (int) : { 'token_str': '@DOC_ID_..@', 'title': str, 'text': str } }
    It first tries to find an explicit '@DOC_ID_...@' token in each record; if absent,
    it looks for a numeric doc id and converts it to the token string.
    """
    tok = AutoTokenizer.from_pretrained(os.path.abspath(model_dir), local_files_only=True, use_fast=True)
    path = _find_documents_path(data_dir)
    if path is None:
        raise FileNotFoundError(f"Could not find documents-*.json under {data_dir}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # normalize payload to list of dicts
    if isinstance(payload, dict):
        items = payload.get("data")
        if not isinstance(items, list):
            # fall back: assume values are doc-like
            items = [v for v in payload.values() if isinstance(v, dict)]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unrecognized documents JSON structure: {type(payload)}")

    out: Dict[int, Dict[str, Any]] = {}
    for rec in items:
        if not isinstance(rec, dict):
            continue

        token_str = _deep_find_doc_token(rec)
        if token_str is None:
            # build from numeric docid if possible
            num = _deep_find_numeric_docid(rec)
            if num is not None:
                token_str = f"@DOC_ID_{num}@"

        if token_str is None:
            continue  # no way to map

        tid = tok.convert_tokens_to_ids(token_str)
        if not isinstance(tid, int) or tid == tok.unk_token_id:
            continue  # token not in vocab

        title, text = _extract_title_text(rec)
        out[tid] = {"token_str": token_str, "title": title, "text": text}

    return out
