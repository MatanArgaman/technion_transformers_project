from __future__ import annotations
import os, io
from typing import Any, Dict
import yaml

_DEFAULT_CFG_PATH = "logit_lens/configs/logit_lens.yaml"

class Cfg(dict):
    """Small dict subclass for dot-ish access via cfg['a']['b'] only (explicit)."""
    pass

def load(path: str = _DEFAULT_CFG_PATH) -> Cfg:
    """Load YAML config into a plain dict (wrapped as Cfg)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with io.open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")
    return Cfg(data)
