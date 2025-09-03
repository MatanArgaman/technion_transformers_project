import re
import yaml

class CategoryMatcher:
    def __init__(self, rules):
        self.rules = []
        for r in (rules or []):
            name = r.get("name", "other")
            incs = [re.compile(p, re.I) for p in r.get("include", [])]
            self.rules.append((name, incs))

    def label(self, text):
        t = (text or "").strip()
        for name, pats in self.rules:
            for p in pats:
                if p.search(t):
                    return name
        return "other"

def load_categories(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return CategoryMatcher(data.get("categories", []))
