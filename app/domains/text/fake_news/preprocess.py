# app/domains/text/fake_news/preprocess.py
import re
from typing import List

_whitespace = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = _whitespace.sub(" ", text)
    return text

def normalize_texts(texts: List[str]) -> List[str]:
    return [normalize_text(t) for t in texts]
