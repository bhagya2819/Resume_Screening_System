"""Extract technical and soft skills via spaCy PhraseMatcher."""
from __future__ import annotations

import json
from functools import lru_cache

from spacy.matcher import PhraseMatcher

from src.config import SKILLS_TAXONOMY_PATH
from src.extraction.nlp_loader import get_nlp


@lru_cache(maxsize=1)
def _load_skills() -> list[str]:
    data = json.loads(SKILLS_TAXONOMY_PATH.read_text())
    seen: dict[str, str] = {}
    for items in data.values():
        for item in items:
            seen.setdefault(item.lower(), item)
    return sorted(seen.values())


@lru_cache(maxsize=1)
def _get_matcher() -> PhraseMatcher:
    nlp = get_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in _load_skills()]
    matcher.add("SKILL", patterns)
    return matcher


def extract_skills(text: str) -> list[str]:
    """Return skills from the taxonomy that appear in *text*, deduplicated."""
    if not text:
        return []

    nlp = get_nlp()
    doc = nlp.make_doc(text)  # tokenizer only — matcher doesn't need tags/ner
    matcher = _get_matcher()

    canonical_by_lower = {s.lower(): s for s in _load_skills()}
    found: set[str] = set()
    for _match_id, start, end in matcher(doc):
        lower = doc[start:end].text.lower()
        found.add(canonical_by_lower.get(lower, doc[start:end].text))

    return sorted(found, key=str.lower)
