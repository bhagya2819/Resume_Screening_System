"""Extract job titles held, via PhraseMatcher against the titles taxonomy."""
from __future__ import annotations

import json
from functools import lru_cache

from spacy.matcher import PhraseMatcher

from src.config import JOB_TITLES_TAXONOMY_PATH
from src.extraction.nlp_loader import get_nlp


@lru_cache(maxsize=1)
def _load_titles() -> list[str]:
    data = json.loads(JOB_TITLES_TAXONOMY_PATH.read_text())
    seen: dict[str, str] = {}
    for items in data.values():
        for item in items:
            seen.setdefault(item.lower(), item)
    # Sort longest-first so matches prefer more specific titles ("Senior Data
    # Scientist" over "Data Scientist") when PhraseMatcher emits both.
    return sorted(seen.values(), key=lambda s: (-len(s), s.lower()))


@lru_cache(maxsize=1)
def _get_matcher() -> PhraseMatcher:
    nlp = get_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(title) for title in _load_titles()]
    matcher.add("TITLE", patterns)
    return matcher


def extract_titles(text: str) -> list[str]:
    if not text:
        return []

    nlp = get_nlp()
    doc = nlp.make_doc(text)
    matcher = _get_matcher()

    spans = [(start, end) for _id, start, end in matcher(doc)]
    # Greedy longest-match pass: keep only spans that are not contained in a longer one.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept: list[tuple[int, int]] = []
    for start, end in spans:
        if any(k_start <= start and end <= k_end for k_start, k_end in kept):
            continue
        kept.append((start, end))

    canonical_by_lower = {t.lower(): t for t in _load_titles()}
    found: set[str] = set()
    for start, end in kept:
        span_text = doc[start:end].text
        found.add(canonical_by_lower.get(span_text.lower(), span_text))

    return sorted(found, key=str.lower)
