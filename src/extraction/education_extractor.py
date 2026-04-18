"""Extract degrees / educational credentials, tagged with a tier."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from spacy.matcher import PhraseMatcher

from src.config import DEGREES_TAXONOMY_PATH
from src.extraction.nlp_loader import get_nlp


EDUCATION_TIERS = ("doctorate", "masters", "bachelors", "associate", "certification")
_TIER_RANK = {tier: i for i, tier in enumerate(reversed(EDUCATION_TIERS))}  # higher = more advanced


@dataclass(frozen=True)
class EducationMatch:
    name: str
    tier: str

    @property
    def rank(self) -> int:
        return _TIER_RANK.get(self.tier, -1)


@lru_cache(maxsize=1)
def _load_degrees() -> dict[str, str]:
    """Return {canonical_name_lower: tier}."""
    data = json.loads(DEGREES_TAXONOMY_PATH.read_text())
    mapping: dict[str, str] = {}
    for tier, items in data.items():
        for item in items:
            mapping.setdefault(item.lower(), tier)
    return mapping


@lru_cache(maxsize=1)
def _get_matcher() -> PhraseMatcher:
    nlp = get_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    by_tier: dict[str, list] = {tier: [] for tier in EDUCATION_TIERS}
    data = json.loads(DEGREES_TAXONOMY_PATH.read_text())
    for tier, items in data.items():
        for item in items:
            by_tier[tier].append(nlp.make_doc(item))
    for tier, patterns in by_tier.items():
        if patterns:
            matcher.add(tier.upper(), patterns)
    return matcher


def extract_education(text: str) -> list[EducationMatch]:
    """Return deduplicated degree matches with tier info."""
    if not text:
        return []

    nlp = get_nlp()
    doc = nlp.make_doc(text)
    matcher = _get_matcher()
    tier_from = _load_degrees()

    seen: dict[str, EducationMatch] = {}
    for match_id, start, end in matcher(doc):
        span_text = doc[start:end].text
        tier = tier_from.get(span_text.lower()) or nlp.vocab.strings[match_id].lower()
        key = span_text.lower()
        if key not in seen:
            seen[key] = EducationMatch(name=span_text, tier=tier)

    return sorted(seen.values(), key=lambda m: (-m.rank, m.name.lower()))


def highest_tier(matches: list[EducationMatch]) -> str | None:
    if not matches:
        return None
    return max(matches, key=lambda m: m.rank).tier
