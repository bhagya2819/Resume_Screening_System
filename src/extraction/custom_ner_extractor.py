"""Extract SKILL / DEGREE / JOB_TITLE / YOE using the custom-trained NER model.

This module is only used when config.USE_CUSTOM_NER is True (or when
entity_extractor.extract_entities() is called with use_custom_ner=True).
"""
from __future__ import annotations

import re
from functools import lru_cache

import spacy
from spacy.language import Language

from src.config import CUSTOM_NER_MODEL_DIR
from src.extraction.education_extractor import (
    EducationMatch,
    highest_tier,
    tier_for_degree,
)


class CustomNERNotTrained(FileNotFoundError):
    pass


@lru_cache(maxsize=1)
def get_custom_nlp() -> Language:
    model_path = CUSTOM_NER_MODEL_DIR / "model-best"
    if not model_path.exists():
        raise CustomNERNotTrained(
            f"Custom NER model not found at {model_path}. "
            f"Train it via `python -m src.training.train_ner` first."
        )
    return spacy.load(model_path)


_YOE_NUMBER_RE = re.compile(r"\d{1,2}")


def _parse_yoe_spans(spans: list[str]) -> int | None:
    """Convert a list of YOE strings like '5 years', '3+ yrs' into the max int."""
    values: list[int] = []
    for raw in spans:
        match = _YOE_NUMBER_RE.search(raw)
        if match:
            try:
                v = int(match.group(0))
                if 0 < v <= 50:
                    values.append(v)
            except ValueError:
                continue
    return max(values) if values else None


def _to_education_matches(spans: list[str]) -> list[EducationMatch]:
    """Map degree strings to EducationMatch using the taxonomy for tier lookup."""
    seen: dict[str, EducationMatch] = {}
    for raw in spans:
        key = raw.strip().lower()
        if not key or key in seen:
            continue
        tier = tier_for_degree(raw) or "certification"
        seen[key] = EducationMatch(name=raw.strip(), tier=tier)
    return sorted(seen.values(), key=lambda m: (-m.rank, m.name.lower()))


def extract_all_entities(text: str) -> dict:
    """Run the custom NER model and return entity lists grouped by our schema.

    Returns a dict with keys:
        skills: list[str]
        degrees: list[EducationMatch]
        titles: list[str]
        yoe: int | None
    """
    if not text:
        return {"skills": [], "degrees": [], "titles": [], "yoe": None}

    nlp = get_custom_nlp()
    doc = nlp(text)

    raw: dict[str, list[str]] = {"SKILL": [], "DEGREE": [], "JOB_TITLE": [], "YOE": []}
    for ent in doc.ents:
        if ent.label_ in raw:
            raw[ent.label_].append(ent.text.strip())

    def _dedupe_case_insensitive(items: list[str]) -> list[str]:
        seen: dict[str, str] = {}
        for s in items:
            seen.setdefault(s.lower(), s)
        return sorted(seen.values(), key=str.lower)

    return {
        "skills": _dedupe_case_insensitive(raw["SKILL"]),
        "degrees": _to_education_matches(raw["DEGREE"]),
        "titles": _dedupe_case_insensitive(raw["JOB_TITLE"]),
        "yoe": _parse_yoe_spans(raw["YOE"]),
    }


def highest_degree_tier(matches: list[EducationMatch]) -> str | None:
    return highest_tier(matches)
